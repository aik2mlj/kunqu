from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .deepseek import DeepSeekConfig, request_json
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt


TONE_CLASSES = {
    "yin_ping",
    "yang_ping",
    "yin_shang",
    "yang_shang",
    "yin_qu",
    "yang_qu",
    "yin_ru",
    "yang_ru",
}
SHANG_SUBTYPES = {"yin_shang", "yang_shang", "yinyang_tongyong"}
HAN_CHARACTER_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class AnnotationDataError(RuntimeError):
    """输入、模型结果或回写数据违反标注结构。"""


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_path: Path
    review_path: Path
    reference_path: Path | None
    batch_lines: int
    batch_characters: int
    confidence_threshold: float
    overwrite_existing: bool
    omit_low_confidence: bool
    restart: bool
    dry_run: bool


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationDataError(f"无法读取 JSON：{path}：{error}") from error


def write_json_atomic(path: Path, value: Any) -> None:
    """先写临时文件再替换，避免中断时留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def load_dotenv(path: Path) -> None:
    """只补充尚未设置的环境变量，命令行环境始终优先。"""
    import os

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_lookup_character(raw_character: str) -> tuple[str, list[str]]:
    matches = HAN_CHARACTER_PATTERN.findall(raw_character)
    warnings: list[str] = []
    if not matches:
        raise AnnotationDataError(f"逐字块 {raw_character!r} 中没有可标注的汉字。")
    if len(matches) > 1:
        warnings.append(f"块内容 {raw_character!r} 含多个汉字，暂以首字 {matches[0]!r} 查音。")
    elif raw_character != matches[0]:
        warnings.append(f"块内容 {raw_character!r} 含标点，查音字为 {matches[0]!r}。")
    return matches[0], warnings


def _load_reference_entries(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    value = read_json(path)
    entries = value.get("entries") if isinstance(value, dict) else value
    if not isinstance(entries, list):
        raise AnnotationDataError("参考资料 JSON 必须是数组，或包含 entries 数组。")
    return [entry for entry in entries if isinstance(entry, dict)]


def _reference_index(entries: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        keys: list[str] = []
        if isinstance(entry.get("character"), str):
            keys.append(entry["character"])
        if isinstance(entry.get("字"), str):
            keys.append(entry["字"])
        variants = entry.get("lookupVariants")
        if isinstance(variants, list):
            keys.extend(value for value in variants if isinstance(value, str))
        for key in set(keys):
            result.setdefault(key, []).append(entry)
    return result


def _project_parts(document: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(document, dict):
        raise AnnotationDataError("项目 JSON 根节点必须是对象。")
    project = document.get("project") if isinstance(document.get("project"), dict) else document
    characters = project.get("characterAnnotations")
    lines = project.get("subtitleLines")
    if not isinstance(characters, list) or not isinstance(lines, list):
        raise AnnotationDataError("项目中缺少 characterAnnotations 或 subtitleLines 数组。")
    valid_characters = [item for item in characters if isinstance(item, dict)]
    valid_lines = [item for item in lines if isinstance(item, dict)]
    return project, valid_characters, valid_lines


def build_line_payloads(
    document: Any,
    reference_entries: list[dict[str, Any]],
    overwrite_existing: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    _, characters, lines = _project_parts(document)
    line_by_id = {line.get("id"): line for line in lines if isinstance(line.get("id"), str)}
    reference_by_character = _reference_index(reference_entries)
    grouped: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []

    for character in characters:
        character_id = character.get("id")
        line_id = character.get("lineId")
        raw_character = character.get("char")
        if not all(isinstance(value, str) for value in (character_id, line_id, raw_character)):
            raise AnnotationDataError(f"逐字块缺少合法 id、lineId 或 char：{character}")
        if not overwrite_existing and character.get("tone") is not None:
            continue
        lookup_character, character_warnings = _extract_lookup_character(raw_character)
        warnings.extend(f"{character_id}: {message}" for message in character_warnings)
        grouped.setdefault(line_id, []).append(
            {
                "id": character_id,
                "char": raw_character,
                "lookupChar": lookup_character,
                # 这不是让模型自行声称“有依据”，而是由本地字表检索结果明确告知。
                # 回写校验也会使用该标志，防止无资料推断被标成高置信结论。
                "referenceAvailable": bool(reference_by_character.get(lookup_character)),
                "startTime": character.get("startTime"),
                "endTime": character.get("endTime"),
            }
        )

    payloads: list[dict[str, Any]] = []
    # 先遵循 subtitleLines 的原始顺序；孤立字块再按时间追加，避免静默遗漏。
    ordered_line_ids = [line.get("id") for line in lines if line.get("id") in grouped]
    ordered_line_ids.extend(line_id for line_id in grouped if line_id not in set(ordered_line_ids))
    for line_id in ordered_line_ids:
        line_characters = sorted(
            grouped[line_id],
            key=lambda item: (float(item.get("startTime") or 0), item["id"]),
        )
        for index, character in enumerate(line_characters):
            character["index"] = index
            character["previousChar"] = line_characters[index - 1]["char"] if index else None
            character["nextChar"] = (
                line_characters[index + 1]["char"] if index + 1 < len(line_characters) else None
            )
            character.pop("startTime", None)
            character.pop("endTime", None)
        lookup_characters = {item["lookupChar"] for item in line_characters}
        references = [
            entry
            for lookup_character in sorted(lookup_characters)
            for entry in reference_by_character.get(lookup_character, [])
        ]
        line = line_by_id.get(line_id, {})
        payloads.append(
            {
                "lineId": line_id,
                "text": line.get("text") if isinstance(line.get("text"), str) else "".join(
                    item["char"] for item in line_characters
                ),
                "characters": line_characters,
                "referenceEntries": references,
            }
        )
    return payloads, warnings


def build_batches(
    line_payloads: list[dict[str, Any]],
    max_lines: int,
    max_characters: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    current_lines: list[dict[str, Any]] = []
    current_character_count = 0

    def flush() -> None:
        nonlocal current_lines, current_character_count
        if current_lines:
            batches.append({"batchId": f"batch-{len(batches) + 1:04d}", "lines": current_lines})
        current_lines = []
        current_character_count = 0

    for line in line_payloads:
        line_character_count = len(line["characters"])
        if current_lines and (
            len(current_lines) >= max_lines
            or current_character_count + line_character_count > max_characters
        ):
            flush()
        current_lines.append(line)
        current_character_count += line_character_count
    flush()
    return batches


def _expected_characters(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        character["id"]: character
        for line in batch["lines"]
        for character in line["characters"]
    }


def validate_batch_result(
    batch: dict[str, Any],
    result: dict[str, Any],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    if result.get("batchId") != batch["batchId"]:
        raise AnnotationDataError(
            f"批次编号不一致：期望 {batch['batchId']}，实际 {result.get('batchId')}"
        )
    annotations = result.get("annotations")
    if not isinstance(annotations, list):
        raise AnnotationDataError(f"{batch['batchId']} 缺少 annotations 数组。")
    expected = _expected_characters(batch)
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []

    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise AnnotationDataError(f"{batch['batchId']} 含有非对象标注。")
        annotation_id = annotation.get("id")
        if annotation_id not in expected:
            raise AnnotationDataError(f"{batch['batchId']} 返回未知 id：{annotation_id}")
        if annotation_id in seen:
            raise AnnotationDataError(f"{batch['batchId']} 重复返回 id：{annotation_id}")
        seen.add(annotation_id)
        source = expected[annotation_id]
        if annotation.get("char") != source["char"]:
            raise AnnotationDataError(
                f"{annotation_id} 的 char 被模型改写：{source['char']!r} -> {annotation.get('char')!r}"
            )
        tone_class = annotation.get("toneClass")
        subtype = annotation.get("yxlzShangSubtype")
        if tone_class not in TONE_CLASSES:
            raise AnnotationDataError(f"{annotation_id} 的 toneClass 非法：{tone_class}")
        if tone_class in {"yin_shang", "yang_shang"}:
            if subtype not in SHANG_SUBTYPES:
                raise AnnotationDataError(f"{annotation_id} 是上声但 subtype 非法：{subtype}")
            if subtype == "yin_shang" and tone_class != "yin_shang":
                raise AnnotationDataError(f"{annotation_id} 的阴上 subtype 与 toneClass 冲突。")
            if subtype in {"yang_shang", "yinyang_tongyong"} and tone_class != "yang_shang":
                raise AnnotationDataError(f"{annotation_id} 的阳上 subtype 与 toneClass 冲突。")
        elif subtype is not None:
            raise AnnotationDataError(f"{annotation_id} 不是上声却携带 subtype。")
        confidence = annotation.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise AnnotationDataError(f"{annotation_id} 的 confidence 非法：{confidence}")
        basis = annotation.get("basis") if isinstance(annotation.get("basis"), dict) else {}
        if not source.get("referenceAvailable", False):
            # 没有本地提供的可核查《韵学骊珠》条目时，模型的任何判断都是语境推断。
            # 即使模型自行写了“韵学骊珠”，也不能把它当成可审计的直接引文。
            basis["source"] = "context_inference"
            annotation["basis"] = basis
            annotation["confidence"] = min(float(confidence), 0.79)
            annotation["needsReview"] = True
        elif basis.get("source") == "context_inference" and float(confidence) > 0.79:
            annotation["confidence"] = 0.79
        annotation["needsReview"] = bool(annotation.get("needsReview")) or (
            float(annotation["confidence"]) < confidence_threshold
        )
        validated.append(annotation)

    missing = set(expected) - seen
    if missing:
        raise AnnotationDataError(f"{batch['batchId']} 遗漏 {len(missing)} 个 id：{sorted(missing)[:10]}")
    return validated


def _work_directory(input_path: Path) -> Path:
    return input_path.parent / f".{input_path.stem}.tone-ai-work"


def _manifest_signature(
    pipeline: PipelineConfig,
    deepseek: DeepSeekConfig,
    input_hash: str,
) -> dict[str, Any]:
    return {
        "promptVersion": PROMPT_VERSION,
        "inputSha256": input_hash,
        "model": deepseek.model,
        "thinkingEnabled": deepseek.thinking_enabled,
        "reasoningEffort": deepseek.reasoning_effort,
        "temperature": deepseek.temperature,
        "batchLines": pipeline.batch_lines,
        "batchCharacters": pipeline.batch_characters,
        "overwriteExisting": pipeline.overwrite_existing,
        "referenceSha256": _sha256(pipeline.reference_path) if pipeline.reference_path else None,
    }


def _check_or_create_manifest(work_dir: Path, signature: dict[str, Any], restart: bool) -> None:
    if restart and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing != signature:
            raise AnnotationDataError(
                "已有断点缓存与本次输入或参数不同。请使用 --restart 重新开始，"
                "或恢复上次的模型、批大小及参考资料参数。"
            )
    else:
        write_json_atomic(manifest_path, signature)


def _tone_for_project(annotation: dict[str, Any]) -> dict[str, Any]:
    tone = {"toneClass": annotation["toneClass"]}
    if annotation["toneClass"] in {"yin_shang", "yang_shang"}:
        tone["yxlzShangSubtype"] = annotation["yxlzShangSubtype"]
    return tone


def _build_same_glyph_conflicts(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """列出同一原字形被赋予不同调类的情况，供人工优先检查多音或模型不一致。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation["char"], []).append(annotation)

    conflicts: list[dict[str, Any]] = []
    for character, items in grouped.items():
        tone_keys = {
            (item["toneClass"], item.get("yxlzShangSubtype"))
            for item in items
        }
        if len(tone_keys) < 2:
            continue
        conflicts.append(
            {
                "char": character,
                "toneOptions": [
                    {
                        "toneClass": tone_class,
                        "yxlzShangSubtype": subtype,
                    }
                    for tone_class, subtype in sorted(tone_keys)
                ],
                "occurrences": [
                    {
                        "id": item["id"],
                        "toneClass": item["toneClass"],
                        "yxlzShangSubtype": item.get("yxlzShangSubtype"),
                        "confidence": item["confidence"],
                    }
                    for item in items
                ],
            }
        )
    return conflicts


def run_pipeline(pipeline: PipelineConfig, deepseek: DeepSeekConfig) -> None:
    source_document = read_json(pipeline.input_path)
    reference_entries = _load_reference_entries(pipeline.reference_path)
    line_payloads, preprocessing_warnings = build_line_payloads(
        source_document,
        reference_entries,
        pipeline.overwrite_existing,
    )
    batches = build_batches(line_payloads, pipeline.batch_lines, pipeline.batch_characters)
    target_count = sum(len(line["characters"]) for line in line_payloads)
    print(f"输入：{pipeline.input_path}")
    print(f"待标注字块：{target_count}；句：{len(line_payloads)}；批次：{len(batches)}")
    for warning in preprocessing_warnings:
        print(f"预处理提示：{warning}")
    if pipeline.dry_run:
        print("dry-run 完成：未调用 API，也未写入结果文件。")
        return
    if target_count == 0:
        print("没有需要标注的字块。若要覆盖已有四声，请添加 --overwrite-existing。")
        return

    work_dir = _work_directory(pipeline.input_path)
    signature = _manifest_signature(pipeline, deepseek, _sha256(pipeline.input_path))
    _check_or_create_manifest(work_dir, signature, pipeline.restart)
    all_annotations: list[dict[str, Any]] = []
    api_batches: list[dict[str, Any]] = []

    for index, batch in enumerate(batches, start=1):
        request_path = work_dir / f"{batch['batchId']}.request.json"
        result_path = work_dir / f"{batch['batchId']}.result.json"
        metadata_path = work_dir / f"{batch['batchId']}.metadata.json"
        write_json_atomic(request_path, batch)
        if result_path.exists():
            result = read_json(result_path)
            metadata = read_json(metadata_path) if metadata_path.exists() else {"resumed": True}
            print(f"[{index}/{len(batches)}] 复用 {batch['batchId']} 已完成结果")
        else:
            print(f"[{index}/{len(batches)}] 请求 {batch['batchId']} ...")
            result, metadata = request_json(deepseek, SYSTEM_PROMPT, build_user_prompt(batch))
            # 先严格校验，合法后才落为可复用缓存。
            validate_batch_result(batch, result, pipeline.confidence_threshold)
            write_json_atomic(result_path, result)
            write_json_atomic(metadata_path, metadata)
        validated = validate_batch_result(batch, result, pipeline.confidence_threshold)
        all_annotations.extend(validated)
        api_batches.append({"batchId": batch["batchId"], **metadata})

    output_document = copy.deepcopy(source_document)
    _, output_characters, _ = _project_parts(output_document)
    result_by_id = {annotation["id"]: annotation for annotation in all_annotations}
    written_count = 0
    omitted_count = 0
    for character in output_characters:
        annotation = result_by_id.get(character.get("id"))
        if annotation is None:
            continue
        if pipeline.omit_low_confidence and annotation["needsReview"]:
            character["tone"] = None
            omitted_count += 1
        else:
            character["tone"] = _tone_for_project(annotation)
            written_count += 1

    review_document = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "inputFile": pipeline.input_path.name,
        "outputFile": pipeline.output_path.name,
        "promptVersion": PROMPT_VERSION,
        "model": deepseek.model,
        "summary": {
            "targetCount": target_count,
            "writtenCount": written_count,
            "omittedLowConfidenceCount": omitted_count,
            "needsReviewCount": sum(bool(item["needsReview"]) for item in all_annotations),
            "batchCount": len(batches),
        },
        "preprocessingWarnings": preprocessing_warnings,
        "annotations": all_annotations,
        "sameGlyphConflicts": _build_same_glyph_conflicts(all_annotations),
        "apiBatches": api_batches,
    }
    write_json_atomic(pipeline.output_path, output_document)
    write_json_atomic(pipeline.review_path, review_document)
    print(f"完成：已写入 {written_count} 个四声标注。")
    print(f"草稿项目：{pipeline.output_path}")
    print(f"复核报告：{pipeline.review_path}")
    print(f"需人工复核：{review_document['summary']['needsReviewCount']} 个")
