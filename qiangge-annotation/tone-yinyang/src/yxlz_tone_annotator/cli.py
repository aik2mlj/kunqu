#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .deepseek import DeepSeekConfig
from .pipeline import (
    AnnotationDataError,
    PipelineConfig,
    load_dotenv,
    run_pipeline,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.tone-ai-draft{input_path.suffix}")


def default_review_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.tone-ai-review.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用 DeepSeek，按《韵学骊珠》体系批量标注内建逐字轨四声。",
    )
    parser.add_argument("input", type=Path, help="标注工具导出的 annotation.json")
    parser.add_argument("--output", type=Path, help="AI 草稿项目输出路径")
    parser.add_argument("--review-output", type=Path, help="复核报告输出路径")
    parser.add_argument("--reference", type=Path, help="机器可读《韵学骊珠》字表 JSON")
    parser.add_argument("--model", help="DeepSeek 模型名；默认读取 DEEPSEEK_MODEL")
    parser.add_argument("--api-base", help="API 根地址；默认读取 DEEPSEEK_API_BASE")
    parser.add_argument("--batch-lines", type=int, default=3, help="每批最多句数，默认 3")
    parser.add_argument("--batch-characters", type=int, default=24, help="每批最多字块数，默认 24")
    parser.add_argument("--confidence-threshold", type=float, default=0.80, help="低于此值进入复核")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度，默认 0.1")
    parser.add_argument("--max-tokens", type=int, default=4096, help="单批最大输出 token")
    parser.add_argument("--timeout", type=float, default=240, help="单次请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=5, help="临时错误最大重试次数")
    parser.add_argument("--retry-base", type=float, default=2.0, help="指数退避基础秒数")
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default="disabled",
        help="是否启用 thinking，默认 disabled；全量批处理建议关闭",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("high", "max"),
        default="high",
        help="推理强度，默认 high",
    )
    parser.add_argument("--overwrite-existing", action="store_true", help="覆盖已经存在的 tone")
    parser.add_argument(
        "--omit-low-confidence",
        action="store_true",
        help="低置信或模型主动标记复核的项目在草稿中仍保留 tone: null",
    )
    parser.add_argument("--restart", action="store_true", help="清空该输入文件的断点缓存并重跑")
    parser.add_argument("--dry-run", action="store_true", help="只检查和分批，不调用 API")
    return parser.parse_args()


def main() -> int:
    # 团队成员既可以在子项目目录放置 .env，也可以从任意工作目录临时覆盖配置。
    # load_dotenv 使用 setdefault，因此显式导出的环境变量始终拥有最高优先级。
    load_dotenv(PROJECT_DIRECTORY / ".env")
    load_dotenv(Path.cwd() / ".env")
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"错误：输入文件不存在：{input_path}", file=sys.stderr)
        return 2
    if args.batch_lines <= 0 or args.batch_characters <= 0:
        print("错误：批次句数和字数必须大于 0。", file=sys.stderr)
        return 2
    if not 0 <= args.confidence_threshold <= 1:
        print("错误：confidence-threshold 必须在 0 到 1 之间。", file=sys.stderr)
        return 2

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        print("错误：请在 .env 或环境变量中设置 DEEPSEEK_API_KEY。", file=sys.stderr)
        return 2
    output_path = (args.output or default_output_path(input_path)).expanduser().resolve()
    review_path = (args.review_output or default_review_path(input_path)).expanduser().resolve()
    reference_path = args.reference.expanduser().resolve() if args.reference else None

    pipeline_config = PipelineConfig(
        input_path=input_path,
        output_path=output_path,
        review_path=review_path,
        reference_path=reference_path,
        batch_lines=args.batch_lines,
        batch_characters=args.batch_characters,
        confidence_threshold=args.confidence_threshold,
        overwrite_existing=args.overwrite_existing,
        omit_low_confidence=args.omit_low_confidence,
        restart=args.restart,
        dry_run=args.dry_run,
    )
    deepseek_config = DeepSeekConfig(
        api_key=api_key,
        api_base=args.api_base or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        model=args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking_enabled=args.thinking == "enabled",
        reasoning_effort=args.reasoning_effort,
    )
    try:
        run_pipeline(pipeline_config, deepseek_config)
    except (AnnotationDataError, RuntimeError) as error:
        print(f"失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
