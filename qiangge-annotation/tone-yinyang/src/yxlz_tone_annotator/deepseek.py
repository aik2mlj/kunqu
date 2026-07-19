from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class DeepSeekApiError(RuntimeError):
    """DeepSeek 请求失败或返回不可用结果。"""


class DeepSeekOutputTruncatedError(DeepSeekApiError):
    """输出长度不足；重试相同请求不会改变结果，必须调整请求参数。"""


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    api_base: str
    model: str
    timeout_seconds: float
    max_retries: int
    retry_base_seconds: float
    max_tokens: int
    temperature: float
    thinking_enabled: bool
    reasoning_effort: str


def _extract_json_object(content: str) -> dict[str, Any]:
    """兼容偶发代码围栏；最终仍要求根节点必须是 JSON 对象。"""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise DeepSeekApiError(f"模型返回的内容不是合法 JSON：{error}") from error
    if not isinstance(value, dict):
        raise DeepSeekApiError("模型返回的 JSON 根节点不是对象。")
    return value


def request_json(
    config: DeepSeekConfig,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """调用 Chat Completions，并对限流、空结果和临时服务器错误重试。"""
    endpoint = f"{config.api_base.rstrip('/')}/chat/completions"
    request_body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled" if config.thinking_enabled else "disabled"},
    }
    if config.thinking_enabled:
        request_body["reasoning_effort"] = config.reasoning_effort

    encoded_body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=encoded_body,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                response_value = json.loads(response.read().decode("utf-8"))
            choices = response_value.get("choices")
            if not isinstance(choices, list) or not choices:
                raise DeepSeekApiError("DeepSeek 响应中没有 choices。")
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                raise DeepSeekOutputTruncatedError(
                    "模型输出被 max_tokens 截断。请减小 --batch-lines / --batch-characters，"
                    "或使用 --thinking disabled。"
                )
            if finish_reason not in ("stop", None):
                raise DeepSeekApiError(f"模型异常结束：{finish_reason}")
            content = choice.get("message", {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise DeepSeekApiError("模型返回了空内容。")
            parsed = _extract_json_object(content)
            metadata = {
                "model": response_value.get("model", config.model),
                "finishReason": finish_reason,
                "usage": response_value.get("usage"),
            }
            return parsed, metadata
        except DeepSeekOutputTruncatedError:
            # 相同提示、相同批次和相同 max_tokens 的重试必然再次截断，不能继续消耗额度。
            raise
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            last_error = DeepSeekApiError(f"DeepSeek HTTP {error.code}：{error_body[:1000]}")
            # 参数、鉴权等 4xx 不会因重试自行恢复；429 限流除外。
            if 400 <= error.code < 500 and error.code != 429:
                raise last_error from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, DeepSeekApiError) as error:
            last_error = error

        if attempt >= config.max_retries:
            break
        delay = config.retry_base_seconds * (2**attempt) + random.uniform(0, 0.5)
        print(f"  请求失败，{delay:.1f} 秒后进行第 {attempt + 2} 次尝试：{last_error}")
        time.sleep(delay)

    raise DeepSeekApiError(f"重试后仍未取得有效结果：{last_error}")
