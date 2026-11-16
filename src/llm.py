"""Gọi LLM cho agent qua API tương thích OpenAI: OpenRouter hoặc Gemini (Google AI Studio).

- Đọc LLM_PROVIDER / LLM_MODEL / OPENROUTER_API_KEY hoặc GEMINI_API_KEY từ .env (không bao giờ in key).
- temperature=0 để dễ tái lập.
- Retry có backoff cho lỗi tạm thời (429 quá tải, 5xx, timeout); hết lượt thì raise LLMError — không có fallback giả.
- Chặn model trả phí nếu chưa đặt ALLOW_PAID_MODEL=1 (tài khoản free tier, tránh phát sinh nợ).
- Ghi usage/cost do OpenRouter trả về cho từng lượt gọi.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.retrievers import load_env

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}
# provider -> (base_url tương thích OpenAI, tên biến key trong .env)
PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}


class LLMError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, attempts: list[dict] | None = None):
        super().__init__(message)
        self.status, self.attempts = status, attempts or []


@dataclass
class LLMResult:
    message: Any                 # openai ChatCompletionMessage
    finish_reason: str | None
    served_model: str | None
    usage: dict
    latency_s: float
    attempts: list[dict] = field(default_factory=list)


class OpenRouterLLM:
    def __init__(self, model: str | None = None, *, temperature: float = 0.0, max_tokens: int = 1200,
                 max_retries: int = 4, backoff_s: tuple[float, ...] = (5, 15, 30, 60), timeout_s: float = 120):
        from openai import OpenAI

        env = load_env()
        self.provider = (env.get("LLM_PROVIDER") or "openrouter").lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER phải là một trong {sorted(PROVIDERS)}")
        base_url, key_name = PROVIDERS[self.provider]
        key = env.get(key_name) or ""
        if not key:
            raise ValueError(f"Thiếu {key_name} trong .env")
        self.model = model or env.get("LLM_MODEL")
        if not self.model:
            raise ValueError("Thiếu LLM_MODEL trong .env")
        # OpenRouter: model không có đuôi :free sẽ trừ credit -> chặn nếu chưa cho phép.
        # Gemini: key free tier (chưa bật billing) không phát sinh phí; vượt hạn mức sẽ nhận 429.
        if self.provider == "openrouter" and not self.model.endswith(":free") and env.get("ALLOW_PAID_MODEL") != "1":
            raise ValueError(f"Model '{self.model}' là model trả phí. Đặt ALLOW_PAID_MODEL=1 trong .env nếu chắc chắn muốn dùng.")
        self.client = OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=timeout_s)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.max_retries, self.backoff_s = max_retries, backoff_s

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | None = None) -> LLMResult:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        attempts: list[dict] = []
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature,
                                  "max_tokens": self.max_tokens}
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        for attempt in range(self.max_retries + 1):
            t = time.perf_counter()
            try:
                r = self.client.chat.completions.create(**kwargs)
            except APIStatusError as e:
                info = {"attempt": attempt + 1, "status": e.status_code, "error": e.response.text[:300],
                        "elapsed_s": round(time.perf_counter() - t, 2)}
                attempts.append(info)
                if e.status_code in RETRYABLE and attempt < self.max_retries:
                    time.sleep(self.backoff_s[min(attempt, len(self.backoff_s) - 1)])
                    continue
                raise LLMError(f"HTTP {e.status_code}", status=e.status_code, attempts=attempts) from e
            except (APIConnectionError, APITimeoutError) as e:
                attempts.append({"attempt": attempt + 1, "status": None, "error": f"{type(e).__name__}: {e}"[:300],
                                 "elapsed_s": round(time.perf_counter() - t, 2)})
                if attempt < self.max_retries:
                    time.sleep(self.backoff_s[min(attempt, len(self.backoff_s) - 1)])
                    continue
                raise LLMError(type(e).__name__, attempts=attempts) from e
            if not r.choices:  # một số provider trả 200 nhưng body là lỗi
                err = getattr(r, "error", None) or "empty choices"
                attempts.append({"attempt": attempt + 1, "status": 200, "error": str(err)[:300]})
                if attempt < self.max_retries:
                    time.sleep(self.backoff_s[min(attempt, len(self.backoff_s) - 1)])
                    continue
                raise LLMError(f"Phản hồi rỗng: {err}", attempts=attempts)
            usage = r.usage.model_dump() if r.usage else {}
            return LLMResult(message=r.choices[0].message, finish_reason=r.choices[0].finish_reason,
                             served_model=r.model, latency_s=round(time.perf_counter() - t, 2), attempts=attempts,
                             usage={"prompt_tokens": usage.get("prompt_tokens"),
                                    "completion_tokens": usage.get("completion_tokens"),
                                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                    "cost_usd": usage.get("cost")})
        raise LLMError("hết lượt retry", attempts=attempts)
