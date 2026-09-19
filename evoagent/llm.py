"""Small OpenAI-compatible JSON client with auditable usage accounting."""
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .telemetry import ExecutionLedger


# Transient server/rate-limit failures worth retrying; other 4xx responses
# are deterministic client errors (bad key, unsupported parameter, ...).
RETRYABLE_HTTP_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_RETRY_DELAY_SECONDS = 8.0
MAX_RETRY_AFTER_SECONDS = 30.0


class JsonChatClient:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        provider: str = "openai-compatible", timeout: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
        retries: int = 2, backoff_seconds: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.retries = max(0, int(retries))
        self.backoff_seconds = max(0.0, float(backoff_seconds))

    def complete_json(
        self, role: str, system: str, user: str,
        ledger: Optional[ExecutionLedger] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.extra_headers)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_message = ""
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            delay = 0.0
            retryable = False
            request = urllib.request.Request(
                self.base_url + "/chat/completions", data=data,
                headers=headers, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content, finish_reason = self._extract_content(body)
                try:
                    result = json.loads(content)
                    if not isinstance(result, dict):
                        raise ValueError("model JSON root is not an object")
                except (TypeError, ValueError) as exc:
                    hint = ""
                    if finish_reason == "length":
                        hint = (
                            " (output truncated at the token limit; "
                            "raise the token budget or shrink the context)"
                        )
                    last_message = "%s returned an invalid JSON response%s: %s" % (
                        self.provider, hint, exc,
                    )
                else:
                    if ledger:
                        ledger.record_model(
                            role, self.provider, self.model, body.get("usage") or {},
                            int((time.monotonic() - started) * 1000), True,
                        )
                    return result
            except urllib.error.HTTPError as exc:
                detail = exc.read(1000).decode("utf-8", errors="replace")
                last_message = "%s API returned HTTP %d: %s" % (
                    self.provider, exc.code, detail,
                )
                retryable = exc.code in RETRYABLE_HTTP_CODES
                if retryable:
                    delay = self._retry_delay(attempt, self._retry_after(exc))
            except (urllib.error.URLError, socket.timeout) as exc:
                last_message = "%s JSON request failed: %s" % (self.provider, exc)
                retryable = True
                delay = self._retry_delay(attempt)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                # Malformed HTTP body or response envelope: not transient.
                last_message = "%s JSON request failed: %s" % (self.provider, exc)
            if ledger:
                ledger.record_model(
                    role, self.provider, self.model, {},
                    int((time.monotonic() - started) * 1000), False, last_message,
                )
            if not retryable or attempt >= self.retries:
                break
            time.sleep(delay)
        raise RuntimeError(last_message)

    @staticmethod
    def _extract_content(body: Dict[str, Any]) -> Tuple[Optional[str], str]:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return message.get("content"), str(choice.get("finish_reason") or "")

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
        headers = getattr(exc, "headers", None)
        raw = headers.get("Retry-After") if headers else None
        if not raw:
            return None
        try:
            return min(MAX_RETRY_AFTER_SECONDS, max(0.0, float(raw)))
        except (TypeError, ValueError):
            return None

    def _retry_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None:
            return retry_after
        return min(MAX_RETRY_DELAY_SECONDS, self.backoff_seconds * (2 ** attempt))
