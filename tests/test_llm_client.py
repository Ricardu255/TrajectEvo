import email.message
import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from evoagent.llm import JsonChatClient
from evoagent.telemetry import ExecutionLedger


def http_error(code, body=b"error", headers=None):
    return urllib.error.HTTPError(
        "https://llm.test/v1/chat/completions", code, "error", headers,
        io.BytesIO(body),
    )


def chat_response(content, finish_reason="stop"):
    payload = {
        "choices": [
            {"message": {"content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
    return response


class JsonChatClientTests(unittest.TestCase):
    def make_client(self, retries=2):
        return JsonChatClient(
            "https://llm.test/v1", "key", "test-model", "test",
            retries=retries, backoff_seconds=0.0,
        )

    def test_retries_transient_http_errors_then_succeeds(self):
        client = self.make_client()
        ledger = ExecutionLedger("test")
        responses = [
            http_error(429), http_error(503), chat_response('{"ok": true}'),
        ]
        with patch("evoagent.llm.urllib.request.urlopen", side_effect=responses):
            result = client.complete_json("role", "system", "user", ledger)
        self.assertEqual({"ok": True}, result)
        self.assertEqual(3, len(ledger.model_calls))
        self.assertEqual(2, sum(1 for item in ledger.model_calls if not item.ok))
        self.assertTrue(ledger.model_calls[-1].ok)
        self.assertEqual(15, ledger.model_calls[-1].input_tokens + ledger.model_calls[-1].output_tokens)

    def test_no_retry_on_client_error(self):
        client = self.make_client(retries=3)
        ledger = ExecutionLedger("test")
        with patch("evoagent.llm.urllib.request.urlopen", side_effect=[http_error(401)]):
            with self.assertRaises(RuntimeError) as ctx:
                client.complete_json("role", "system", "user", ledger)
        self.assertIn("HTTP 401", str(ctx.exception))
        self.assertEqual(1, len(ledger.model_calls))
        self.assertFalse(ledger.model_calls[0].ok)

    def test_retries_network_errors(self):
        client = self.make_client(retries=1)
        with patch(
            "evoagent.llm.urllib.request.urlopen",
            side_effect=[
                urllib.error.URLError("connection refused"), chat_response('{"ok": 1}'),
            ],
        ):
            self.assertEqual({"ok": 1}, client.complete_json("role", "system", "user"))

    def test_truncated_output_reports_clear_error(self):
        client = self.make_client()
        with patch(
            "evoagent.llm.urllib.request.urlopen",
            side_effect=[chat_response('{"findings": [', finish_reason="length")],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                client.complete_json("role", "system", "user")
        self.assertIn("truncated", str(ctx.exception))

    def test_retry_after_header_overrides_backoff(self):
        client = self.make_client(retries=1)
        headers = email.message.Message()
        headers["Retry-After"] = "2"
        responses = [http_error(429, headers=headers), chat_response('{"ok": 1}')]
        with patch("evoagent.llm.urllib.request.urlopen", side_effect=responses):
            with patch("evoagent.llm.time.sleep") as sleeper:
                self.assertEqual({"ok": 1}, client.complete_json("role", "system", "user"))
        sleeper.assert_called_once_with(2.0)

    def test_exhausted_retries_raise_last_error(self):
        client = self.make_client(retries=1)
        with patch(
            "evoagent.llm.urllib.request.urlopen",
            side_effect=[http_error(503), http_error(503)],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                client.complete_json("role", "system", "user")
        self.assertIn("HTTP 503", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
