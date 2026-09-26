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
        self.assertIn("finish_reason=length", str(ctx.exception))

    def test_null_content_response_is_retried(self):
        client = self.make_client()
        responses = [
            chat_response(None, finish_reason="stop"),
            chat_response('{"ok": 1}'),
        ]
        with patch("evoagent.llm.urllib.request.urlopen", side_effect=responses):
            self.assertEqual({"ok": 1}, client.complete_json("role", "system", "user"))

    def test_length_truncation_retries_with_doubled_budget(self):
        client = self.make_client()
        captured = []

        def capture_urlopen(request, timeout=None):
            captured.append(json.loads(request.data.decode("utf-8")))
            if len(captured) == 1:
                return chat_response("", finish_reason="length")
            return chat_response('{"ok": 1}')

        with patch("evoagent.llm.urllib.request.urlopen", side_effect=capture_urlopen):
            self.assertEqual(
                {"ok": 1}, client.complete_json("role", "system", "user", max_tokens=4000)
            )
        self.assertEqual(4000, captured[0]["max_tokens"])
        self.assertEqual(8000, captured[1]["max_tokens"])

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


class ParseModelJsonTests(unittest.TestCase):
    def _parse(self, content, finish_reason="stop"):
        from evoagent.llm import parse_model_json

        return parse_model_json(content, finish_reason)

    def test_plain_json(self):
        self.assertEqual({"a": 1}, self._parse('{"a": 1}'))

    def test_markdown_fences_are_stripped(self):
        content = '```json\n{"findings": [{"path": "a.py"}]}\n```'
        self.assertEqual({"findings": [{"path": "a.py"}]}, self._parse(content))

    def test_trailing_prose_is_ignored(self):
        content = '{"ok": true}\n\nI have returned the JSON as requested.'
        self.assertEqual({"ok": True}, self._parse(content))

    def test_raw_control_characters_in_strings(self):
        content = '{"evidence": "line1\nline2\ttabbed"}'
        self.assertEqual({"evidence": "line1\nline2\ttabbed"}, self._parse(content))

    def test_failure_includes_content_preview(self):
        with self.assertRaises(ValueError) as ctx:
            self._parse("not json at all", finish_reason="length")
        message = str(ctx.exception)
        self.assertIn("finish_reason=length", message)
        self.assertIn("not json at all", message)


if __name__ == "__main__":
    unittest.main()


class TrailingCommaRepairTests(unittest.TestCase):
    def _parse(self, content):
        from evoagent.llm import parse_model_json

        return parse_model_json(content)

    def test_trailing_comma_before_closing_brace(self):
        self.assertEqual(
            {"findings": [{"a": 1}]},
            self._parse('{"findings": [{"a": 1,},]}'),
        )

    def test_trailing_comma_in_nested_string_content_survives(self):
        content = '{"evidence": "keep, this comma"}'
        self.assertEqual({"evidence": "keep, this comma"}, self._parse(content))


class WhitespaceResponseRetryTests(unittest.TestCase):
    def test_whitespace_only_response_is_retried(self):
        client = JsonChatClient(
            "https://llm.test/v1", "key", "test-model", "test",
            retries=2, backoff_seconds=0.0,
        )
        responses = [
            chat_response("   ", finish_reason="stop"),
            chat_response("   \n  ", finish_reason="stop"),
            chat_response('{"ok": 1}'),
        ]
        with patch("evoagent.llm.urllib.request.urlopen", side_effect=responses):
            self.assertEqual({"ok": 1}, client.complete_json("role", "system", "user"))

    def test_whitespace_response_exhausting_retries_raises(self):
        client = JsonChatClient(
            "https://llm.test/v1", "key", "test-model", "test",
            retries=1, backoff_seconds=0.0,
        )
        with patch(
            "evoagent.llm.urllib.request.urlopen",
            side_effect=[chat_response("   ", finish_reason="stop")] * 2,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                client.complete_json("role", "system", "user")
        self.assertIn("invalid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
