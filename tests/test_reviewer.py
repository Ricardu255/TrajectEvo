import unittest

from evoagent.diff_parser import parse_unified_diff
from evoagent.reviewer import LocalRuleReviewer, OpenAICompatibleReviewer


DIFF = '--- a/app.py\n+++ b/app.py\n@@ -1 +1,2 @@\n+password = "secret"\n+eval(user_input)\n'


class LocalReviewerTests(unittest.TestCase):
    def test_detects_security_findings_only_on_added_lines(self):
        diff = """--- a/app.py
+++ b/app.py
@@ -1,2 +1,3 @@
-eval(old_input)
+password = "super-secret"
+eval(user_input)
 safe = True
"""
        findings = LocalRuleReviewer().review(diff, parse_unified_diff(diff))
        self.assertEqual({"SEC-EVAL", "SEC-HARDCODED-SECRET"}, {item.rule_id for item in findings})
        self.assertTrue(all(item.line in {1, 2} for item in findings))


class OpenAICompatibleParseFindingsTests(unittest.TestCase):
    def test_null_confidence_and_bad_line_do_not_crash(self):
        result = {"findings": [
            {"path": "app.py", "line": 1, "severity": "high", "confidence": None},
            {"path": "app.py", "line": "abc"},
            {"path": "app.py", "line": None},
            {"path": "app.py", "line": 2, "confidence": "0.9"},
        ]}
        findings = OpenAICompatibleReviewer._parse_findings(
            result, parse_unified_diff(DIFF)
        )
        self.assertEqual(2, len(findings))
        self.assertEqual(1, findings[0].line)
        self.assertEqual(0.7, findings[0].confidence)
        self.assertEqual(2, findings[1].line)
        self.assertEqual(0.9, findings[1].confidence)


if __name__ == "__main__":
    unittest.main()

