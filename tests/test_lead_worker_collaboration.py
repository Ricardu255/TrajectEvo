import json
import os
import tempfile
import unittest

from evoagent.agentic_core import AgenticReviewer
from evoagent.diff_parser import parse_unified_diff
from evoagent.memory import MemoryManager
from evoagent.models import Finding, Severity
from evoagent.store import TaskStore


DIFF = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+eval(user_input)\n"


class HierarchicalClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self):
        self.calls = []
        self.security_calls = 0

    def complete_json(self, role, _system, user, ledger=None, max_tokens=None):
        managed = json.loads(user)
        task = json.loads(managed["task"])
        self.calls.append((role, task.get("phase", "worker")))
        if ledger:
            ledger.record_model(
                role, self.provider, self.model,
                {"prompt_tokens": 10, "completion_tokens": 5}, 1,
            )
        if role == "lead":
            if task["phase"] == "delegate":
                return {
                    "action": "final", "delegations": [
                        {
                            "assignment_id": "security-1", "worker": "security",
                            "objective": "Trace the changed input into dangerous calls.",
                        },
                        {
                            "assignment_id": "reliability-1",
                            "worker": "correctness-reliability",
                            "objective": "Review failure and resource behavior.",
                        },
                    ], "risk_level": "high",
                }
            if task["phase"] == "assess-workers" and task["revision_round"] == 0:
                return {
                    "action": "final", "revision_requests": [{
                        "assignment_id": "security-1", "worker": "security",
                        "guidance": "Add an exact changed-line finding for dynamic execution.",
                        "required_evidence": ["changed-line evidence"],
                    }], "critic_objective": "Challenge every proposed finding.",
                }
            if task["phase"] == "assess-workers":
                return {
                    "action": "final", "revision_requests": [],
                    "critic_objective": "Challenge every proposed finding.",
                }
            if task["phase"] == "finalize":
                return {
                    "action": "final",
                    "accepted_finding_indices": list(
                        range(len(task["candidate_findings"]))
                    ),
                    "confidence_adjustments": [],
                    "resolution_summary": "Workers supplied evidence and Critic approved.",
                }
        if role == "security":
            self.security_calls += 1
            if self.security_calls == 1:
                return {"action": "final", "findings": []}
            return {"action": "final", "findings": [{
                "rule_id": "SEC-LEAD-REVISION", "severity": "medium",
                "title": "Dynamic execution", "explanation": "Input is executed as code.",
                "path": "app.py", "line": 1, "evidence": "eval(user_input)",
                "fix": "Use a constrained parser.",
                "test": "Prove expressions are treated as data.", "confidence": 0.9,
            }]}
        if role == "correctness-reliability":
            return {"action": "final", "findings": []}
        if role == "critic":
            return {
                "action": "final", "decisions": [
                    {
                        "finding_index": index, "accepted": True,
                        "objections": [], "confidence_adjustment": 0.0,
                    }
                    for index, _item in enumerate(task["candidates"])
                ],
            }
        raise AssertionError((role, task))


class LeadWorkerCollaborationTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.store.create("task", "org/repo", 1, {
            "mode": "agentic",
            "enabled_agents": [
                "lead", "security", "correctness-reliability", "critic",
            ],
        })

    def tearDown(self):
        os.unlink(self.path)

    def test_lead_delegates_requests_revision_and_synthesizes(self):
        client = HierarchicalClient()
        reviewer = AgenticReviewer(self.store, client)

        findings = reviewer.review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )
        summary = reviewer.collaboration_summary("task")

        self.assertIn("SEC-LEAD-REVISION", {item.rule_id for item in findings})
        self.assertEqual("lead-workers", summary["collaboration"]["protocol"])
        self.assertEqual(2, client.security_calls)
        self.assertEqual(1, len(summary["collaboration"]["revision_results"]))
        self.assertEqual(
            "high-risk-one-revision-round",
            summary["collaboration"]["stop_reason"],
        )
        self.assertEqual(1, summary["collaboration"]["revision_rounds"])
        self.assertEqual(
            1, client.calls.count(("lead", "assess-workers"))
        )
        self.assertEqual(7, summary["execution"]["llm_calls"])
        session_events = {
            item["event"]
            for item in summary["execution"]["agent_traces"]["lead-session"]
        }
        self.assertTrue({
            "assignment_created", "worker_reported", "revision_completed",
            "lead_activated", "lead_completed",
        }.issubset(session_events))
        checkpoint = self.store.load_checkpoints("task")["agentic-lead-session"]
        self.assertEqual("completed", checkpoint["status"])
        self.assertEqual("completed", checkpoint["state"]["session"]["phase"])

    def test_completed_session_resumes_without_repeating_agent_calls(self):
        first = HierarchicalClient()
        AgenticReviewer(self.store, first).review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )
        resumed_client = HierarchicalClient()
        resumed = AgenticReviewer(self.store, resumed_client)

        findings = resumed.review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )

        self.assertIn("SEC-LEAD-REVISION", {item.rule_id for item in findings})
        self.assertEqual([], resumed_client.calls)
        self.assertGreater(
            resumed.collaboration_summary("task")["execution"]["llm_calls"], 0
        )

    def test_gate_decisions_are_archived_for_future_agent_recall(self):
        memory = MemoryManager(self.store)
        reviewer = AgenticReviewer(self.store, HierarchicalClient(), memory_manager=memory)

        reviewer.review_with_context("task", DIFF, parse_unified_diff(DIFF), "org/repo")

        episodes = memory.recall("default", "org/repo", "SEC-LEAD-REVISION")
        self.assertTrue(any(item["kind"] == "finding_approved" for item in episodes))
        self.assertTrue(any(item["kind"] == "task_summary" for item in episodes))


class ModelOutputParsingTests(unittest.TestCase):
    @staticmethod
    def _finding(index):
        return Finding(
            rule_id="LLM-REVIEW", severity=Severity.MEDIUM, title="title",
            explanation="explanation", path="app.py", line=index + 1,
            evidence="evidence", fix="fix", test="test", confidence=0.8,
        )

    def test_critic_accepts_only_real_boolean_true(self):
        result = {"_observations": [], "decisions": [
            {"finding_index": 0, "accepted": "false", "confidence_adjustment": 0},
            {"finding_index": 1, "accepted": "true", "confidence_adjustment": 0},
            {"finding_index": 2, "accepted": True, "confidence_adjustment": 0},
            {"finding_index": 3, "accepted": False, "confidence_adjustment": 0},
            {"finding_index": 4, "accepted": None, "confidence_adjustment": 0},
        ]}
        candidates = [self._finding(index) for index in range(5)]
        candidates, decisions = AgenticReviewer._apply_critic(result, candidates)
        self.assertEqual(
            [False, True, True, False, False],
            [item["accepted"] for item in decisions],
        )

    def test_critic_missing_decision_defaults_to_rejected(self):
        result = {"_observations": [], "decisions": []}
        candidates = [self._finding(0)]
        candidates, decisions = AgenticReviewer._apply_critic(result, candidates)
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(
            ["critic returned no explicit decision"], decisions[0]["objections"],
        )

    def test_delegation_string_fields_stay_whole(self):
        raw = [{
            "assignment_id": "sec-1", "worker": "security",
            "objective": "Check secrets.",
            "files": "src/app.py",
            "risk_domains": "injection",
            "required_evidence": "changed-line evidence",
            "skills": "security-review",
        }]
        values = AgenticReviewer._normalize_delegations(
            raw, ["security"], ["app.py"], {"security-review"}, ["security-review"],
        )
        self.assertEqual(["src/app.py"], values[0]["files"])
        self.assertEqual(["injection"], values[0]["risk_domains"])
        self.assertEqual(["changed-line evidence"], values[0]["required_evidence"])
        self.assertEqual(["security-review"], values[0]["skills"])

    def test_delegation_unusable_files_fall_back_to_changed_files(self):
        raw = [
            {"assignment_id": "sec-1", "worker": "security", "files": {"a": 1}},
            {"assignment_id": "sec-2", "worker": "security", "files": None},
        ]
        values = AgenticReviewer._normalize_delegations(raw, ["security"], ["app.py"])
        self.assertEqual(["app.py"], values[0]["files"])
        self.assertEqual(["app.py"], values[1]["files"])

    def test_delegation_accepts_proper_lists_unchanged(self):
        raw = [{
            "assignment_id": "sec-1", "worker": "security",
            "files": ["src/app.py", "src/util.py"],
            "risk_domains": ["injection", "authorization"],
            "required_evidence": ["changed-line evidence", "scanner output"],
            "skills": ["security-review"],
        }]
        values = AgenticReviewer._normalize_delegations(
            raw, ["security"], ["app.py"], {"security-review"}, [],
        )
        self.assertEqual(["src/app.py", "src/util.py"], values[0]["files"])
        self.assertEqual(
            ["injection", "authorization"], values[0]["risk_domains"],
        )
        self.assertEqual(
            ["changed-line evidence", "scanner output"],
            values[0]["required_evidence"],
        )
        self.assertEqual(["security-review"], values[0]["skills"])

    def test_revision_request_string_evidence_stays_whole(self):
        raw = [{
            "assignment_id": "sec-1", "worker": "security",
            "guidance": "Cite concrete evidence.",
            "required_evidence": "scanner output",
        }]
        assignments = [{"assignment_id": "sec-1", "worker": "security"}]
        values = AgenticReviewer._normalize_revision_requests(raw, assignments)
        self.assertEqual(["scanner output"], values[0]["required_evidence"])


if __name__ == "__main__":
    unittest.main()
