import os
import tempfile
import unittest

from evoagent.context_manager import ContextManager
from evoagent.lead_session import CHECKPOINT_NAME, PROTOCOL, LeadSession
from evoagent.store import TaskStore
from evoagent.telemetry import ExecutionLedger


class LeadSessionCheckpointTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.addCleanup(os.unlink, self.path)

    def test_fresh_session_when_no_checkpoint(self):
        ledger = ExecutionLedger("agentic")
        session = LeadSession.load(self.store, "", ledger, ContextManager())
        self.assertEqual("created", session.phase)
        self.assertFalse(session.scanner_complete)
        self.assertEqual("normal", session.risk_level)

    def test_save_and_restore_round_trip(self):
        ledger = ExecutionLedger("agentic")
        ledger.record_model(
            "lead", "fake", "fake-model",
            {"prompt_tokens": 10, "completion_tokens": 5}, 1,
        )
        session = LeadSession()
        session.phase = "critic-completed"
        session.delegations = [{
            "assignment_id": "security-1", "worker": "security",
            "objective": "Review", "files": ["app.py"],
            "risk_domains": [], "required_evidence": [], "skills": [],
        }]
        session.worker_results = {"security-1": {"status": "completed", "findings": []}}
        session.critic_complete = True
        session.risk_level = "high"
        session.revision_rounds = 1
        context = ContextManager()
        context.begin("task-1")
        context.compress_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n+eval(x)\n", "task-1", "review",
        )
        session.save(self.store, "task-1", ledger, context.summary("task-1"))

        restored_ledger = ExecutionLedger("agentic")
        restored = LeadSession.load(
            self.store, "task-1", restored_ledger, ContextManager(),
        )
        self.assertEqual("critic-completed", restored.phase)
        self.assertEqual("high", restored.risk_level)
        self.assertEqual(session.delegations, restored.delegations)
        self.assertTrue(restored.critic_complete)
        self.assertEqual(1, restored.revision_rounds)
        self.assertEqual("lead", restored_ledger.model_calls[0].role)
        self.assertEqual(
            session.context_management["compression_calls"],
            restored.context_management["compression_calls"],
        )

    def test_checkpoint_shape_is_stable(self):
        session = LeadSession()
        session.save(self.store, "task-1", ExecutionLedger("agentic"), {})
        checkpoint = self.store.load_checkpoints("task-1")[CHECKPOINT_NAME]
        self.assertEqual("in_progress", checkpoint["status"])
        self.assertEqual(PROTOCOL, checkpoint["state"]["protocol"])
        self.assertIn("session", checkpoint["state"])
        self.assertIn("execution", checkpoint["state"])

    def test_unknown_protocol_starts_a_fresh_session(self):
        self.store.save_checkpoint("task-1", CHECKPOINT_NAME, {
            "protocol": "lead-workers-v9", "session": {"phase": "completed"},
            "execution": {},
        }, "completed", 1)
        session = LeadSession.load(
            self.store, "task-1", ExecutionLedger("agentic"), ContextManager(),
        )
        self.assertEqual("created", session.phase)
        self.assertFalse(session.critic_complete)


if __name__ == "__main__":
    unittest.main()
