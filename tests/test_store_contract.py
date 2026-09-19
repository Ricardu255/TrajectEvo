"""One behavioural contract for every StoreProtocol backend.

The same ``StoreContract`` tests run against the SQLite store and — when
``EVOAGENT_TEST_DATABASE_URL`` points at a live PostgreSQL instance — against
``PostgresTaskStore``, so the two backends cannot drift apart silently.
Identifiers are randomised per run so the suite can be re-run against a
persistent PostgreSQL database.
"""
import os
import tempfile
import unittest
import uuid

from evoagent.models import ReviewReport, TaskState, TraceEvent
from evoagent.store import TaskStore, utc_now
from evoagent.store_contract import StoreProtocol

DATABASE_URL = os.environ.get("EVOAGENT_TEST_DATABASE_URL", "")
EXPIRED = "2020-01-01T00:00:00+00:00"


def _uid() -> str:
    return uuid.uuid4().hex[:10]


def _trace(step: int, state: TaskState = TaskState.PLANNING) -> TraceEvent:
    return TraceEvent(step, state, "step %d" % step, utc_now())


def _report(repository: str = "demo/api") -> ReviewReport:
    return ReviewReport(
        repository=repository, pull_request=1, summary="ok", risk="low",
    )


def _memory(memory_id: str, **overrides) -> dict:
    value = {
        "id": memory_id, "tenant_id": "default", "repository": "demo/api",
        "task_id": "t1", "agent": "lead", "scope": "working",
        "kind": "observation", "content": "content", "keywords": ["alpha"],
        "metadata": {"key": "value"}, "importance": 0.5,
        "created_at": utc_now(), "expires_at": None,
    }
    value.update(overrides)
    return value


class StoreContract:
    """Behavioural contract shared by every store backend test-case."""

    def make_store(self):
        raise NotImplementedError

    def setUp(self):
        self.store = self.make_store()
        self.assertIsInstance(self.store, StoreProtocol)

    def test_task_lifecycle_and_tenant_isolation(self):
        run = _uid()
        self.store.create(
            "t1-%s" % run, "demo/api", 1, {"mode": "agentic"}, tenant_id="tenant-a"
        )
        self.store.create(
            "t2-%s" % run, "demo/api", None, {"mode": "single"}, tenant_id="tenant-b"
        )
        task = self.store.get("t1-%s" % run, "tenant-a")
        self.assertEqual("PENDING", task["state"])
        self.assertEqual({"mode": "agentic"}, task["input"])
        self.assertIsNone(self.store.get("t1-%s" % run, "tenant-b"))
        self.assertIsNone(self.store.get("missing-%s" % run))

        self.store.transition(
            "t1-%s" % run, _trace(1, TaskState.EXECUTING)
        )
        self.store.fail("t1-%s" % run, "boom", _trace(2, TaskState.FAILED))
        failed = self.store.get("t1-%s" % run, "tenant-a")
        self.assertEqual("FAILED", failed["state"])
        self.assertEqual("boom", failed["error"])
        self.assertEqual(2, len(failed["trace"]))

        self.store.succeed(
            "t2-%s" % run, _report(), _trace(1, TaskState.SUCCESS)
        )
        ok = self.store.get("t2-%s" % run, "tenant-b")
        self.assertEqual("SUCCESS", ok["state"])
        self.assertEqual("demo/api", ok["report"]["repository"])

        self.assertEqual(
            {"t1-%s" % run, "t2-%s" % run},
            {item["id"] for item in self.store.list_tasks()},
        )
        self.assertEqual(
            {"t2-%s" % run},
            {item["id"] for item in self.store.list_tasks(tenant_id="tenant-b")},
        )

    def test_checkpoints_and_task_payload(self):
        run = _uid()
        self.store.create("t1-%s" % run, "demo/api", None, {})
        self.store.save_checkpoint("t1-%s" % run, "planning", {"plan": [1]})
        self.store.save_checkpoint(
            "t1-%s" % run, "planning", {"plan": [1, 2]}, attempt=2
        )
        self.store.save_checkpoint(
            "t1-%s" % run, "executing", {}, "failed", 1, "boom"
        )
        checkpoints = self.store.load_checkpoints("t1-%s" % run)
        self.assertEqual({"plan": [1, 2]}, checkpoints["planning"]["state"])
        self.assertEqual(2, checkpoints["planning"]["attempt"])
        self.assertEqual("failed", checkpoints["executing"]["status"])
        self.assertEqual("boom", checkpoints["executing"]["error"])

        self.store.save_task_payload("t1-%s" % run, "diff --git a/x b/x")
        self.assertEqual(
            "diff --git a/x b/x", self.store.get_task_payload("t1-%s" % run)
        )
        self.assertIsNone(self.store.get_task_payload("missing-%s" % run))

        self.store.update_task_input("t1-%s" % run, {"repository_root": "/tmp"})
        self.assertEqual(
            "/tmp", self.store.get("t1-%s" % run)["input"]["repository_root"]
        )
        with self.assertRaises(ValueError):
            self.store.update_task_input("missing-%s" % run, {})

    def test_cancellation(self):
        run = _uid()
        self.store.create("t1-%s" % run, "demo/api", None, {}, tenant_id="tenant-a")
        self.assertFalse(self.store.is_cancelled("t1-%s" % run))
        self.assertTrue(self.store.request_cancel("t1-%s" % run, "tenant-a"))
        self.assertFalse(self.store.request_cancel("t1-%s" % run, "other-tenant"))
        self.assertTrue(self.store.is_cancelled("t1-%s" % run))
        self.store.cancel(
            "t1-%s" % run, _trace(1, TaskState.CANCELLED)
        )
        self.assertEqual("CANCELLED", self.store.get("t1-%s" % run)["state"])

    def test_agent_messages_and_memory(self):
        run = _uid()
        self.store.create("t1-%s" % run, "demo/api", None, {})
        self.store.record_agent_message("t1-%s" % run, {
            "sender": "lead", "recipient": "security", "kind": "assignment",
            "correlation_id": "c1", "content": {"objective": "check"},
        })
        task = self.store.get("t1-%s" % run)
        self.assertEqual(1, len(task["collaboration"]))
        self.assertEqual(
            {"objective": "check"}, task["collaboration"][0]["content"]
        )

        self.store.save_agent_memory(_memory("m1-%s" % run, importance=0.9))
        self.store.save_agent_memory(_memory(
            "m2-%s" % run, scope="episode", expires_at=EXPIRED,
        ))
        rows = self.store.list_agent_memories(
            "default", "demo/api", ("working", "episode")
        )
        self.assertEqual(["m1-%s" % run], [row["id"] for row in rows])
        self.assertEqual(["alpha"], rows[0]["keywords"])
        self.assertEqual({"key": "value"}, rows[0]["metadata"])
        self.assertEqual(
            1, self.store.delete_agent_memories(task_id="t1", scope="episode")
        )
        self.assertEqual(0, self.store.purge_expired_agent_memories())
        with self.assertRaises(ValueError):
            self.store.delete_agent_memories()

    def test_failure_cases(self):
        run = _uid()
        self.store.create(
            "t1-%s" % run, "demo/api", None, {}, tenant_id="tenant-a"
        )
        self.store.record_failure_case(
            "t1-%s" % run, "execution_error", {"error": "boom"}
        )
        rows = self.store.list_task_failure_cases("t1-%s" % run, "tenant-a")
        self.assertEqual(1, len(rows))
        self.assertEqual({"error": "boom"}, rows[0]["payload"])

        unresolved = self.store.list_failure_cases(unresolved_only=True)
        self.assertEqual(1, len(unresolved))
        self.assertEqual(
            1, len(self.store.list_failure_cases(tenant_id="tenant-a"))
        )
        self.store.resolve_failure_cases([unresolved[0]["id"]])
        self.assertEqual(0, len(self.store.list_failure_cases(unresolved_only=True)))

    def test_skill_versions(self):
        run = _uid()
        name = "llm-review-%s" % run
        first = self.store.save_skill_version(name, "v1 prompt", 0.5, activate=True)
        self.assertEqual(1, first["version"])
        second = self.store.save_skill_version(name, "v2 prompt", 0.8)
        self.assertEqual(2, second["version"])
        self.assertEqual(1, self.store.get_active_skill_version(name)["version"])
        self.assertTrue(self.store.activate_skill_version(name, 2))
        self.assertFalse(self.store.activate_skill_version(name, 99))
        self.assertEqual(2, self.store.get_active_skill_version(name)["version"])
        self.assertEqual(2, len(self.store.list_skill_versions(name)))

    def test_skill_artifacts(self):
        run = _uid()
        name = "security-review-%s" % run
        saved = self.store.save_skill_artifact(
            name, {"rules": ["a"]}, 0.7, activate=True
        )
        self.assertEqual(1, saved["version"])
        self.assertEqual(64, len(saved["artifact_sha256"]))
        active = self.store.get_active_skill_artifact(name)
        self.assertEqual({"rules": ["a"]}, active["artifact"])
        self.assertTrue(active["active"])
        second = self.store.save_skill_artifact(name, {"rules": ["b"]}, 0.9)
        self.assertEqual(2, second["version"])
        self.assertIsNone(
            self.store.get_active_skill_artifact(name, tenant_id="other")
        )
        self.assertEqual(1, len(self.store.list_active_skill_artifacts("default")))

        self.assertFalse(self.store.activate_skill_artifact(name, 2))
        self.store.save_skill_evolution_run({
            "id": "run-%s" % run, "skill_name": name, "candidate_version": 2,
            "baseline_version": 1, "decision": "activated",
            "candidate_score": 0.9, "baseline_score": 0.7,
            "metrics": {"f1": 0.9}, "created_at": utc_now(),
        })
        self.assertTrue(self.store.activate_skill_artifact(name, 2))
        self.assertFalse(self.store.activate_skill_artifact(name, 99))
        self.assertEqual(
            2, self.store.get_active_skill_artifact(name)["version"]
        )
        self.assertEqual(
            2, len(self.store.list_skill_artifact_versions(name))
        )
        self.assertEqual(
            1, len(self.store.list_skill_evolution_runs(tenant_id="default"))
        )

    def test_evolution_runs_and_evaluation_cases(self):
        run = _uid()
        self.store.save_evolution_run({
            "id": "e1-%s" % run, "skill_name": "llm-review",
            "candidate_version": 2, "baseline_version": 1,
            "decision": "pending", "candidate_score": 0.8,
            "baseline_score": 0.75, "metrics": {"f1": 0.8},
            "created_at": utc_now(),
        })
        self.assertTrue(
            self.store.update_evolution_run("e1-%s" % run, "activated", {"f1": 0.86})
        )
        self.assertFalse(self.store.update_evolution_run("missing", "x", {}))
        runs = self.store.list_evolution_runs()
        self.assertEqual("activated", runs[0]["decision"])
        self.assertEqual({"f1": 0.86}, runs[0]["metrics"])

        name = "case-%s" % run
        case = self.store.save_evaluation_case(
            name, "validation", "diff", [{"path": "a.py"}]
        )
        self.assertEqual([{"path": "a.py"}], case["expected"])
        self.assertTrue(case["active"])
        again = self.store.save_evaluation_case(
            name, "validation", "diff", [{"path": "a.py"}]
        )
        self.assertEqual(name, again["name"])
        with self.assertRaises(ValueError):
            self.store.save_evaluation_case(
                name, "holdout", "diff", [{"path": "a.py"}]
            )
        self.assertEqual(1, len(self.store.list_evaluation_cases()))
        self.assertEqual(0, len(self.store.list_evaluation_cases(split="holdout")))

    def test_webhook_claim_idempotency(self):
        run = _uid()
        self.assertTrue(
            self.store.claim_webhook("d-%s" % run, "default", "opened", "sha-a")
        )
        self.assertFalse(
            self.store.claim_webhook("d-%s" % run, "default", "opened", "sha-a")
        )
        with self.assertRaises(ValueError):
            self.store.claim_webhook(
                "d-%s" % run, "default", "opened", "sha-different"
            )
        with self.assertRaises(ValueError):
            self.store.claim_webhook("", "default", "opened", "sha")
        self.store.complete_webhook("d-%s" % run, "task-1")
        delivery = self.store.get_webhook("d-%s" % run)
        self.assertEqual("task-1", delivery["task_id"])
        self.assertIsNone(self.store.get_webhook("missing"))

    def test_users_repositories_and_audit(self):
        run = _uid()
        self.store.create_user(
            "u1-%s" % run, "alice-%s" % run, "hash", "tenant-a", "admin"
        )
        self.store.create_user(
            "u1-%s" % run, "alice-%s" % run, "hash", "tenant-b", "viewer"
        )
        user = self.store.get_user("alice-%s" % run)
        self.assertEqual("hash", user["password_hash"])
        self.assertEqual(
            {"tenant-a": "admin", "tenant-b": "viewer"},
            {item["tenant_id"]: item["role"] for item in user["memberships"]},
        )
        self.assertIsNone(self.store.get_user("bob-%s" % run))

        self.assertTrue(self.store.repository_allowed("tenant-a", "demo/api"))
        self.store.grant_repository("tenant-a", "demo/api")
        self.assertTrue(self.store.repository_allowed("tenant-a", "demo/api"))
        self.assertFalse(self.store.repository_allowed("tenant-a", "other/api"))
        self.assertFalse(
            self.store.repository_allowed(
                "tenant-a", "demo/api", require_auto_fix=True
            )
        )
        self.store.grant_repository("tenant-a", "demo/api", auto_fix=True)
        self.assertTrue(
            self.store.repository_allowed(
                "tenant-a", "demo/api", require_auto_fix=True
            )
        )

        self.store.audit("tenant-a", "actor", "review", "demo/api")
        self.store.audit(
            "tenant-a", "actor", "login", "session", {"ip": "127.0.0.1"}
        )
        entries = self.store.list_audit("tenant-a")
        self.assertEqual(2, len(entries))
        self.assertEqual({"ip": "127.0.0.1"}, entries[0]["detail"])

    def test_deployments_shadow_observations_and_rollback(self):
        run = _uid()
        name = "security-review-%s" % run
        self.store.save_deployment("default", name, {
            "stable_version": 1, "candidate_version": 2, "status": "running",
            "min_samples": 2, "max_error_rate": 0.5,
            "max_disagreement_rate": 0.5, "auto_promote": True,
            "shadow_percent": 100,
        })
        result = self.store.record_shadow_observation(
            "default", name, "task-1", "shadow",
            {"finding": "a"}, {"finding": "a"}, 0.0,
        )
        self.assertEqual("running", result["status"])
        self.assertEqual(1, result["shadow_samples"])
        result = self.store.record_shadow_observation(
            "default", name, "task-2", "shadow",
            {"finding": "b"}, None, 1.0,
        )
        self.assertEqual("promoted", result["status"])
        deployment = self.store.get_deployment("default", name)
        self.assertEqual("promoted", deployment["status"])
        self.assertEqual(2, deployment["stable_version"])

        observations = self.store.list_release_observations("default", name)
        self.assertEqual(2, len(observations))
        self.assertEqual({"finding": "b"}, observations[0]["primary"])
        self.assertIsNone(observations[0]["candidate"])
        self.assertEqual({"finding": "a"}, observations[1]["primary"])
        self.assertEqual([], self.store.list_release_observations("default", name + "-x"))

        other = "llm-review-%s" % run
        self.store.save_deployment("default", other, {
            "candidate_version": 3, "status": "running",
            "min_samples": 1, "max_error_rate": 0.0,
        })
        result = self.store.record_deployment_result("default", other, True)
        self.assertEqual("rolled_back", result["status"])
        self.assertIsNone(
            self.store.record_deployment_result("default", "missing", False)
        )

    def test_alerts_installations_and_dashboard(self):
        run = _uid()
        self.store.create(
            "t1-%s" % run, "demo/api", None, {}, tenant_id="tenant-a"
        )
        self.store.succeed(
            "t1-%s" % run, _report(), _trace(1, TaskState.SUCCESS)
        )
        self.store.create_alert("tenant-a", "error-rate", "high", "too many failures")
        self.store.create_alert("tenant-a", "error-rate", "high", "too many failures")
        self.assertEqual(1, len(self.store.list_alerts("tenant-a")))
        # One open alert per key: severity alone does not duplicate it.
        self.store.create_alert("tenant-a", "error-rate", "critical", "worse")
        self.assertEqual(1, len(self.store.list_alerts("tenant-a")))

        self.store.save_installation(42, "demo-org", "tenant-a")
        self.assertEqual("tenant-a", self.store.installation_tenant(42))
        self.assertIsNone(self.store.installation_tenant(99))

        stats = self.store.dashboard_stats("tenant-a")
        self.assertEqual(1, stats["tasks_total"])
        self.assertEqual(1, stats["tasks_success"])
        self.assertEqual(1.0, stats["success_rate"])
        self.assertIn("active_skill_versions", stats)


class SqliteStoreContractTests(StoreContract, unittest.TestCase):
    def make_store(self):
        handle, path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.addCleanup(os.unlink, path)
        return TaskStore(path)


@unittest.skipUnless(
    DATABASE_URL,
    "set EVOAGENT_TEST_DATABASE_URL (e.g. postgresql://user:pass@host/db) "
    "to run the PostgreSQL contract",
)
class PostgresStoreContractTests(StoreContract, unittest.TestCase):
    def make_store(self):
        from evoagent.postgres_store import PostgresTaskStore

        return PostgresTaskStore(DATABASE_URL)


if __name__ == "__main__":
    unittest.main()
