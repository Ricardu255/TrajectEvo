"""Durable Lead↔workers session state, checkpointed apart from the engine.

``LeadSession`` owns the review-session shape (delegations, worker results,
critic decisions, Lead finalisation) and the checkpoint contract used to
resume it, so ``AgenticReviewer`` orchestrates without knowing the persisted
format.  The serialized shape is unchanged from the original inline dict so
checkpoints written by older code still resume.
"""
from typing import Any, Dict, List

from .telemetry import ExecutionLedger


CHECKPOINT_NAME = "agentic-lead-session"
PROTOCOL = "lead-workers-v3"

# Fields restored from a checkpoint; anything else in the saved dict is
# ignored so unknown keys from future versions cannot corrupt a resume.
_STATE_FIELDS = frozenset({
    "phase", "scanner_complete", "scanner_findings", "scanner_components",
    "delegations", "lead_delegation", "risk_level", "worker_results",
    "lead_assessments", "revision_results", "revision_rounds",
    "critic_decisions", "critic_candidates", "critic_complete",
    "candidate_findings_before_critic", "lead_final", "accepted_findings",
    "stop_reason", "context_management",
})


class LeadSession:
    def __init__(self):
        self.phase = "created"
        self.scanner_complete = False
        self.scanner_findings: List[Dict[str, Any]] = []
        self.scanner_components: List[Dict[str, Any]] = []
        self.delegations: List[Dict[str, Any]] = []
        self.lead_delegation: Dict[str, Any] = {}
        self.risk_level = "normal"
        self.worker_results: Dict[str, Dict[str, Any]] = {}
        self.lead_assessments: List[Dict[str, Any]] = []
        self.revision_results: Dict[str, Dict[str, Any]] = {}
        self.revision_rounds = 0
        self.critic_decisions: List[Dict[str, Any]] = []
        self.critic_candidates: List[Dict[str, Any]] = []
        self.critic_complete = False
        self.candidate_findings_before_critic = 0
        self.lead_final: Dict[str, Any] = {}
        self.accepted_findings: List[Dict[str, Any]] = []
        self.stop_reason = ""
        self.context_management: Dict[str, Any] = {}

    @classmethod
    def load(cls, store, task_id: str, ledger: ExecutionLedger, context_manager) -> "LeadSession":
        """Restore a session (and its ledger/context history) from checkpoints."""
        session = cls()
        if not task_id:
            return session
        checkpoint = (store.load_checkpoints(task_id) or {}).get(CHECKPOINT_NAME) or {}
        state = checkpoint.get("state") or {}
        if state.get("protocol") != PROTOCOL:
            return session
        if state.get("execution"):
            ledger.restore(state["execution"])
        restored = state.get("session") or {}
        for key, value in restored.items():
            if key in _STATE_FIELDS:
                setattr(session, key, value)
        context_manager.restore(task_id, restored.get("context_management"))
        return session

    def save(
        self, store, task_id: str, ledger: ExecutionLedger,
        context_summary: Dict[str, Any], completed: bool = False,
    ) -> None:
        if not task_id:
            return
        self.context_management = context_summary
        store.save_checkpoint(
            task_id, CHECKPOINT_NAME,
            {
                "protocol": PROTOCOL, "session": self.to_state(),
                "execution": ledger.summary(),
            },
            "completed" if completed else "in_progress",
            max(1, len(ledger.model_calls)),
        )

    def to_state(self) -> Dict[str, Any]:
        return {
            key: getattr(self, key) for key in (
                "phase", "scanner_complete", "scanner_findings",
                "scanner_components", "delegations", "lead_delegation",
                "risk_level", "worker_results", "lead_assessments",
                "revision_results", "revision_rounds", "critic_decisions",
                "critic_candidates", "critic_complete",
                "candidate_findings_before_critic", "lead_final",
                "accepted_findings", "stop_reason", "context_management",
            )
        }

    @staticmethod
    def normalize_risk_level(value) -> str:
        risk = str(value or "normal").strip().lower()
        return risk if risk in {"low", "normal", "high"} else "normal"

    @staticmethod
    def normalize_delegations(
        raw, worker_roles: List[str], changed_files: List[str],
        available_skills=None, requested_skills=None,
    ) -> List[Dict[str, Any]]:
        available = set(available_skills or set())
        requested = [
            name for name in requested_skills or [] if name in available
        ]
        values, seen_ids, covered = [], set(), set()
        for index, item in enumerate(raw or []):
            if not isinstance(item, dict):
                continue
            worker = str(item.get("worker", ""))
            if worker not in worker_roles:
                continue
            assignment_id = str(
                item.get("assignment_id") or "%s-%d" % (worker, index + 1)
            )[:100]
            if not assignment_id or assignment_id in seen_ids:
                continue
            seen_ids.add(assignment_id)
            covered.add(worker)
            values.append({
                "assignment_id": assignment_id, "worker": worker,
                "objective": str(item.get("objective") or "Review the assigned risk domain.")[:2000],
                "files": [
                    str(value)[:500] for value in (
                        _string_list(item.get("files")) or changed_files
                    )
                ][:100],
                "risk_domains": [
                    str(value)[:100] for value in _string_list(item.get("risk_domains"))
                ][:20],
                "required_evidence": [
                    str(value)[:200] for value in _string_list(item.get("required_evidence"))
                ][:20],
                "skills": list(dict.fromkeys(requested + [
                    str(value) for value in _string_list(item.get("skills"))
                    if str(value) in available
                ])),
            })
            if len(values) >= 12:
                break
        defaults = {
            "security": "Review security, authorization, input and sensitive-data risks.",
            "correctness-reliability": (
                "Review correctness, failure handling, concurrency, resources and compatibility."
            ),
        }
        for worker in worker_roles:
            if worker in covered or len(values) >= 12:
                continue
            values.append({
                "assignment_id": "%s-default" % worker, "worker": worker,
                "objective": defaults[worker], "files": list(changed_files)[:100],
                "risk_domains": [], "required_evidence": ["changed-line evidence"],
                "skills": list(requested),
            })
        return values

    @staticmethod
    def normalize_revision_requests(raw, assignments) -> List[Dict[str, Any]]:
        by_id = {item["assignment_id"]: item for item in assignments}
        values, seen = [], set()
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            assignment_id = str(item.get("assignment_id", ""))
            original = by_id.get(assignment_id)
            if not original or assignment_id in seen:
                continue
            worker = str(item.get("worker") or original["worker"])
            if worker != original["worker"]:
                continue
            guidance = str(item.get("guidance", "")).strip()
            if not guidance:
                continue
            seen.add(assignment_id)
            values.append({
                "assignment_id": assignment_id, "worker": worker,
                "guidance": guidance[:2000],
                "required_evidence": [
                    str(value)[:200] for value in _string_list(item.get("required_evidence"))
                ][:20],
            })
        return values


def _string_list(value) -> List[str]:
    """Coerce a model-provided array field; a bare string is one element."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []
