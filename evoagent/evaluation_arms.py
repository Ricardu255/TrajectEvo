"""The reviewer arms under evaluation: topology definitions and factories.

One arm is a configured reviewer (a cheap single-model baseline or a product
``AgenticReviewer`` topology) that evaluation harnesses replay labelled cases
against.  The harnesses and suites that score these arms live in
``evaluation_v2``.
"""
from collections import Counter
import json
import time
from typing import Any, Callable, Dict, List, Optional

from .agentic_core import AgenticReviewer
from .finding_identity import canonical_identity
from .llm import JsonChatClient
from .models import Finding, Severity
from .reviewer import LocalRuleReviewer, Reviewer
from .review_rules import ContextRuleReviewer
from .skills import AgentSkill
from .telemetry import ExecutionLedger


REQUIRED_ARMS = (
    "multi-llm-no-critic", "full-agentic",
)

EXPERIMENT_ARMS = (
    "single-llm",
    "single-llm-scanner",
    "multi-llm-no-critic",
    "full-agentic",
    "full-agentic-evolved-skill",
)

ARM_TOPOLOGY = {
    "single-llm": {
        "mode": "single",
        "roles": ("single-reviewer",),
        "deterministic_scanners": False,
    },
    "single-llm-scanner": {
        "mode": "single",
        "roles": ("single-reviewer",),
        "deterministic_scanners": True,
    },
    "multi-llm-no-critic": {
        "mode": "agentic",
        "roles": ("lead", "security", "correctness-reliability"),
        "deterministic_scanners": True,
    },
    "full-agentic": {
        "mode": "agentic",
        "roles": ("lead", "security", "correctness-reliability", "critic"),
        "deterministic_scanners": True,
    },
    "full-agentic-evolved-skill": {
        "mode": "agentic",
        "roles": ("lead", "security", "correctness-reliability", "critic"),
        "deterministic_scanners": True,
    },
}


SINGLE_REVIEW_PROMPT = """You are a single-model code reviewer. Review only defects introduced
by added lines in the supplied unified diff. Return actionable findings, not style comments. Use
the exact changed path and line. Return JSON only:
{"findings":[{"cwe":"CWE-...","rule_id":"...","severity":"critical|high|medium|low",
"title":"...","explanation":"...","path":"...","line":1,"evidence":"exact code",
"fix":"...","test":"...","confidence":0.0}]}"""


class _EvaluationTaskStore:
    """Minimal store used by AgenticReviewer during replay.

    Implements the ``ReviewStore`` surface; checkpoints are intentionally
    discarded because every replay case starts from a fresh session.
    """

    def __init__(self, task_input: dict):
        self.task_input = dict(task_input)

    def get(self, _task_id: str, _tenant_id: Optional[str] = None) -> dict:
        return {"input": dict(self.task_input)}

    def save_checkpoint(
        self, _task_id: str, _node: str, _state: dict,
        _status: str = "completed", _attempt: int = 1, _error: str = "",
    ) -> None:
        return None

    def load_checkpoints(self, _task_id: str) -> dict:
        return {}


class SingleModelReviewer(Reviewer):
    """One-call model baseline, optionally merged with the shared 14 rules."""

    def __init__(
        self, arm: str, client: JsonChatClient, total_token_budget: int,
        total_time_budget_seconds: int = 120,
    ):
        if arm not in {"single-llm", "single-llm-scanner"}:
            raise ValueError("single-model reviewer received invalid arm: %s" % arm)
        if total_token_budget < 256:
            raise ValueError("single-model review requires at least 256 tokens")
        if total_time_budget_seconds < 1:
            raise ValueError("total_time_budget_seconds must be positive")
        self.arm = arm
        self.name = arm
        self.client = client
        self.total_token_budget = int(total_token_budget)
        self.total_time_budget_seconds = int(total_time_budget_seconds)
        self.use_scanners = bool(ARM_TOPOLOGY[arm]["deterministic_scanners"])
        self.local = LocalRuleReviewer()
        self.context = ContextRuleReviewer()
        self._last_execution: Dict[str, Any] = {}

    @staticmethod
    def _parse_findings(raw_items, parsed) -> List[Finding]:
        valid = {(item.path, int(item.line)) for item in parsed.added_lines}
        findings = []
        for raw in raw_items or []:
            if not isinstance(raw, dict):
                continue
            try:
                path, line = str(raw.get("path", "")), int(raw.get("line", 0))
            except (TypeError, ValueError):
                continue
            if (path, line) not in valid:
                continue
            try:
                severity = Severity(str(raw.get("severity", "medium")).lower())
            except ValueError:
                severity = Severity.MEDIUM
            try:
                confidence = float(raw.get("confidence", 0.7))
            except (TypeError, ValueError):
                confidence = 0.7
            findings.append(Finding(
                rule_id=str(raw.get("rule_id", "LLM-REVIEW"))[:80],
                cwe=str(raw.get("cwe", "")).strip().upper() or None,
                severity=severity,
                title=str(raw.get("title", "Review finding"))[:200],
                explanation=str(raw.get("explanation", ""))[:4000],
                path=path, line=line,
                evidence=str(raw.get("evidence", ""))[:500],
                fix=str(raw.get("fix", ""))[:4000],
                test=str(raw.get("test", ""))[:4000],
                confidence=max(0.0, min(1.0, confidence)),
                source="single-reviewer",
            ))
        return findings

    @staticmethod
    def _merge(findings: List[Finding]) -> List[Finding]:
        merged = {}
        for finding in findings:
            key = (
                finding.path, int(finding.line),
                canonical_identity(finding.rule_id, finding.cwe),
            )
            current = merged.get(key)
            if current is None or finding.confidence > current.confidence:
                merged[key] = finding
        severity_order = {
            Severity.CRITICAL: 0, Severity.HIGH: 1,
            Severity.MEDIUM: 2, Severity.LOW: 3,
        }
        return sorted(
            merged.values(),
            key=lambda item: (severity_order[item.severity], item.path, item.line),
        )

    def review(self, diff: str, parsed) -> List[Finding]:
        return self.review_case({"diff": diff, "repository": ""}, parsed)

    def review_case(self, case: dict, parsed) -> List[Finding]:
        ledger = ExecutionLedger(self.arm)
        started = time.monotonic()
        payload = {
            "repository": str(case.get("repository", "")),
            "pull_request": case.get("pull_request"),
            "unified_diff": case["diff"],
            "changed_files": list(parsed.files),
        }
        result = self.client.complete_json(
            "single-reviewer", SINGLE_REVIEW_PROMPT,
            json.dumps(payload, ensure_ascii=False), ledger,
            max_tokens=self.total_token_budget,
        )
        if time.monotonic() - started > self.total_time_budget_seconds:
            raise RuntimeError("single-model review exceeded its total time budget")
        findings = self._parse_findings(result.get("findings"), parsed)
        if self.use_scanners:
            findings.extend(self.local.review(case["diff"], parsed))
            findings.extend(self.context.review(case["diff"], parsed))
        self._last_execution = ledger.summary()
        return self._merge(findings)

    def evaluation_execution(self) -> dict:
        return dict(self._last_execution)

    def evaluation_collaboration(self) -> dict:
        return {}

    def evaluation_config(self) -> dict:
        return {
            "arm": self.arm,
            "mode": "single",
            "roles": ["single-reviewer"],
            "deterministic_rules": 14 if self.use_scanners else 0,
            "total_token_budget_per_pr": self.total_token_budget,
            "total_time_budget_seconds_per_pr": self.total_time_budget_seconds,
        }


class ProductArmReviewer:
    """Run one ablation arm through the product AgenticReviewer."""

    def __init__(
        self, arm: str, client: JsonChatClient, total_token_budget: int,
        total_time_budget_seconds: int = 120, evolved_skill_artifact: Optional[dict] = None,
    ):
        if arm not in ARM_TOPOLOGY or ARM_TOPOLOGY[arm]["mode"] != "agentic":
            raise ValueError("unknown evaluation arm: %s" % arm)
        if arm == "full-agentic-evolved-skill" and not evolved_skill_artifact:
            raise ValueError("full-agentic-evolved-skill requires an evolved Skill artifact")
        topology = ARM_TOPOLOGY[arm]
        roles = tuple(topology["roles"])
        llm_role_count = max(1, len(roles))
        if total_token_budget < 256 * llm_role_count:
            raise ValueError(
                "%s requires at least %d total tokens" % (arm, 256 * llm_role_count)
            )
        if total_time_budget_seconds < llm_role_count:
            raise ValueError("total_time_budget_seconds is too small for %s" % arm)
        per_role_tokens = max(256, total_token_budget // llm_role_count)
        per_role_seconds = max(1, total_time_budget_seconds // llm_role_count)
        enabled = set(roles)
        task_input = {
            "mode": topology["mode"],
            "enabled_agents": sorted(enabled),
        }
        self.evolved_skill = (
            AgentSkill.from_artifact(evolved_skill_artifact)
            if evolved_skill_artifact else None
        )
        if self.evolved_skill is not None:
            task_input["enabled_skills"] = [self.evolved_skill.name]
        self.arm = arm
        self.name = arm
        self.client = client
        self.total_token_budget = int(total_token_budget)
        self.total_time_budget_seconds = int(total_time_budget_seconds)
        self.per_role_token_budget = per_role_tokens
        self.per_role_time_budget_seconds = per_role_seconds
        self.expected_roles = roles
        self.store = _EvaluationTaskStore(task_input)
        # LocalRuleReviewer contributes six rules. ContextRuleReviewer contributes
        # the same eight supplemental rules to every arm, for exactly 14 total.
        self.agentic = AgenticReviewer(
            self.store, client,
            default_token_budget=per_role_tokens,
            default_time_budget=per_role_seconds,
            enabled_roles=enabled,
            scanners=[ContextRuleReviewer()],
            skill_provider=(
                (lambda _tenant: [self.evolved_skill])
                if self.evolved_skill is not None else None
            ),
        )
        self._sequence = 0
        self._last_summary: Dict[str, Any] = {}

    def review(self, diff: str, parsed) -> list:
        return self.review_case({"diff": diff, "repository": ""}, parsed)

    def review_case(self, case: dict, parsed) -> list:
        self._sequence += 1
        task_id = "evaluation:%s:%d" % (self.arm, self._sequence)
        repository_root = str(case.get("repository_root") or "")
        findings = self.agentic.review_with_context(
            task_id, case["diff"], parsed,
            repository=repository_root or str(case.get("repository") or ""),
        )
        self._last_summary = self.agentic.collaboration_summary(task_id)
        self._validate_execution()
        return findings

    def _validate_execution(self) -> None:
        execution = self._last_summary.get("execution") or {}
        calls = execution.get("model_call_log") or []
        actual = Counter(
            str(item.get("role")) for item in calls if bool(item.get("ok", True))
        )
        required = set(self.expected_roles)
        if self.arm in {"full-agentic", "full-agentic-evolved-skill"}:
            collaboration = self._last_summary.get("collaboration") or {}
            proposed = int(
                collaboration.get("candidate_findings_before_critic", 0) or 0
            )
            if proposed == 0:
                required.discard("critic")
        missing = sorted(role for role in required if actual[role] < 1)
        if missing:
            raise RuntimeError(
                "%s completed without successful LLM role(s): %s"
                % (self.arm, ", ".join(missing))
            )

    def evaluation_execution(self) -> dict:
        return dict(self._last_summary.get("execution") or {})

    def evaluation_collaboration(self) -> dict:
        return dict(self._last_summary.get("collaboration") or {})

    def evaluation_config(self) -> dict:
        return {
            "arm": self.arm,
            "mode": ARM_TOPOLOGY[self.arm]["mode"],
            "roles": list(self.expected_roles),
            "deterministic_rules": 14,
            "total_token_budget_per_pr": self.total_token_budget,
            "per_role_token_budget": self.per_role_token_budget,
            "total_time_budget_seconds_per_pr": self.total_time_budget_seconds,
            "per_role_time_budget_seconds": self.per_role_time_budget_seconds,
            "skill": self.evolved_skill.name if self.evolved_skill is not None else None,
        }


def product_reviewer_factories(
    client: JsonChatClient, total_time_budget_seconds: int = 120,
) -> Dict[str, Callable[[str, int], ProductArmReviewer]]:
    """Create the two agentic topology arms with one shared model client."""

    def build(arm: str, model: str, token_budget: int) -> ProductArmReviewer:
        if str(client.model) != str(model):
            raise ValueError(
                "evaluation model %s does not match client model %s"
                % (model, client.model)
            )
        return ProductArmReviewer(
            arm, client, token_budget, total_time_budget_seconds,
        )

    return {
        arm: (
            lambda model, budget, selected=arm: build(selected, model, budget)
        )
        for arm in REQUIRED_ARMS
    }


def experiment_reviewer_factories(
    client: JsonChatClient, total_time_budget_seconds: int = 120,
    evolved_skill_artifact: Optional[dict] = None,
) -> Dict[str, Callable[[str, int], Reviewer]]:
    """Build the complete collaboration ablation matrix.

    The evolved-Skill arm is included only when an artifact is supplied, so a
    missing experimental input is visible instead of silently substituting an
    empty or static Skill.
    """

    def build(arm: str, model: str, token_budget: int) -> Reviewer:
        if str(client.model) != str(model):
            raise ValueError(
                "evaluation model %s does not match client model %s"
                % (model, client.model)
            )
        if ARM_TOPOLOGY[arm]["mode"] == "single":
            return SingleModelReviewer(
                arm, client, token_budget, total_time_budget_seconds,
            )
        return ProductArmReviewer(
            arm, client, token_budget, total_time_budget_seconds,
            evolved_skill_artifact=(
                evolved_skill_artifact
                if arm == "full-agentic-evolved-skill" else None
            ),
        )

    selected = [
        arm for arm in EXPERIMENT_ARMS
        if arm != "full-agentic-evolved-skill" or evolved_skill_artifact is not None
    ]
    return {
        arm: (lambda model, budget, selected_arm=arm: build(
            selected_arm, model, budget,
        ))
        for arm in selected
    }
