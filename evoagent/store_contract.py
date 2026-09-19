"""Shared persistence contract for every task-store backend.

``TaskStore`` (SQLite) and ``PostgresTaskStore`` both implement
``StoreProtocol``.  The protocol makes the contract explicit so that a
backend missing a method is caught by the contract test-suite in
``tests/test_store_contract.py`` (which runs the same behavioural tests
against every available backend) instead of failing silently at runtime.
"""
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .models import ReviewReport, TraceEvent


@runtime_checkable
class StoreProtocol(Protocol):
    """Persistence operations every task-store backend must provide."""

    # Task lifecycle ---------------------------------------------------
    def create(
        self, task_id: str, repository: str, pull_request: Optional[int],
        payload: Dict[str, Any], tenant_id: str = "default",
    ) -> None:
        ...

    def transition(self, task_id: str, event: TraceEvent) -> None:
        ...

    def succeed(self, task_id: str, report: ReviewReport, event: TraceEvent) -> None:
        ...

    def fail(self, task_id: str, error: str, event: TraceEvent) -> None:
        ...

    def get(
        self, task_id: str, tenant_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ...

    def list_tasks(
        self, limit: int = 50, tenant_id: Optional[str] = None,
    ) -> list:
        ...

    def save_checkpoint(
        self, task_id: str, node: str, state: Dict[str, Any],
        status: str = "completed", attempt: int = 1, error: str = "",
    ) -> None:
        ...

    def load_checkpoints(self, task_id: str) -> Dict[str, Dict[str, Any]]:
        ...

    def save_task_payload(self, task_id: str, diff: str) -> None:
        ...

    def update_task_input(self, task_id: str, updates: Dict[str, Any]) -> None:
        ...

    def get_task_payload(self, task_id: str) -> Optional[str]:
        ...

    def request_cancel(
        self, task_id: str, tenant_id: Optional[str] = None,
    ) -> bool:
        ...

    def is_cancelled(self, task_id: str) -> bool:
        ...

    def cancel(self, task_id: str, event: TraceEvent) -> None:
        ...

    # Agent collaboration and memory -----------------------------------
    def record_agent_message(
        self, task_id: str, message: Dict[str, Any],
    ) -> None:
        ...

    def save_agent_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def list_agent_memories(
        self, tenant_id: str, repository: str, scopes: tuple, limit: int = 100,
    ) -> list:
        ...

    def delete_agent_memories(
        self, task_id: str = "", scope: str = "",
    ) -> int:
        ...

    def purge_expired_agent_memories(self) -> int:
        ...

    # Failure cases -----------------------------------------------------
    def record_failure_case(
        self, task_id: str, category: str, payload: Dict[str, Any],
    ) -> None:
        ...

    def list_failure_cases(
        self, unresolved_only: bool = False, limit: int = 100,
        tenant_id: Optional[str] = None,
    ) -> list:
        ...

    def list_task_failure_cases(
        self, task_id: str, tenant_id: Optional[str] = None,
    ) -> list:
        ...

    def resolve_failure_cases(self, case_ids: list) -> None:
        ...

    # Prompt evolution --------------------------------------------------
    def save_evaluation_case(
        self, name: str, split: str, diff: str, expected: list,
        source: str = "manual", active: bool = True,
    ) -> Dict[str, Any]:
        ...

    def list_evaluation_cases(
        self, split: Optional[str] = None, active_only: bool = True,
        limit: int = 100,
    ) -> list:
        ...

    def save_evolution_run(self, run: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def list_evolution_runs(self, limit: int = 50) -> list:
        ...

    def update_evolution_run(
        self, run_id: str, decision: str, metrics: Dict[str, Any],
    ) -> bool:
        ...

    # Skill versions and artifacts --------------------------------------
    def save_skill_version(
        self, skill_name: str, prompt: str, score: float,
        activate: bool = False,
    ) -> Dict[str, Any]:
        ...

    def get_active_skill_version(
        self, skill_name: str,
    ) -> Optional[Dict[str, Any]]:
        ...

    def list_skill_versions(self, skill_name: str) -> list:
        ...

    def activate_skill_version(self, skill_name: str, version: int) -> bool:
        ...

    def save_skill_artifact(
        self, skill_name: str, artifact: Dict[str, Any], score: float,
        activate: bool = False, tenant_id: str = "default",
    ) -> Dict[str, Any]:
        ...

    def get_active_skill_artifact(
        self, skill_name: str, tenant_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        ...

    def list_active_skill_artifacts(
        self, tenant_id: str = "default",
    ) -> list:
        ...

    def list_skill_artifact_versions(
        self, skill_name: str, tenant_id: str = "default",
    ) -> list:
        ...

    def activate_skill_artifact(
        self, skill_name: str, version: int, tenant_id: str = "default",
    ) -> bool:
        ...

    def save_skill_evolution_run(self, run: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def list_skill_evolution_runs(
        self, limit: int = 50, tenant_id: Optional[str] = None,
    ) -> list:
        ...

    # GitHub installations -----------------------------------------------
    def save_installation(
        self, installation_id: int, account_login: str,
        tenant_id: str = "default",
    ) -> None:
        ...

    def installation_tenant(self, installation_id: int) -> Optional[str]:
        ...

    # Webhook idempotency -------------------------------------------------
    def claim_webhook(
        self, delivery_id: str, tenant_id: str, event_type: str,
        payload_sha256: str,
    ) -> bool:
        ...

    def complete_webhook(self, delivery_id: str, task_id: Optional[str]) -> None:
        ...

    def get_webhook(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        ...

    # Auth, RBAC and audit -------------------------------------------------
    def create_user(
        self, user_id: str, username: str, password_hash: str,
        tenant_id: str, role: str,
    ) -> None:
        ...

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        ...

    def grant_repository(
        self, tenant_id: str, repository: str, auto_fix: bool = False,
    ) -> None:
        ...

    def repository_allowed(
        self, tenant_id: str, repository: str, require_auto_fix: bool = False,
    ) -> bool:
        ...

    def audit(
        self, tenant_id: str, actor: str, action: str, resource: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    def list_audit(self, tenant_id: str, limit: int = 100) -> list:
        ...

    # Progressive delivery ---------------------------------------------------
    def save_deployment(
        self, tenant_id: str, skill_name: str, config: Dict[str, Any],
    ) -> None:
        ...

    def get_deployment(
        self, tenant_id: str, skill_name: str,
    ) -> Optional[Dict[str, Any]]:
        ...

    def record_deployment_result(
        self, tenant_id: str, skill_name: str, failed: bool,
    ) -> Optional[Dict[str, Any]]:
        ...

    def record_shadow_observation(
        self, tenant_id: str, skill_name: str, task_id: str, lane: str,
        primary: Dict[str, Any], candidate: Optional[Dict[str, Any]],
        disagreement: float, candidate_failed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        ...

    def list_release_observations(
        self, tenant_id: str, skill_name: str, limit: int = 100,
    ) -> list:
        ...

    # Observability ------------------------------------------------------------
    def create_alert(
        self, tenant_id: str, alert_key: str, severity: str, message: str,
    ) -> None:
        ...

    def list_alerts(self, tenant_id: str, limit: int = 100) -> list:
        ...

    def dashboard_stats(
        self, tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...
