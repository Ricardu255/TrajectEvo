"""The bounded tool-using loop every review role runs inside."""
import json
import time
from typing import Any, Dict, List, Optional

from .context_manager import ContextManager
from .llm import JsonChatClient
from .runtime import RuntimeBudgetExceeded, ToolRegistry
from .telemetry import ExecutionLedger


# A findings JSON smaller than this is likely to truncate mid-object and void
# the whole role run, so it is worth exceeding the remaining token budget.
MIN_OUTPUT_TOKENS = 768

# Final-action payloads by phase.  Real models sometimes emit the phase
# payload without the {"action": "final"} envelope; when one of these keys is
# present and no tool action was requested, the response is treated as final.
FINAL_PAYLOAD_KEYS = frozenset({
    "delegations", "revision_requests", "accepted_finding_indices",
    "findings", "decisions",
})


class BoundedRole:
    def __init__(
        self, name: str, prompt: str, client: JsonChatClient,
        token_budget: int, time_budget: int, max_steps: int = 4,
        context_manager: Optional[ContextManager] = None,
        working_memory_supplier=None, observation_sink=None,
        max_output_tokens: int = 4000,
    ):
        self.name = name
        self.prompt = prompt
        self.client = client
        self.token_budget = token_budget
        self.time_budget = time_budget
        self.max_steps = max_steps
        self.context_manager = context_manager or ContextManager()
        self.working_memory_supplier = working_memory_supplier
        self.observation_sink = observation_sink
        self.max_output_tokens = max(128, int(max_output_tokens))

    def run(
        self, user_context: str, tools: ToolRegistry, ledger: ExecutionLedger,
    ) -> Dict[str, Any]:
        started = time.monotonic()
        observations: List[dict] = []
        starting_tokens = sum(
            item.input_tokens + item.output_tokens
            for item in ledger.model_calls if item.role == self.name
        )
        ledger.trace(
            self.name, "started", token_budget=self.token_budget,
            time_budget_seconds=self.time_budget, tools=tools.names(),
        )
        for step in range(1, self.max_steps + 1):
            elapsed = time.monotonic() - started
            used = sum(
                item.input_tokens + item.output_tokens
                for item in ledger.model_calls if item.role == self.name
            ) - starting_tokens
            if elapsed >= self.time_budget or used >= self.token_budget:
                ledger.trace(self.name, "budget_exhausted", step=step, tokens_used=used)
                raise RuntimeBudgetExceeded("%s budget exhausted" % self.name)
            output_allowance = self.context_manager.output_token_limit(
                self.prompt, min(
                    self.max_output_tokens,
                    max(MIN_OUTPUT_TOKENS, self.token_budget - used),
                )
            )
            current_context = user_context
            if self.working_memory_supplier is not None:
                try:
                    working = self.working_memory_supplier()
                    if working:
                        task_context = json.loads(user_context)
                        task_context["working_memory"] = working
                        current_context = json.dumps(task_context, ensure_ascii=False)
                except Exception as exc:
                    ledger.trace(
                        self.name, "working_memory_unavailable", error=str(exc)[:500],
                    )
            managed, context_stats = self.context_manager.build_managed_context(
                current_context, tools.catalog(), observations,
                max(0, self.token_budget - used),
                max(0, int(self.time_budget - elapsed)),
                system_prompt=self.prompt, max_output_tokens=output_allowance,
            )
            ledger.trace(
                self.name, "context_prepared", step=step,
                estimated_input_tokens=context_stats["estimated_input_tokens_after"],
                input_token_limit=context_stats["input_token_limit"],
                observations_summarized=context_stats["observations"]["summarized"],
                observations_dropped=context_stats["observations"]["dropped"],
            )
            action = self.client.complete_json(
                self.name, self.prompt,
                json.dumps(managed, ensure_ascii=False, default=str),
                ledger, max_tokens=output_allowance,
            )
            kind = str(action.get("action", "")).strip().lower()
            if kind not in {"tool", "final"} and isinstance(action.get("content"), dict):
                # Some models echo the response_format spec as an outer
                # envelope and nest the real action payload under "content".
                action = action["content"]
                kind = str(action.get("action", "")).strip().lower()
            if kind != "tool" and any(key in action for key in FINAL_PAYLOAD_KEYS):
                kind = "final"
            ledger.trace(
                self.name, "autonomous_decision", step=step, action=kind,
                tool=str(action.get("tool", "")), reason=str(action.get("reason", ""))[:500],
            )
            if kind == "final":
                action["_observations"] = observations
                action["_steps"] = step
                ledger.trace(self.name, "finished", step=step)
                return action
            if kind != "tool":
                raise ValueError(
                    "%s returned an invalid action: %s"
                    % (self.name, json.dumps(action, ensure_ascii=False, default=str)[:300])
                )
            tool_name = str(action.get("tool", ""))
            arguments = action.get("arguments") or {}
            try:
                value = tools.invoke(tool_name, arguments)
                observation = {
                    "step": step, "tool": tool_name, "ok": True, "result": value,
                }
            except Exception as exc:
                observation = {
                    "step": step, "tool": tool_name, "ok": False,
                    "error": str(exc)[:1000],
                }
            observations.append(observation)
            if self.observation_sink is not None:
                try:
                    self.observation_sink(self.name, observation)
                except Exception as exc:
                    ledger.trace(
                        self.name, "working_memory_write_failed", error=str(exc)[:500],
                    )
            ledger.trace(
                self.name, "tool_observation", step=step, tool=tool_name,
                ok=observation["ok"],
            )
        ledger.trace(self.name, "budget_exhausted", budget="steps")
        raise RuntimeBudgetExceeded("%s step budget exhausted" % self.name)
