"""Product-backed agentic evaluation suite for labelled PRs.

The reviewer arms under evaluation (topologies, single-model baseline and
factories) live in ``evaluation_arms``; this module owns the scoring side:
dataset validation, the production harness, bootstrap comparison and the
fair ablation suite.
"""
from collections import Counter
import random
import time
from typing import Any, Callable, Dict, List, Mapping

from .evaluation_arms import (  # noqa: F401 - re-exported for compatibility
    ARM_TOPOLOGY,
    EXPERIMENT_ARMS,
    REQUIRED_ARMS,
    SINGLE_REVIEW_PROMPT,
    ProductArmReviewer,
    SingleModelReviewer,
    experiment_reviewer_factories,
    product_reviewer_factories,
)
from .evaluation_harness import EndToEndEvaluationHarness, dataset_fingerprint, one_to_one_match
from .llm import JsonChatClient


def validate_real_dataset(cases: List[dict], minimum_cases: int = 300) -> Dict[str, Any]:
    repositories_by_split = {}
    cases_by_split = Counter()
    source_kinds = set()
    for case in cases:
        split = str(case.get("split", ""))
        if split not in {"train", "validation", "holdout"}:
            raise ValueError("every case must use train, validation or holdout split")
        repositories_by_split.setdefault(split, set()).add(str(case.get("repository", "")))
        cases_by_split[split] += 1
        source_kinds.add(str((case.get("source") or {}).get("kind", "unknown")))
        if not isinstance(case.get("expected_findings"), list):
            raise ValueError("every real PR must include human expected_findings")
        for finding in case["expected_findings"]:
            if "should_comment" not in finding:
                raise ValueError("human labels must include should_comment")
            if not all(key in finding for key in ("severity", "path")) or not (
                "line" in finding or "start_line" in finding
            ):
                raise ValueError(
                    "human labels require severity, path and line/start_line"
                )
    overlaps = {}
    splits = sorted(repositories_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            shared = repositories_by_split[left].intersection(repositories_by_split[right])
            if shared:
                overlaps["%s:%s" % (left, right)] = sorted(shared)
    public_or_historical = source_kinds.issubset({
        "public-github-pr", "private-historical-pr",
    }) and bool(source_kinds)
    gates = {
        "minimum_300_cases": len(cases) >= minimum_cases,
        "real_provenance": public_or_historical,
        "repository_isolation": not overlaps,
        "train_present": bool(repositories_by_split.get("train")),
        "validation_present": bool(repositories_by_split.get("validation")),
        "hidden_holdout_present": bool(repositories_by_split.get("holdout")),
    }
    return {
        "ready": all(gates.values()), "gates": gates, "cases": len(cases),
        "repositories": len({str(case.get("repository")) for case in cases}),
        "repositories_by_split": {
            key: len(value) for key, value in repositories_by_split.items()
        },
        "cases_by_split": {
            key: int(cases_by_split.get(key, 0))
            for key in ("train", "validation", "holdout")
        },
        "repository_overlap": overlaps, "source_kinds": sorted(source_kinds),
        "dataset_sha256": dataset_fingerprint(cases),
    }


class ProductionEvaluationHarness(EndToEndEvaluationHarness):
    def _run_case(self, reviewer, case):
        class RecordingReviewer:
            def __init__(self, delegate):
                self.delegate = delegate
                self.name = delegate.name
                self.findings = []

            def review(self, diff, parsed):
                self.findings = self.delegate.review(diff, parsed)
                return self.findings

            def review_case(self, case, parsed):
                self.findings = self.delegate.review_case(case, parsed)
                return self.findings

        recording = RecordingReviewer(reviewer)
        started = time.monotonic()
        result = super()._run_case(recording, case)
        result.update({
            "invalid_comments": result["fp"],
            "exact_location_hits": 0,
            "evidence_hits": 0,
            "accepted_comments": int(case.get("accepted_comments", 0) or 0),
            "closed_comments": int(case.get("closed_comments", 0) or 0),
            "cost_usd": 0.0,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model_roles": {},
            "critic_accepted": 0,
            "critic_rejected": 0,
            "revision_requests": 0,
            "revision_results": 0,
        })
        if result["execution_success"]:
            findings = recording.findings
            expected = [
                item for item in case["expected_findings"]
                if bool(item.get("should_comment", True))
            ]
            matches = one_to_one_match(
                expected, findings, self.line_tolerance
            )
            for match in matches:
                finding = findings[match.predicted_index]
                result["exact_location_hits"] += int(match.location_distance == 0)
                result["evidence_hits"] += int(bool(
                    finding.evidence_refs or finding.call_chain or finding.evidence.strip()
                ))
            execution = reviewer.evaluation_execution() or {}
            result["cost_usd"] = float(execution.get("cost_usd", 0) or 0)
            result["latency_ms"] = int(execution.get("duration_ms", result["latency_ms"]))
            result["llm_calls"] = int(execution.get("llm_calls", 0) or 0)
            result["input_tokens"] = int(execution.get("input_tokens", 0) or 0)
            result["output_tokens"] = int(execution.get("output_tokens", 0) or 0)
            result["total_tokens"] = int(execution.get("total_tokens", 0) or 0)
            result["model_roles"] = dict(Counter(
                str(item.get("role"))
                for item in execution.get("model_call_log") or []
            ))
            collaboration = reviewer.evaluation_collaboration() or {}
            decisions = (
                list(collaboration.get("critic_decisions") or [])
                if "critic" in set(collaboration.get("roles") or []) else []
            )
            result["critic_accepted"] = sum(
                bool(item.get("accepted")) for item in decisions
            )
            result["critic_rejected"] = sum(
                not bool(item.get("accepted")) for item in decisions
            )
            result["revision_requests"] = sum(
                len(item.get("revision_requests") or [])
                for item in (collaboration.get("lead") or {}).get("assessments") or []
            )
            result["revision_results"] = len(
                collaboration.get("revision_results") or []
            )
        return result

    @staticmethod
    def _empty_totals():
        values = EndToEndEvaluationHarness._empty_totals()
        values.update({
            "invalid_comments": 0, "exact_location_hits": 0, "evidence_hits": 0,
            "accepted_comments": 0, "closed_comments": 0,
            "latency_ms": 0, "cost_microusd": 0,
            "llm_calls": 0, "input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0,
            "critic_accepted": 0, "critic_rejected": 0,
            "revision_requests": 0, "revision_results": 0,
        })
        return values

    @staticmethod
    def _accumulate(totals, result):
        EndToEndEvaluationHarness._accumulate(totals, result)
        for field in (
            "invalid_comments", "exact_location_hits", "evidence_hits",
            "accepted_comments", "closed_comments", "latency_ms",
            "llm_calls", "input_tokens", "output_tokens", "total_tokens",
            "critic_accepted", "critic_rejected",
            "revision_requests", "revision_results",
        ):
            totals[field] += int(result.get(field, 0))
        totals["cost_microusd"] += int(float(result.get("cost_usd", 0)) * 1_000_000)

    @staticmethod
    def _metrics(totals):
        values = EndToEndEvaluationHarness._metrics(totals)
        cases = totals["cases"] or 1
        tp = totals["tp"] or 1
        commented = totals["accepted_comments"] + totals["closed_comments"]
        values.update({
            "invalid_comments_per_pr": round(totals["invalid_comments"] / cases, 4),
            "exact_line_accuracy": round(totals["exact_location_hits"] / tp, 4),
            "evidence_accuracy": round(totals["evidence_hits"] / tp, 4),
            "comment_acceptance_rate": round(
                totals["accepted_comments"] / commented, 4
            ) if commented else None,
            "average_cost_usd_per_pr": round(
                totals["cost_microusd"] / 1_000_000 / cases, 8
            ),
            "average_latency_ms_per_pr": round(totals["latency_ms"] / cases, 2),
            "average_llm_calls_per_pr": round(totals["llm_calls"] / cases, 4),
            "average_input_tokens_per_pr": round(totals["input_tokens"] / cases, 2),
            "average_output_tokens_per_pr": round(totals["output_tokens"] / cases, 2),
            "average_total_tokens_per_pr": round(totals["total_tokens"] / cases, 2),
            "failure_rate": round(
                1 - totals["execution_successes"] / cases, 4
            ),
            "critic_acceptance_rate": round(
                totals["critic_accepted"]
                / (totals["critic_accepted"] + totals["critic_rejected"]), 4
            ) if totals["critic_accepted"] + totals["critic_rejected"] else None,
            "critic_accepted_per_pr": round(totals["critic_accepted"] / cases, 4),
            "critic_rejected_per_pr": round(totals["critic_rejected"] / cases, 4),
            "revision_requests_per_pr": round(totals["revision_requests"] / cases, 4),
            "revision_results_per_pr": round(totals["revision_results"] / cases, 4),
        })
        return values


DEFAULT_COMPARISON_METRICS = (
    "f1", "precision", "recall", "high_risk_recall",
    "severity_accuracy", "clean_accuracy", "exact_line_accuracy",
    "evidence_accuracy", "invalid_comments_per_pr",
    "average_total_tokens_per_pr", "average_latency_ms_per_pr",
    "average_cost_usd_per_pr", "failure_rate",
)


def paired_bootstrap_comparison(
    left: dict, right: dict, iterations: int = 2000, seed: int = 20260819,
    metrics=DEFAULT_COMPARISON_METRICS,
) -> dict:
    """Paired case bootstrap for two reports evaluated in identical order."""
    left_cases = left["case_results"]
    right_cases = right["case_results"]
    if [item["id"] for item in left_cases] != [item["id"] for item in right_cases]:
        raise ValueError("paired comparison requires identical ordered case ids")
    count = len(left_cases)
    iterations = max(200, int(iterations)) if count else 0
    output = {}
    for metric_index, metric in enumerate(metrics):
        if not count:
            output[metric] = {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 0}
            continue
        rng = random.Random(int(seed) + metric_index)
        deltas = []
        for _ in range(iterations):
            left_totals = ProductionEvaluationHarness._empty_totals()
            right_totals = ProductionEvaluationHarness._empty_totals()
            for _sample in range(count):
                index = rng.randrange(count)
                ProductionEvaluationHarness._accumulate(left_totals, left_cases[index])
                ProductionEvaluationHarness._accumulate(right_totals, right_cases[index])
            left_value = ProductionEvaluationHarness._metrics(left_totals)[metric]
            right_value = ProductionEvaluationHarness._metrics(right_totals)[metric]
            deltas.append(float(right_value) - float(left_value))
        deltas.sort()
        lower = deltas[int((len(deltas) - 1) * 0.025)]
        upper = deltas[int((len(deltas) - 1) * 0.975)]
        point = float(right["metrics"][metric]) - float(left["metrics"][metric])
        output[metric] = {
            "delta": round(point, 4),
            "ci95": [round(lower, 4), round(upper, 4)],
            "iterations": iterations,
        }
    return output


class FairAblationSuite:
    """Run fair, paired collaboration ablations on an identical case order."""

    def __init__(
        self, reviewer_factories: Mapping[str, Callable[[str, int], Any]],
        model: str, token_budget: int, require_production_ready: bool = True,
        bootstrap_iterations: int = 2000, bootstrap_seed: int = 20260819,
    ):
        missing = set(REQUIRED_ARMS).difference(reviewer_factories)
        if missing:
            raise ValueError("missing ablation arms: %s" % ", ".join(sorted(missing)))
        unknown = set(reviewer_factories).difference(EXPERIMENT_ARMS)
        if unknown:
            raise ValueError("unknown ablation arms: %s" % ", ".join(sorted(unknown)))
        self.factories = reviewer_factories
        self.arm_order = [name for name in EXPERIMENT_ARMS if name in reviewer_factories]
        self.model = model
        self.token_budget = token_budget
        self.require_production_ready = bool(require_production_ready)
        self.bootstrap_iterations = max(200, int(bootstrap_iterations))
        self.bootstrap_seed = int(bootstrap_seed)

    @staticmethod
    def _role_totals(case_results: List[dict]) -> dict:
        totals = Counter()
        for case in case_results:
            totals.update(case.get("model_roles") or {})
        return dict(sorted(totals.items()))

    def _paired_delta(
        self, left: dict, right: dict, metric: str, seed_offset: int,
    ) -> dict:
        left_cases = left["case_results"]
        right_cases = right["case_results"]
        if [item["id"] for item in left_cases] != [item["id"] for item in right_cases]:
            raise ValueError("paired comparison requires identical ordered case ids")
        count = len(left_cases)
        if not count:
            return {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 0}
        rng = random.Random(self.bootstrap_seed + seed_offset)
        deltas = []
        for _ in range(self.bootstrap_iterations):
            left_totals = ProductionEvaluationHarness._empty_totals()
            right_totals = ProductionEvaluationHarness._empty_totals()
            for _sample in range(count):
                index = rng.randrange(count)
                ProductionEvaluationHarness._accumulate(left_totals, left_cases[index])
                ProductionEvaluationHarness._accumulate(right_totals, right_cases[index])
            left_value = ProductionEvaluationHarness._metrics(left_totals)[metric]
            right_value = ProductionEvaluationHarness._metrics(right_totals)[metric]
            deltas.append(float(right_value) - float(left_value))
        deltas.sort()
        lower = deltas[int((len(deltas) - 1) * 0.025)]
        upper = deltas[int((len(deltas) - 1) * 0.975)]
        point = float(right["metrics"][metric]) - float(left["metrics"][metric])
        return {
            "delta": round(point, 4),
            "ci95": [round(lower, 4), round(upper, 4)],
            "iterations": self.bootstrap_iterations,
        }

    def _comparison(self, left: dict, right: dict, seed_offset: int) -> dict:
        return paired_bootstrap_comparison(
            left, right, self.bootstrap_iterations,
            self.bootstrap_seed + seed_offset,
        )

    @staticmethod
    def _split_view(arm: dict, split: str) -> dict:
        return {
            "metrics": arm["by_split"][split],
            "case_results": [
                item for item in arm["case_results"] if item["split"] == split
            ],
        }

    def run(self, cases: List[dict]) -> Dict[str, Any]:
        readiness = validate_real_dataset(cases)
        if self.require_production_ready and not readiness["ready"]:
            raise ValueError("real PR dataset failed readiness gates: %s" % readiness["gates"])
        harness = ProductionEvaluationHarness()
        arms = {}
        for name in self.arm_order:
            reviewer = self.factories[name](self.model, self.token_budget)
            arms[name] = harness.run(reviewer, cases, name)
            engine = (
                reviewer.agentic if isinstance(reviewer, ProductArmReviewer) else reviewer
            )
            arms[name]["fairness"] = {
                "model": self.model, "token_budget_per_pr": self.token_budget,
                "product_runtime": type(engine).__name__,
                "configuration": reviewer.evaluation_config(),
            }
            arms[name]["execution"] = {
                "model_role_calls": self._role_totals(arms[name]["case_results"]),
                "average_llm_calls_per_pr": arms[name]["metrics"]["average_llm_calls_per_pr"],
                "average_total_tokens_per_pr": arms[name]["metrics"]["average_total_tokens_per_pr"],
                "average_latency_ms_per_pr": arms[name]["metrics"]["average_latency_ms_per_pr"],
                "average_cost_usd_per_pr": arms[name]["metrics"]["average_cost_usd_per_pr"],
                "critic_acceptance_rate": arms[name]["metrics"]["critic_acceptance_rate"],
                "critic_rejected_per_pr": arms[name]["metrics"]["critic_rejected_per_pr"],
                "revision_requests_per_pr": arms[name]["metrics"]["revision_requests_per_pr"],
                "revision_results_per_pr": arms[name]["metrics"]["revision_results_per_pr"],
            }
        no_critic_holdout = self._split_view(
            arms["multi-llm-no-critic"], "holdout",
        )
        full_holdout = self._split_view(arms["full-agentic"], "holdout")
        candidate = full_holdout["metrics"]
        critic_comparison = self._comparison(
            no_critic_holdout, full_holdout, 200,
        )
        no_critic = no_critic_holdout["metrics"]
        critic_false_positive_non_regression = (
            candidate["invalid_comments_per_pr"]
            <= no_critic["invalid_comments_per_pr"]
        )
        critic_recall_non_regression = candidate["recall"] >= no_critic["recall"] - 0.01
        critic_statistically_positive = (
            critic_comparison["f1"]["ci95"][0] > 0
            or (
                critic_comparison["precision"]["ci95"][0] > 0
                and critic_recall_non_regression
            )
        )
        comparisons = {
            "scope": "hidden-holdout",
            "critic_vs_no_critic": critic_comparison,
        }
        pair_specs = (
            ("scanner_vs_single", "single-llm", "single-llm-scanner", 400),
            (
                "multi_agent_vs_single_scanner", "single-llm-scanner",
                "multi-llm-no-critic", 600,
            ),
            (
                "evolved_skill_vs_full_agentic", "full-agentic",
                "full-agentic-evolved-skill", 800,
            ),
        )
        for label, left_name, right_name, seed_offset in pair_specs:
            if left_name in arms and right_name in arms:
                comparisons[label] = self._comparison(
                    self._split_view(arms[left_name], "holdout"),
                    self._split_view(arms[right_name], "holdout"),
                    seed_offset,
                )
        return {
            "schema_version": 5, "dataset": readiness, "arms": arms,
            "requested_arms": list(self.arm_order),
            "omitted_arms": [name for name in EXPERIMENT_ARMS if name not in arms],
            "comparisons": comparisons,
            "critic_gate": {
                "passed": bool(
                    readiness["ready"] and critic_statistically_positive
                    and critic_false_positive_non_regression
                    and critic_recall_non_regression
                ),
                "statistically_positive": critic_statistically_positive,
                "false_positive_non_regression": critic_false_positive_non_regression,
                "recall_non_regression_with_1pp_tolerance": critic_recall_non_regression,
                "production_dataset_ready": readiness["ready"],
                "decision": (
                    "keep-critic" if (
                        readiness["ready"] and critic_statistically_positive
                        and critic_false_positive_non_regression
                        and critic_recall_non_regression
                    ) else "critic-not-proven"
                ),
            },
            "claim_scope": (
                "Evidence applies to this labelled holdout and model version; "
                "it does not prove universal superiority."
            ),
        }
