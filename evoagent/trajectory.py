"""Trajectory-level regression evaluation, first-error attribution and release gates.

模块职责（移植自 TrajectIQ，适配 EVO 的真实 Agent 执行轨迹）：
1. 以标准化轨迹（TrajectorySpan 列表）为评测对象，不依赖某个具体 Agent 实现；
2. 对同一评测任务集运行两个版本（baseline / candidate），识别"基线通过、候选失败"
   的回归任务，并按任务类别切片对比质量与成本指标；
3. 对每条回归任务逐 step 对齐工具调用序列，定位第一处分歧（首错归因）；
4. 按 YAML 阈值输出 PASS / WARNING / BLOCK 发布门禁结论，BLOCK 时进程退出码为 1，
   可直接接入 CI。

与 TrajectIQ 的差异：
- 真实对接点是 ``ledger_to_spans``：把 EVO 的 ExecutionLedger 摘要转成标准轨迹，
  因此评测对象可以是真实多角色 Agent，而不是规则模拟器；
- 评测结果支持 JSON 落盘（save_results/load_results），两个版本可分别运行后再比较；
- 内置 build_demo_* 确定性夹具，无需 API Key 即可离线演示完整 PASS/BLOCK 流程。

调用关系：
- 上游：harness/agentic_core 的 ExecutionLedger.summary()，或离线 JSON 结果；
- 下游：CI 发布门禁、进化引擎（evolution.py）的候选版本验证。
"""
import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. 数据模型
# ---------------------------------------------------------------------------
@dataclass
class TrajectorySpan:
    """单步执行轨迹：一次规划、一次工具调用或一次最终回答。"""

    step: int
    kind: str  # "planner" / "tool" / "final"
    name: str
    input: Any
    output: Any
    error: Optional[str] = None
    start_time_ms: int = 0
    end_time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryTask:
    """一条评测任务：输入 + 期望的工具链 / 参数 / 答案断言。"""

    task_id: str
    category: str
    input: str
    expected_tools: Tuple[str, ...]
    expected_arguments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    expected_answer_contains: Tuple[str, ...] = ()
    critical: bool = False
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskEvaluation:
    """单任务评测结果：工具链、参数、答案三组独立断言。"""

    task_id: str
    is_success: bool
    has_correct_tools: bool
    has_correct_arguments: bool
    has_expected_answer: bool
    is_critical: bool
    actual_tools: Tuple[str, ...]
    expected_tools: Tuple[str, ...]


@dataclass(frozen=True)
class VersionMetrics:
    """一个版本在整个任务集上的聚合指标（质量组 + 成本组）。"""

    version: str
    task_count: int
    success_rate: float
    tool_selection_accuracy: float
    tool_argument_accuracy: float
    answer_coverage: float
    average_steps: float
    critical_task_success_rate: float
    average_latency_ms: float
    average_prompt_tokens: float
    average_completion_tokens: float
    average_total_tokens: float
    average_cost_usd: float


@dataclass(frozen=True)
class SliceMetrics:
    """按任务类别切片的对比指标。"""

    category: str
    task_count: int
    baseline_success_rate: float
    candidate_success_rate: float
    regression_count: int


@dataclass(frozen=True)
class RegressionReport:
    """版本回归报告：两侧指标 + 回归任务清单 + 类别切片。"""

    dataset: str
    baseline: VersionMetrics
    candidate: VersionMetrics
    regressions: Tuple[TaskEvaluation, ...]
    slices: Tuple[SliceMetrics, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["deltas"] = {
            "success_rate": self.candidate.success_rate - self.baseline.success_rate,
            "tool_selection_accuracy": (
                self.candidate.tool_selection_accuracy - self.baseline.tool_selection_accuracy
            ),
            "tool_argument_accuracy": (
                self.candidate.tool_argument_accuracy - self.baseline.tool_argument_accuracy
            ),
            "answer_coverage": self.candidate.answer_coverage - self.baseline.answer_coverage,
            "average_latency_ms": (
                self.candidate.average_latency_ms - self.baseline.average_latency_ms
            ),
            "average_total_tokens": (
                self.candidate.average_total_tokens - self.baseline.average_total_tokens
            ),
            "average_cost_usd": self.candidate.average_cost_usd - self.baseline.average_cost_usd,
        }
        return payload


@dataclass(frozen=True)
class FailureAttribution:
    """单条回归任务的首错归因。"""

    task_id: str
    category: str
    step: int
    baseline_span: Optional[str]
    candidate_span: Optional[str]
    reason: str
    confidence: float
    is_critical: bool


@dataclass(frozen=True)
class GateViolation:
    rule: str
    severity: str  # "block" / "warning"
    actual: float
    threshold: float
    message: str


@dataclass(frozen=True)
class TrajectoryGateResult:
    status: str  # "PASS" / "WARNING" / "BLOCK"
    baseline: str
    candidate: str
    violations: Tuple[GateViolation, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 2. 轨迹指标聚合
# ---------------------------------------------------------------------------
def span_metrics(span: Dict[str, Any]) -> Dict[str, float]:
    """从单个 span 读取延迟 / token / 成本，容忍字段缺失。"""
    start = span.get("start_time_ms", 0) or 0
    end = span.get("end_time_ms", start) or start
    duration = span.get("duration_ms")
    if duration is None:
        duration = max(0, end - start)
    prompt = span.get("prompt_tokens", 0) or 0
    completion = span.get("completion_tokens", 0) or 0
    cost = span.get("cost_usd", 0.0) or 0.0
    return {
        "duration_ms": float(duration),
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(prompt + completion),
        "cost_usd": float(cost),
    }


def aggregate_run_metrics(spans: List[Dict[str, Any]]) -> Dict[str, float]:
    """把一条任务轨迹的所有 span 聚合成运行级指标。"""
    metrics = [span_metrics(span) for span in spans]
    if not metrics:
        return {
            "duration_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0,
        }
    return {
        "duration_ms": sum(item["duration_ms"] for item in metrics),
        "prompt_tokens": sum(item["prompt_tokens"] for item in metrics),
        "completion_tokens": sum(item["completion_tokens"] for item in metrics),
        "total_tokens": sum(item["total_tokens"] for item in metrics),
        "cost_usd": round(sum(item["cost_usd"] for item in metrics), 8),
    }


# ---------------------------------------------------------------------------
# 3. 任务评测与版本回归
# ---------------------------------------------------------------------------
def _get_tool_spans(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [span for span in result.get("spans", []) if span.get("kind") == "tool"]


def _expected_arguments_match(expected: Dict[str, Any], actual: Any) -> bool:
    """子集语义：任务声明的每个期望参数都必须在实际入参中出现且相等。

    实际入参携带的额外键（如 request_id、时间戳等与断言无关的字段）不会导致失败，
    从而避免"整个 dict 精确相等"对无关字段过度敏感。
    """
    if not isinstance(actual, dict):
        return False
    for key, expected_value in (expected or {}).items():
        if key not in actual or actual[key] != expected_value:
            return False
    return True


def evaluate_task(*, task: TrajectoryTask, result: Dict[str, Any]) -> TaskEvaluation:
    """三断言评测：工具链完全一致 + 关键参数子集匹配 + 答案包含期望文本。"""
    tool_spans = _get_tool_spans(result)
    actual_tools = tuple(span["name"] for span in tool_spans)
    # ① 工具选择：实际工具序列必须与期望完全一致（顺序敏感）
    has_correct_tools = actual_tools == task.expected_tools
    # ② 工具参数：工具链正确的前提下，逐工具校验"声明的期望参数"是实际入参子集；
    #    任务未规定参数的工具不校验，实际入参里的额外无关字段不判错
    has_correct_arguments = has_correct_tools and all(
        _expected_arguments_match(
            task.expected_arguments.get(span["name"], {}), span.get("input")
        )
        for span in tool_spans
    )
    # ③ 答案覆盖：所有期望子串都必须出现在最终答案中（大小写不敏感）
    answer = str(result.get("answer", ""))
    has_expected_answer = all(
        expected_text.lower() in answer.lower()
        for expected_text in task.expected_answer_contains
    )
    return TaskEvaluation(
        task_id=task.task_id,
        is_success=has_correct_tools and has_correct_arguments and has_expected_answer,
        has_correct_tools=has_correct_tools,
        has_correct_arguments=has_correct_arguments,
        has_expected_answer=has_expected_answer,
        is_critical=task.critical,
        actual_tools=actual_tools,
        expected_tools=task.expected_tools,
    )


def evaluate_results(
    *,
    version_name: str,
    tasks: Tuple[TrajectoryTask, ...],
    results_by_task_id: Dict[str, Dict[str, Any]],
) -> Tuple[VersionMetrics, Tuple[TaskEvaluation, ...]]:
    """评测一个版本在整份任务集上的表现（输入为标准化轨迹集合）。"""
    if not tasks:
        raise ValueError("Cannot evaluate an empty task set.")
    missing = [task.task_id for task in tasks if task.task_id not in results_by_task_id]
    if missing:
        raise ValueError("Missing trajectory results for tasks: %s" % ", ".join(sorted(missing)))
    runs = tuple((task, results_by_task_id[task.task_id]) for task in tasks)
    evaluations = tuple(evaluate_task(task=task, result=result) for task, result in runs)
    task_count = len(evaluations)
    critical_evaluations = tuple(item for item in evaluations if item.is_critical)
    run_metrics = tuple(aggregate_run_metrics(result.get("spans", [])) for _, result in runs)
    metrics = VersionMetrics(
        version=version_name,
        task_count=task_count,
        success_rate=sum(item.is_success for item in evaluations) / task_count,
        tool_selection_accuracy=sum(item.has_correct_tools for item in evaluations) / task_count,
        tool_argument_accuracy=sum(item.has_correct_arguments for item in evaluations) / task_count,
        answer_coverage=sum(item.has_expected_answer for item in evaluations) / task_count,
        average_steps=sum(len(_get_tool_spans(result)) for _, result in runs) / task_count,
        critical_task_success_rate=(
            sum(item.is_success for item in critical_evaluations) / len(critical_evaluations)
            if critical_evaluations else 1.0
        ),
        average_latency_ms=sum(item["duration_ms"] for item in run_metrics) / task_count,
        average_prompt_tokens=sum(item["prompt_tokens"] for item in run_metrics) / task_count,
        average_completion_tokens=sum(item["completion_tokens"] for item in run_metrics) / task_count,
        average_total_tokens=sum(item["total_tokens"] for item in run_metrics) / task_count,
        average_cost_usd=sum(item["cost_usd"] for item in run_metrics) / task_count,
    )
    return metrics, evaluations


def compare_results(
    *,
    baseline_name: str,
    candidate_name: str,
    tasks: Tuple[TrajectoryTask, ...],
    baseline_results: Dict[str, Dict[str, Any]],
    candidate_results: Dict[str, Dict[str, Any]],
    dataset_name: str,
) -> RegressionReport:
    """对比两个版本的标准化轨迹集合，产出回归报告。"""
    if not tasks:
        raise ValueError("Cannot compare versions on an empty task set.")
    task_ids = {task.task_id for task in tasks}
    missing_baseline = sorted(task_ids - set(baseline_results))
    missing_candidate = sorted(task_ids - set(candidate_results))
    coverage_problems = []
    if missing_baseline:
        coverage_problems.append("baseline missing results for: %s" % ", ".join(missing_baseline))
    if missing_candidate:
        coverage_problems.append("candidate missing results for: %s" % ", ".join(missing_candidate))
    if coverage_problems:
        raise ValueError(
            "Trajectory results do not cover the task set; " + "; ".join(coverage_problems)
        )
    baseline_metrics, baseline_evaluations = evaluate_results(
        version_name=baseline_name, tasks=tasks, results_by_task_id=baseline_results
    )
    candidate_metrics, candidate_evaluations = evaluate_results(
        version_name=candidate_name, tasks=tasks, results_by_task_id=candidate_results
    )
    baseline_by_task_id = {item.task_id: item for item in baseline_evaluations}
    # 回归定义：基线成功、候选失败
    regressions = tuple(
        item for item in candidate_evaluations
        if baseline_by_task_id[item.task_id].is_success and not item.is_success
    )
    candidate_by_task_id = {item.task_id: item for item in candidate_evaluations}
    regressed_task_ids = {item.task_id for item in regressions}
    slices: List[SliceMetrics] = []
    for category in sorted({task.category for task in tasks}):
        category_task_ids = tuple(task.task_id for task in tasks if task.category == category)
        baseline_slice = tuple(baseline_by_task_id[task_id] for task_id in category_task_ids)
        candidate_slice = tuple(candidate_by_task_id[task_id] for task_id in category_task_ids)
        slices.append(
            SliceMetrics(
                category=category,
                task_count=len(category_task_ids),
                baseline_success_rate=(
                    sum(item.is_success for item in baseline_slice) / len(baseline_slice)
                ),
                candidate_success_rate=(
                    sum(item.is_success for item in candidate_slice) / len(candidate_slice)
                ),
                regression_count=sum(
                    item.task_id in regressed_task_ids for item in candidate_slice
                ),
            )
        )
    return RegressionReport(
        dataset=dataset_name,
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        regressions=regressions,
        slices=tuple(slices),
    )


# ---------------------------------------------------------------------------
# 4. 首错归因
# ---------------------------------------------------------------------------
def locate_first_error(
    *,
    task: TrajectoryTask,
    baseline_result: Dict[str, Any],
    candidate_result: Dict[str, Any],
) -> FailureAttribution:
    """逐 step 对齐两条轨迹的工具序列，返回第一处分歧（顺序即优先级）。"""
    baseline_tools = _get_tool_spans(baseline_result)
    candidate_tools = _get_tool_spans(candidate_result)
    maximum_steps = max(len(baseline_tools), len(candidate_tools))

    for index in range(maximum_steps):
        baseline_span = baseline_tools[index] if index < len(baseline_tools) else None
        candidate_span = candidate_tools[index] if index < len(candidate_tools) else None
        step = index + 2  # step 1 是 planner，工具从 step 2 开始
        if baseline_span is None:
            # 候选多调了基线没有的工具
            return FailureAttribution(
                task_id=task.task_id, category="tool_selection_error", step=step,
                baseline_span=None,
                candidate_span=candidate_span["name"] if candidate_span else None,
                reason="Candidate invoked an extra tool that is absent from the baseline trajectory.",
                confidence=1.0, is_critical=task.critical,
            )
        if candidate_span is None:
            # 候选漏掉了基线调用过的工具
            return FailureAttribution(
                task_id=task.task_id, category="tool_selection_error", step=step,
                baseline_span=baseline_span["name"], candidate_span=None,
                reason="Candidate omitted a tool required by the baseline trajectory.",
                confidence=1.0, is_critical=task.critical,
            )
        if baseline_span["name"] != candidate_span["name"]:
            # 同一位置选了不同工具
            return FailureAttribution(
                task_id=task.task_id, category="tool_selection_error", step=step,
                baseline_span=baseline_span["name"], candidate_span=candidate_span["name"],
                reason="Candidate selected a different tool at the first divergent step.",
                confidence=1.0, is_critical=task.critical,
            )
        if baseline_span["input"] != candidate_span["input"]:
            # 工具相同但参数不同
            return FailureAttribution(
                task_id=task.task_id, category="tool_argument_error", step=step,
                baseline_span=baseline_span["name"], candidate_span=candidate_span["name"],
                reason="Candidate passed different arguments to the same tool.",
                confidence=1.0, is_critical=task.critical,
            )
        if candidate_span.get("error"):
            # 工具执行报错
            return FailureAttribution(
                task_id=task.task_id, category="tool_execution_error", step=step,
                baseline_span=baseline_span["name"], candidate_span=candidate_span["name"],
                reason="Candidate tool failed with %s." % candidate_span["error"],
                confidence=1.0, is_critical=task.critical,
            )
    # 工具轨迹完全一致但最终答案不达标
    return FailureAttribution(
        task_id=task.task_id, category="generation_error",
        step=len(candidate_tools) + 2,
        baseline_span="final_answer", candidate_span="final_answer",
        reason="Tool trajectory matches the baseline but final answer expectations failed.",
        confidence=0.9, is_critical=task.critical,
    )


def attribute_results(
    *,
    tasks: Tuple[TrajectoryTask, ...],
    report: RegressionReport,
    baseline_results: Dict[str, Dict[str, Any]],
    candidate_results: Dict[str, Dict[str, Any]],
) -> Tuple[FailureAttribution, ...]:
    """对报告中的每条回归任务执行首错归因。"""
    tasks_by_id = {task.task_id: task for task in tasks}
    attributions: List[FailureAttribution] = []
    for regression in report.regressions:
        task = tasks_by_id[regression.task_id]
        attributions.append(
            locate_first_error(
                task=task,
                baseline_result=baseline_results[regression.task_id],
                candidate_result=candidate_results[regression.task_id],
            )
        )
    return tuple(attributions)


# ---------------------------------------------------------------------------
# 5. 发布门禁
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "minimum_success_rate": 0.90,
    "maximum_success_rate_drop": 0.03,
    "minimum_tool_selection_accuracy": 0.95,
    "maximum_critical_task_regressions": 0,
    "maximum_task_regressions": 0,
    "maximum_average_latency_ms": 2000.0,
    "maximum_average_total_tokens": 20000,
    "maximum_average_cost_usd": 0.05,
}


def validate_gate_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """校验门禁配置：thresholds 键名合法、severity_overrides 的规则与级别合法。

    拼错的阈值键若被静默忽略，会让门禁在"以为已收紧"时仍按默认值放行，因此这里
    采取 fail-fast：遇到未知键或非法 severity 直接报错。
    """
    if not isinstance(payload, dict):
        raise ValueError("Gate configuration must be a YAML mapping.")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Gate configuration must contain a thresholds mapping.")
    unknown_thresholds = sorted(set(thresholds) - set(DEFAULT_THRESHOLDS))
    if unknown_thresholds:
        raise ValueError(
            "Unknown gate threshold key(s): %s; allowed keys are: %s"
            % (", ".join(unknown_thresholds), ", ".join(sorted(DEFAULT_THRESHOLDS)))
        )
    overrides = payload.get("severity_overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError("severity_overrides must be a mapping of rule -> severity.")
    unknown_rules = sorted(set(overrides) - set(DEFAULT_THRESHOLDS))
    if unknown_rules:
        raise ValueError(
            "severity_overrides references unknown rule(s): %s" % ", ".join(unknown_rules)
        )
    bad_severity_rules = sorted(
        rule for rule, severity in overrides.items() if severity not in ("block", "warning")
    )
    if bad_severity_rules:
        raise ValueError(
            "severity_overrides values must be 'block' or 'warning'; invalid rules: %s"
            % ", ".join(bad_severity_rules)
        )
    return payload


def load_gate_config(path: Path) -> Dict[str, Any]:
    """加载并校验 YAML 门禁配置（必须包含 thresholds 映射）。"""
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate_gate_config(payload)


def evaluate_gate(*, report: RegressionReport, config: Dict[str, Any]) -> TrajectoryGateResult:
    """按阈值评估候选版本，任一 block 级违规即 BLOCK。"""
    thresholds = config.get("thresholds", DEFAULT_THRESHOLDS)
    severity_overrides = config.get("severity_overrides", {}) or {}
    violations: List[GateViolation] = []

    minimum_success_rate = thresholds.get("minimum_success_rate")
    if minimum_success_rate is not None and report.candidate.success_rate < minimum_success_rate:
        violations.append(GateViolation(
            "minimum_success_rate", "block", report.candidate.success_rate,
            minimum_success_rate,
            "Candidate task success rate is below the required minimum.",
        ))

    maximum_success_rate_drop = thresholds.get("maximum_success_rate_drop")
    success_rate_drop = report.baseline.success_rate - report.candidate.success_rate
    if maximum_success_rate_drop is not None and success_rate_drop > maximum_success_rate_drop:
        violations.append(GateViolation(
            "maximum_success_rate_drop", "block", success_rate_drop,
            maximum_success_rate_drop,
            "Candidate task success rate regressed beyond the allowed drop.",
        ))

    minimum_tool_selection_accuracy = thresholds.get("minimum_tool_selection_accuracy")
    if (
        minimum_tool_selection_accuracy is not None
        and report.candidate.tool_selection_accuracy < minimum_tool_selection_accuracy
    ):
        violations.append(GateViolation(
            "minimum_tool_selection_accuracy", "block",
            report.candidate.tool_selection_accuracy, minimum_tool_selection_accuracy,
            "Candidate tool selection accuracy is below the required minimum.",
        ))

    maximum_critical_task_regressions = thresholds.get("maximum_critical_task_regressions")
    critical_regressions = sum(item.is_critical for item in report.regressions)
    if (
        maximum_critical_task_regressions is not None
        and critical_regressions > maximum_critical_task_regressions
    ):
        violations.append(GateViolation(
            "maximum_critical_task_regressions", "block", critical_regressions,
            maximum_critical_task_regressions,
            "Candidate regressed on more critical tasks than allowed.",
        ))

    maximum_task_regressions = thresholds.get("maximum_task_regressions")
    if maximum_task_regressions is not None and len(report.regressions) > maximum_task_regressions:
        violations.append(GateViolation(
            "maximum_task_regressions", "warning", len(report.regressions),
            maximum_task_regressions,
            "Candidate has more task regressions than the warning threshold.",
        ))

    # 成本组指标默认 warning 级
    metric_rules = (
        ("maximum_average_latency_ms", report.candidate.average_latency_ms, "block",
         "Candidate average latency is above the allowed threshold."),
        ("maximum_average_total_tokens", report.candidate.average_total_tokens, "warning",
         "Candidate average token usage is above the allowed threshold."),
        ("maximum_average_cost_usd", report.candidate.average_cost_usd, "warning",
         "Candidate average estimated cost is above the allowed threshold."),
    )
    for rule, actual, severity, message in metric_rules:
        threshold = thresholds.get(rule)
        if threshold is not None and actual > threshold:
            violations.append(GateViolation(rule, severity, actual, threshold, message))

    # 允许通过 severity_overrides 把任意规则的默认级别改成 block / warning
    # （例如把 maximum_task_regressions 从默认 warning 提升为 block）
    violations = [
        replace(item, severity=severity_overrides.get(item.rule, item.severity))
        for item in violations
    ]

    status = (
        "BLOCK" if any(item.severity == "block" for item in violations)
        else "WARNING" if violations else "PASS"
    )
    return TrajectoryGateResult(
        status=status, baseline=report.baseline.version,
        candidate=report.candidate.version, violations=tuple(violations),
    )


# ---------------------------------------------------------------------------
# 6. Markdown 渲染
# ---------------------------------------------------------------------------
def render_regression_markdown(report: RegressionReport) -> str:
    def format_percent(value: float) -> str:
        return "%.1f%%" % (value * 100)

    metric_rows = (
        ("Task success rate", report.baseline.success_rate, report.candidate.success_rate, True),
        ("Tool selection accuracy", report.baseline.tool_selection_accuracy,
         report.candidate.tool_selection_accuracy, True),
        ("Tool argument accuracy", report.baseline.tool_argument_accuracy,
         report.candidate.tool_argument_accuracy, True),
        ("Answer coverage", report.baseline.answer_coverage,
         report.candidate.answer_coverage, True),
        ("Critical task success rate", report.baseline.critical_task_success_rate,
         report.candidate.critical_task_success_rate, True),
        ("Average latency (ms)", report.baseline.average_latency_ms,
         report.candidate.average_latency_ms, False),
        ("Average total tokens", report.baseline.average_total_tokens,
         report.candidate.average_total_tokens, False),
        ("Average cost (USD)", report.baseline.average_cost_usd,
         report.candidate.average_cost_usd, False),
    )
    lines = [
        "# TrajectEvo Regression Report", "",
        "Baseline: %s" % report.baseline.version,
        "Candidate: %s" % report.candidate.version,
        "Dataset: %s" % report.dataset, "",
        "## Metrics", "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, baseline_value, candidate_value, is_rate in metric_rows:
        baseline_text = format_percent(baseline_value) if is_rate else "%.4f" % baseline_value
        candidate_text = format_percent(candidate_value) if is_rate else "%.4f" % candidate_value
        lines.append(
            "| %s | %s | %s | %+.4f |"
            % (label, baseline_text, candidate_text, candidate_value - baseline_value)
        )
    lines.extend(["", "## Regressions", ""])
    if not report.regressions:
        lines.append("No task regressions detected.")
    else:
        lines.extend(
            "- %s: expected %s, got %s"
            % (item.task_id, ", ".join(item.expected_tools), ", ".join(item.actual_tools))
            for item in report.regressions
        )
    lines.extend([
        "", "## Category slices", "",
        "| Category | Tasks | Baseline | Candidate | Regressions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    lines.extend(
        "| %s | %d | %s | %s | %d |"
        % (
            item.category, item.task_count,
            format_percent(item.baseline_success_rate),
            format_percent(item.candidate_success_rate), item.regression_count,
        )
        for item in report.slices
    )
    return "\n".join(lines) + "\n"


def render_attribution_markdown(attributions: Tuple[FailureAttribution, ...]) -> str:
    lines = ["# TrajectEvo First-Error Diagnostics", ""]
    if not attributions:
        lines.append("No regressions to diagnose.")
    else:
        lines.extend(
            "- %s | step %d | %s | %s -> %s | %s"
            % (
                item.task_id, item.step, item.category,
                item.baseline_span or "none", item.candidate_span or "none", item.reason,
            )
            for item in attributions
        )
    return "\n".join(lines) + "\n"


def render_gate_markdown(
    *, result: TrajectoryGateResult, attributions: Tuple[FailureAttribution, ...]
) -> str:
    lines = [
        "# TrajectEvo Release Gate", "",
        "Status: **%s**" % result.status,
        "Baseline: %s" % result.baseline,
        "Candidate: %s" % result.candidate, "",
    ]
    if not result.violations:
        lines.append("All configured quality thresholds passed.")
        return "\n".join(lines) + "\n"
    lines.extend(["## Gate findings", ""])
    lines.extend(
        "- [%s] %s: actual=%s, threshold=%s. %s"
        % (item.severity.upper(), item.rule, item.actual, item.threshold, item.message)
        for item in result.violations
    )
    if attributions:
        lines.extend(["", "## First-error diagnostics", ""])
        lines.extend(
            "- %s: %s at step %d (%s -> %s)"
            % (
                item.task_id, item.category, item.step,
                item.baseline_span, item.candidate_span,
            )
            for item in attributions
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 7. EVO 真实轨迹适配器：ExecutionLedger -> 标准轨迹
# ---------------------------------------------------------------------------
def ledger_to_spans(
    ledger_summary: Dict[str, Any],
    answer: str = "",
    planner_input: str = "",
) -> Dict[str, Any]:
    """把 ExecutionLedger.summary() 的输出转换成标准评测轨迹。

    EVO 的真实多角色 Agent 每次工具调用都会经 ledger.record_tool 留痕，
    本函数按记录顺序还原 tool span（含参数、成败、耗时与错误），并补 planner /
    final 两个边界 span，使真实 Agent 轨迹可以直接进入 evaluate_results。

    计量口径（串行步骤求和，不重不漏）：第一次模型调用计入 planner span，其余
    模型调用（worker / critic / 综合）合计计入 final span，工具调用各占一个 tool
    span。因此 token / 成本总量等于全部模型调用之和，延迟总量等于全部 LLM 调用与
    工具调用的耗时之和（串行口径下约等于任务墙钟时长）。
    """
    spans: List[Dict[str, Any]] = []
    model_log = ledger_summary.get("model_call_log") or []
    tool_log = ledger_summary.get("tool_call_log") or []

    def _model_usage(item: Dict[str, Any]) -> Tuple[int, int, float, int]:
        return (
            int(item.get("input_tokens", 0) or 0),
            int(item.get("output_tokens", 0) or 0),
            float(item.get("cost_usd", 0.0) or 0.0),
            int(item.get("duration_ms", 0) or 0),
        )

    # planner span 承担第一次模型调用（Lead 规划）的 token / 成本 / 耗时。
    clock_ms = 0
    if model_log:
        p_prompt, p_completion, p_cost, p_duration = _model_usage(model_log[0])
    else:
        p_prompt = p_completion = 0
        p_cost = 0.0
        p_duration = 0
    spans.append(TrajectorySpan(
        step=1, kind="planner", name="plan", input=planner_input, output=None,
        start_time_ms=clock_ms, end_time_ms=clock_ms + p_duration,
        prompt_tokens=p_prompt, completion_tokens=p_completion, cost_usd=p_cost,
    ).to_dict())
    clock_ms += p_duration

    # tool span 承担每次工具调用的耗时；工具本身不重复计 LLM token。
    for index, tool in enumerate(tool_log, start=2):
        duration = int(tool.get("duration_ms", 0) or 0)
        spans.append(TrajectorySpan(
            step=index, kind="tool", name=tool.get("tool", ""),
            input=tool.get("arguments", {}),
            output=tool.get("result_preview", ""),
            error=None if tool.get("ok", True) else tool.get("error", "tool_failed"),
            start_time_ms=clock_ms, end_time_ms=clock_ms + duration,
        ).to_dict())
        clock_ms += duration

    # final span 承担"其余"模型调用（worker / critic / 综合）的合计。
    # 切片 [1:] 是关键：第一次调用已计入 planner，不能再重复累加一次。
    remaining_models = model_log[1:]
    f_prompt = sum(int(item.get("input_tokens", 0) or 0) for item in remaining_models)
    f_completion = sum(int(item.get("output_tokens", 0) or 0) for item in remaining_models)
    f_cost = round(sum(float(item.get("cost_usd", 0.0) or 0.0) for item in remaining_models), 8)
    f_duration = sum(int(item.get("duration_ms", 0) or 0) for item in remaining_models)
    spans.append(TrajectorySpan(
        step=len(spans) + 1, kind="final", name="final_answer", input=None, output=answer,
        start_time_ms=clock_ms, end_time_ms=clock_ms + f_duration,
        prompt_tokens=f_prompt, completion_tokens=f_completion, cost_usd=f_cost,
    ).to_dict())
    return {"answer": answer, "spans": spans}


# ---------------------------------------------------------------------------
# 8. 结果 JSON 持久化（两个版本可分别运行、离线比较）
# ---------------------------------------------------------------------------
def save_results(path: Path, results: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Trajectory result file must be a JSON object keyed by task id.")
    return payload


# ---------------------------------------------------------------------------
# 9. 确定性演示夹具（代码审查场景，无需 API Key）
# ---------------------------------------------------------------------------
# 演示工具集对齐 EVO 的 RepositoryToolSuite：AST 分析 / 符号查询 /
# git 上下文 / 安全扫描 / 类型检查。
_DEMO_TOOL_OUTPUT = {
    "ast_analyze": "AST analysis found no structural defect.",
    "symbol": "Symbol definitions and call sites resolved.",
    "git_context": "Git blame and recent change context collected.",
    "run_scanners": "Security scanners (semgrep/bandit) finished clean.",
    "typecheck": "Typecheck passed on changed files.",
}


def _demo_task(
    task_id: str, category: str, text: str, tools: Tuple[str, ...],
    answer_keyword: str, critical: bool = False,
) -> TrajectoryTask:
    arguments = {tool: {"target": task_id} for tool in tools}
    return TrajectoryTask(
        task_id=task_id, category=category, input=text,
        expected_tools=tools, expected_arguments=arguments,
        expected_answer_contains=(answer_keyword,), critical=critical,
    )


def build_demo_dataset() -> Tuple[TrajectoryTask, ...]:
    """12 条代码审查评测任务，覆盖安全 / 可靠性 / 密钥泄露 / API 兼容四类。"""
    tasks: List[TrajectoryTask] = []
    # 安全类：必须 AST 分析 + 安全扫描（前两条为关键任务）
    for index, critical in enumerate((True, True, False, False), start=1):
        tasks.append(_demo_task(
            "sec_%03d" % index, "security",
            "Review SQL sink in service_%d.py" % index,
            ("ast_analyze", "run_scanners"), "scanners", critical=critical,
        ))
    # 可靠性类：AST 分析 + 符号查询
    for index in range(1, 5):
        tasks.append(_demo_task(
            "rel_%03d" % index, "reliability",
            "Check null dereference path %d" % index,
            ("ast_analyze", "symbol"), "symbol",
        ))
    # 密钥泄露类：git 上下文 + 安全扫描（第一条关键）
    for index, critical in enumerate((True, False), start=1):
        tasks.append(_demo_task(
            "secret_%03d" % index, "secret_leak",
            "Inspect committed credential candidate %d" % index,
            ("git_context", "run_scanners"), "scanners", critical=critical,
        ))
    # API 兼容类：符号查询 + 类型检查
    for index in range(1, 3):
        tasks.append(_demo_task(
            "compat_%03d" % index, "compatibility",
            "Check public API signature change %d" % index,
            ("symbol", "typecheck"), "typecheck",
        ))
    return tuple(tasks)


def _demo_planned_tools(version: str, task: TrajectoryTask) -> List[Tuple[str, Dict[str, Any]]]:
    """回归版本故意把安全类任务的 run_scanners 错换成 symbol。"""
    planned: List[Tuple[str, Dict[str, Any]]] = []
    for tool in task.expected_tools:
        chosen = tool
        if version == "regression" and task.category == "security" and tool == "run_scanners":
            chosen = "symbol"
        planned.append((chosen, task.expected_arguments.get(tool, {"target": task.task_id})))
    return planned


def build_demo_results(version: str) -> Dict[str, Dict[str, Any]]:
    """为指定版本（baseline/regression/fixed）确定性生成全部任务轨迹。"""
    results: Dict[str, Dict[str, Any]] = {}
    for task in build_demo_dataset():
        planned = _demo_planned_tools(version, task)
        spans: List[Dict[str, Any]] = [TrajectorySpan(
            step=1, kind="planner", name="plan", input=task.input, output=None,
            start_time_ms=0, end_time_ms=12, prompt_tokens=16, completion_tokens=8,
            cost_usd=0.000006,
        ).to_dict()]
        clock_ms = 12
        answer_parts: List[str] = []
        for step, (tool, arguments) in enumerate(planned, start=2):
            duration = 8 + len(tool) % 5
            output = _DEMO_TOOL_OUTPUT.get(tool, "Tool output unavailable.")
            spans.append(TrajectorySpan(
                step=step, kind="tool", name=tool, input=arguments, output=output,
                start_time_ms=clock_ms, end_time_ms=clock_ms + duration,
                prompt_tokens=6, completion_tokens=4, cost_usd=0.000003,
            ).to_dict())
            clock_ms += duration
            answer_parts.append(output)
        # 答案包含期望关键词（回归版本仅工具选择错误，答案断言仍成立，便于展示三断言独立性）
        answer = " ".join(answer_parts)
        spans.append(TrajectorySpan(
            step=len(spans) + 1, kind="final", name="final_answer",
            input=None, output=answer, start_time_ms=clock_ms,
            end_time_ms=clock_ms + 10, prompt_tokens=10, completion_tokens=10,
            cost_usd=0.0000075,
        ).to_dict())
        results[task.task_id] = {"version": version, "answer": answer, "spans": spans}
    return results


DEMO_VERSIONS = ("baseline", "regression", "fixed")


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------
def _write_or_print(rendered: str, output: Optional[Path]) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="" if rendered.endswith("\n") else "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrajectEvo trajectory regression, attribution and release gate"
    )
    subparsers = parser.add_subparsers(dest="command")

    compare_parser = subparsers.add_parser("compare", help="compare two trajectory result files")
    compare_parser.add_argument("--baseline", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--dataset", default="trajectory_dataset")
    compare_parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    compare_parser.add_argument("--output", type=Path)

    gate_parser = subparsers.add_parser("gate", help="evaluate a release gate from result files")
    gate_parser.add_argument("--baseline", required=True, type=Path)
    gate_parser.add_argument("--candidate", required=True, type=Path)
    gate_parser.add_argument("--config", type=Path)
    gate_parser.add_argument("--dataset", default="trajectory_dataset")
    gate_parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    gate_parser.add_argument("--output", type=Path)

    subparsers.add_parser("demo", help="run the built-in deterministic demo (baseline vs regression vs fixed)")

    args = parser.parse_args()

    if args.command == "demo":
        tasks = build_demo_dataset()
        for candidate in ("regression", "fixed"):
            report = compare_results(
                baseline_name="baseline", candidate_name=candidate, tasks=tasks,
                baseline_results=build_demo_results("baseline"),
                candidate_results=build_demo_results(candidate),
                dataset_name="code_review_demo",
            )
            attributions = attribute_results(
                tasks=tasks, report=report,
                baseline_results=build_demo_results("baseline"),
                candidate_results=build_demo_results(candidate),
            )
            config = {"thresholds": DEFAULT_THRESHOLDS}
            gate = evaluate_gate(report=report, config=config)
            print(render_regression_markdown(report))
            print(render_attribution_markdown(attributions))
            print(render_gate_markdown(result=gate, attributions=attributions))
        return

    if args.command in ("compare", "gate"):
        # 文件对比模式：任务集由内置数据集承担（真实使用时可替换为加载的标注任务集）
        tasks = build_demo_dataset()
        baseline_results = load_results(args.baseline)
        candidate_results = load_results(args.candidate)
        report = compare_results(
            baseline_name="baseline", candidate_name="candidate", tasks=tasks,
            baseline_results=baseline_results, candidate_results=candidate_results,
            dataset_name=args.dataset,
        )
        if args.command == "compare":
            rendered = (
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
                if args.format == "json" else render_regression_markdown(report)
            )
            _write_or_print(rendered, args.output)
            return
        config = (
            load_gate_config(args.config) if args.config
            else {"thresholds": DEFAULT_THRESHOLDS}
        )
        gate = evaluate_gate(report=report, config=config)
        attributions = attribute_results(
            tasks=tasks, report=report,
            baseline_results=baseline_results, candidate_results=candidate_results,
        )
        rendered = (
            json.dumps(gate.to_dict(), ensure_ascii=False, indent=2)
            if args.format == "json"
            else render_gate_markdown(result=gate, attributions=attributions)
        )
        _write_or_print(rendered, args.output)
        if gate.status == "BLOCK":
            raise SystemExit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
