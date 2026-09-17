"""TrajectEvo 轨迹回归层测试：三断言、回归识别、首错归因、门禁与 Ledger 适配。"""
import json
import tempfile
import unittest
from pathlib import Path

from evoagent.trajectory import (
    DEFAULT_THRESHOLDS,
    FailureAttribution,
    TrajectorySpan,
    TrajectoryTask,
    aggregate_run_metrics,
    attribute_results,
    build_demo_dataset,
    build_demo_results,
    compare_results,
    evaluate_gate,
    evaluate_results,
    evaluate_task,
    ledger_to_spans,
    load_results,
    locate_first_error,
    save_results,
    validate_gate_config,
)


def make_result(tools, answer="ok", errors=None):
    """用 (工具名, 入参) 列表快速构造一条标准轨迹。"""
    errors = errors or {}
    spans = [TrajectorySpan(1, "planner", "plan", "task", None).to_dict()]
    answer_parts = []
    for step, (name, arguments) in enumerate(tools, start=2):
        output = answer if name in errors or name not in _OUTPUT else _OUTPUT[name]
        spans.append(TrajectorySpan(
            step, "tool", name, arguments, output,
            error=errors.get(name),
            start_time_ms=(step - 2) * 10, end_time_ms=(step - 1) * 10,
            prompt_tokens=5, completion_tokens=3, cost_usd=0.001,
        ).to_dict())
        answer_parts.append(output)
    spans.append(TrajectorySpan(
        len(spans) + 1, "final", "final_answer", None, answer
    ).to_dict())
    return {"answer": answer, "spans": spans}


_OUTPUT = {
    "ast_analyze": "ast ok",
    "symbol": "symbol ok",
    "run_scanners": "scanners ok",
    "typecheck": "typecheck ok",
}


class EvaluateTaskTests(unittest.TestCase):
    def _task(self, tools, arguments=None, answer_contains=("ok",)):
        return TrajectoryTask(
            task_id="t1", category="security", input="review",
            expected_tools=tuple(tools),
            expected_arguments=arguments or {name: {"target": "t1"} for name in tools},
            expected_answer_contains=answer_contains,
        )

    def test_all_correct_is_success(self):
        task = self._task(["ast_analyze", "run_scanners"])
        result = make_result([
            ("ast_analyze", {"target": "t1"}),
            ("run_scanners", {"target": "t1"}),
        ], answer="ast ok scanners ok")
        evaluation = evaluate_task(task=task, result=result)
        self.assertTrue(evaluation.is_success)
        self.assertTrue(evaluation.has_correct_tools)
        self.assertTrue(evaluation.has_correct_arguments)
        self.assertTrue(evaluation.has_expected_answer)

    def test_wrong_tool_sequence_fails_selection(self):
        task = self._task(["ast_analyze", "run_scanners"])
        result = make_result([
            ("ast_analyze", {"target": "t1"}),
            ("symbol", {"target": "t1"}),
        ], answer="ast ok symbol ok")
        evaluation = evaluate_task(task=task, result=result)
        self.assertFalse(evaluation.is_success)
        self.assertFalse(evaluation.has_correct_tools)
        # 工具链错误时参数断言不再成立（短路）
        self.assertFalse(evaluation.has_correct_arguments)

    def test_wrong_arguments_fails_argument_check(self):
        task = self._task(["ast_analyze"])
        result = make_result([("ast_analyze", {"target": "OTHER"})], answer="ast ok")
        evaluation = evaluate_task(task=task, result=result)
        self.assertTrue(evaluation.has_correct_tools)
        self.assertFalse(evaluation.has_correct_arguments)
        self.assertFalse(evaluation.is_success)

    def test_extra_irrelevant_arguments_are_ignored(self):
        # 期望只声明 target；实际入参额外携带 request_id / timestamp 等无关字段，
        # 子集语义下应判通过，不应因整个 dict 不精确相等而误判。
        task = self._task(["ast_analyze"])
        result = make_result([
            ("ast_analyze", {"target": "t1", "request_id": "abc-123", "ts": 1700000000}),
        ], answer="ast ok")
        evaluation = evaluate_task(task=task, result=result)
        self.assertTrue(evaluation.has_correct_arguments)
        self.assertTrue(evaluation.is_success)

    def test_missing_answer_keyword_fails_coverage(self):
        task = self._task(["ast_analyze"], answer_contains=("scanners",))
        result = make_result([("ast_analyze", {"target": "t1"})], answer="ast ok")
        evaluation = evaluate_task(task=task, result=result)
        self.assertTrue(evaluation.has_correct_tools)
        self.assertTrue(evaluation.has_correct_arguments)
        self.assertFalse(evaluation.has_expected_answer)
        self.assertFalse(evaluation.is_success)


class CompareResultsTests(unittest.TestCase):
    def test_regression_means_baseline_pass_candidate_fail(self):
        tasks = (
            TrajectoryTask(
                "t1", "security", "x", ("ast_analyze", "run_scanners"),
                {n: {} for n in ("ast_analyze", "run_scanners")}, ("ok",),
            ),
            TrajectoryTask(
                "t2", "reliability", "y", ("ast_analyze", "symbol"),
                {n: {} for n in ("ast_analyze", "symbol")}, ("ok",),
            ),
        )
        baseline = {
            "t1": make_result([("ast_analyze", {}), ("run_scanners", {})]),
            "t2": make_result([("ast_analyze", {}), ("symbol", {})]),
        }
        # t1 候选版本选错工具；t2 保持正确
        candidate = {
            "t1": make_result([("ast_analyze", {}), ("symbol", {})]),
            "t2": make_result([("ast_analyze", {}), ("symbol", {})]),
        }
        report = compare_results(
            baseline_name="baseline", candidate_name="candidate", tasks=tasks,
            baseline_results=baseline, candidate_results=candidate,
            dataset_name="unit",
        )
        self.assertEqual(1.0, report.baseline.success_rate)
        self.assertEqual(0.5, report.candidate.success_rate)
        self.assertEqual(("t1",), tuple(item.task_id for item in report.regressions))
        # 类别切片：security 退化到 0，reliability 保持 1
        slices = {item.category: item for item in report.slices}
        self.assertEqual(0.0, slices["security"].candidate_success_rate)
        self.assertEqual(1, slices["security"].regression_count)
        self.assertEqual(1.0, slices["reliability"].candidate_success_rate)

    def test_missing_result_raises(self):
        tasks = (TrajectoryTask("t1", "x", "y", ("ast_analyze",), {}, ("ok",)),)
        with self.assertRaises(ValueError):
            compare_results(
                baseline_name="b", candidate_name="c", tasks=tasks,
                baseline_results={}, candidate_results={}, dataset_name="unit",
            )

    def test_empty_task_set_raises_instead_of_division_by_zero(self):
        with self.assertRaises(ValueError):
            evaluate_results(
                version_name="v", tasks=(), results_by_task_id={},
            )
        with self.assertRaises(ValueError):
            compare_results(
                baseline_name="b", candidate_name="c", tasks=(),
                baseline_results={}, candidate_results={}, dataset_name="unit",
            )


class FirstErrorAttributionTests(unittest.TestCase):
    TASK = TrajectoryTask(
        "t1", "security", "x", ("a", "b"),
        {"a": {"k": 1}, "b": {"k": 2}}, ("ok",),
    )

    def test_extra_tool(self):
        baseline = make_result([("a", {"k": 1}), ("b", {"k": 2})])
        candidate = make_result([("a", {"k": 1}), ("b", {"k": 2}), ("c", {})])
        attribution = locate_first_error(
            task=self.TASK, baseline_result=baseline, candidate_result=candidate
        )
        self.assertEqual("tool_selection_error", attribution.category)
        self.assertEqual("c", attribution.candidate_span)

    def test_omitted_tool(self):
        baseline = make_result([("a", {"k": 1}), ("b", {"k": 2})])
        candidate = make_result([("a", {"k": 1})])
        attribution = locate_first_error(
            task=self.TASK, baseline_result=baseline, candidate_result=candidate
        )
        self.assertEqual("tool_selection_error", attribution.category)
        self.assertEqual("b", attribution.baseline_span)
        self.assertIsNone(attribution.candidate_span)

    def test_different_tool(self):
        baseline = make_result([("a", {"k": 1}), ("b", {"k": 2})])
        candidate = make_result([("a", {"k": 1}), ("x", {"k": 2})])
        attribution = locate_first_error(
            task=self.TASK, baseline_result=baseline, candidate_result=candidate
        )
        self.assertEqual("tool_selection_error", attribution.category)
        self.assertEqual(("b", "x"), (attribution.baseline_span, attribution.candidate_span))

    def test_argument_error(self):
        baseline = make_result([("a", {"k": 1}), ("b", {"k": 2})])
        candidate = make_result([("a", {"k": 1}), ("b", {"k": 99})])
        attribution = locate_first_error(
            task=self.TASK, baseline_result=baseline, candidate_result=candidate
        )
        self.assertEqual("tool_argument_error", attribution.category)

    def test_execution_error(self):
        baseline = make_result([("a", {"k": 1}), ("b", {"k": 2})])
        candidate = make_result(
            [("a", {"k": 1}), ("b", {"k": 2})], errors={"b": "timeout"}
        )
        attribution = locate_first_error(
            task=self.TASK, baseline_result=baseline, candidate_result=candidate
        )
        self.assertEqual("tool_execution_error", attribution.category)
        self.assertIn("timeout", attribution.reason)

    def test_generation_error_when_trajectories_match(self):
        # 工具轨迹完全一致，但候选答案缺少期望子串
        task = TrajectoryTask("t1", "x", "y", ("a", "b"), {}, ("NEEDLE",))
        baseline = make_result([("a", {}), ("b", {})], answer="NEEDLE ok")
        candidate = make_result([("a", {}), ("b", {})], answer="something else")
        attribution = locate_first_error(
            task=task, baseline_result=baseline, candidate_result=candidate
        )
        self.assertEqual("generation_error", attribution.category)
        self.assertEqual(0.9, attribution.confidence)


class GateTests(unittest.TestCase):
    def _report_pair(self, candidate_results):
        tasks = build_demo_dataset()
        return tasks, compare_results(
            baseline_name="baseline", candidate_name="candidate", tasks=tasks,
            baseline_results=build_demo_results("baseline"),
            candidate_results=candidate_results, dataset_name="unit",
        )

    def test_fixed_version_passes(self):
        tasks, report = self._report_pair(build_demo_results("fixed"))
        gate = evaluate_gate(report=report, config={"thresholds": DEFAULT_THRESHOLDS})
        self.assertEqual("PASS", gate.status)
        self.assertEqual((), gate.violations)

    def test_regression_version_blocks(self):
        tasks, report = self._report_pair(build_demo_results("regression"))
        gate = evaluate_gate(report=report, config={"thresholds": DEFAULT_THRESHOLDS})
        self.assertEqual("BLOCK", gate.status)
        rules = {item.rule for item in gate.violations if item.severity == "block"}
        self.assertIn("minimum_success_rate", rules)
        self.assertIn("maximum_critical_task_regressions", rules)

    def test_warning_only_when_soft_threshold_breached(self):
        # 只配一个宽松的质量下限 + 一个普通回归上限（warning）
        tasks, report = self._report_pair(build_demo_results("regression"))
        gate = evaluate_gate(report=report, config={"thresholds": {
            "minimum_success_rate": 0.5,
            "maximum_task_regressions": 1,
        }})
        self.assertEqual("WARNING", gate.status)
        self.assertTrue(all(item.severity == "warning" for item in gate.violations))

    def test_severity_override_can_block_normal_regressions(self):
        # 默认普通回归只是 warning；通过 severity_overrides 提升为 block 后应 BLOCK
        tasks, report = self._report_pair(build_demo_results("regression"))
        config = {
            "thresholds": {
                "minimum_success_rate": 0.5,
                "maximum_task_regressions": 1,
            },
            "severity_overrides": {"maximum_task_regressions": "block"},
        }
        gate = evaluate_gate(report=report, config=config)
        self.assertEqual("BLOCK", gate.status)
        rule = next(item for item in gate.violations
                    if item.rule == "maximum_task_regressions")
        self.assertEqual("block", rule.severity)

    def test_unknown_threshold_key_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_gate_config({"thresholds": {"minimum_success_rat": 0.9}})

    def test_illegal_severity_override_value_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_gate_config({
                "thresholds": {"maximum_task_regressions": 0},
                "severity_overrides": {"maximum_task_regressions": "BLOCK!"},
            })

    def test_legal_config_passes_validation(self):
        config = validate_gate_config({
            "thresholds": {"minimum_success_rate": 0.9},
            "severity_overrides": {"maximum_task_regressions": "warning"},
        })
        self.assertIn("thresholds", config)


class LedgerAdapterTests(unittest.TestCase):
    def test_ledger_tool_calls_become_tool_spans(self):
        ledger_summary = {
            "model_call_log": [{
                "role": "lead", "provider": "openai", "model": "gpt",
                "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.001,
                "duration_ms": 50, "ok": True, "error": "",
            }],
            "tool_call_log": [
                {"role": "security", "tool": "ast_analyze", "arguments": {"f": "a.py"},
                 "ok": True, "duration_ms": 12, "result_preview": "ok", "error": ""},
                {"role": "security", "tool": "run_scanners", "arguments": {},
                 "ok": False, "duration_ms": 8, "result_preview": "", "error": "boom"},
            ],
        }
        result = ledger_to_spans(ledger_summary, answer="final text", planner_input="diff")
        tool_spans = [span for span in result["spans"] if span["kind"] == "tool"]
        self.assertEqual(["ast_analyze", "run_scanners"], [s["name"] for s in tool_spans])
        # 失败工具映射为 error，成功工具 error 为 None
        self.assertIsNone(tool_spans[0]["error"])
        self.assertEqual("boom", tool_spans[1]["error"])
        # planner / final 边界齐全
        kinds = [span["kind"] for span in result["spans"]]
        self.assertEqual(["planner", "tool", "tool", "final"], kinds)
        self.assertEqual("final text", result["answer"])

    def test_tokens_and_latency_count_each_llm_call_once(self):
        def model(prompt, completion, cost, duration_ms):
            return {
                "role": "lead", "provider": "openai", "model": "gpt",
                "input_tokens": prompt, "output_tokens": completion,
                "cost_usd": cost, "duration_ms": duration_ms,
                "ok": True, "error": "",
            }

        ledger_summary = {
            # 三次模型调用：第一次归 planner，其余归 final，不得重复计算第一次
            "model_call_log": [
                model(100, 20, 0.001, 50),
                model(200, 30, 0.002, 70),
                model(300, 40, 0.003, 90),
            ],
            "tool_call_log": [
                {"role": "security", "tool": "ast_analyze", "arguments": {},
                 "ok": True, "duration_ms": 12, "result_preview": "ok", "error": ""},
            ],
        }
        result = ledger_to_spans(ledger_summary, answer="done")
        metrics = aggregate_run_metrics(result["spans"])
        # token / 成本 = 全部模型调用之和，第一次调用不被算两遍
        self.assertEqual(600, metrics["prompt_tokens"])
        self.assertEqual(90, metrics["completion_tokens"])
        self.assertAlmostEqual(0.006, metrics["cost_usd"])
        # 延迟 = 三次 LLM（50+70+90）+ 一次工具（12）= 222，LLM 耗时被纳入
        self.assertEqual(222.0, metrics["duration_ms"])

    def test_aggregate_run_metrics_tolerates_empty(self):
        self.assertEqual(0, aggregate_run_metrics([])["total_tokens"])


class DemoFixtureTests(unittest.TestCase):
    def test_demo_regression_shape(self):
        tasks = build_demo_dataset()
        self.assertEqual(12, len(tasks))
        critical = sum(task.critical for task in tasks)
        self.assertEqual(3, critical)
        baseline = build_demo_results("baseline")
        regression = build_demo_results("regression")
        fixed = build_demo_results("fixed")
        report_reg = compare_results(
            baseline_name="baseline", candidate_name="regression", tasks=tasks,
            baseline_results=baseline, candidate_results=regression,
            dataset_name="demo",
        )
        report_fixed = compare_results(
            baseline_name="baseline", candidate_name="fixed", tasks=tasks,
            baseline_results=baseline, candidate_results=fixed,
            dataset_name="demo",
        )
        self.assertEqual(1.0, report_reg.baseline.success_rate)
        self.assertEqual(4, len(report_reg.regressions))
        self.assertEqual(2, sum(item.is_critical for item in report_reg.regressions))
        self.assertEqual((), report_fixed.regressions)
        # 归因结果与回归一一对应
        attributions = attribute_results(
            tasks=tasks, report=report_reg,
            baseline_results=baseline, candidate_results=regression,
        )
        self.assertEqual(4, len(attributions))
        self.assertTrue(
            all(isinstance(item, FailureAttribution) for item in attributions)
        )


class ResultPersistenceTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        results = build_demo_results("baseline")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "results.json"
            save_results(path, results)
            loaded = load_results(path)
        self.assertEqual(set(results), set(loaded))
        self.assertEqual(
            results["sec_001"]["spans"][0]["name"],
            loaded["sec_001"]["spans"][0]["name"],
        )


if __name__ == "__main__":
    unittest.main()
