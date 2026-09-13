# TrajectEvo Release Gate

Status: **BLOCK**
Baseline: baseline
Candidate: candidate

## Gate findings

- [BLOCK] minimum_success_rate: actual=0.6666666666666666, threshold=0.9. Candidate task success rate is below the required minimum.
- [BLOCK] maximum_success_rate_drop: actual=0.33333333333333337, threshold=0.03. Candidate task success rate regressed beyond the allowed drop.
- [BLOCK] minimum_tool_selection_accuracy: actual=0.6666666666666666, threshold=0.95. Candidate tool selection accuracy is below the required minimum.
- [BLOCK] maximum_critical_task_regressions: actual=2, threshold=0. Candidate regressed on more critical tasks than allowed.
- [WARNING] maximum_task_regressions: actual=4, threshold=0. Candidate has more task regressions than the warning threshold.

## First-error diagnostics

- sec_001: tool_selection_error at step 3 (run_scanners -> symbol)
- sec_002: tool_selection_error at step 3 (run_scanners -> symbol)
- sec_003: tool_selection_error at step 3 (run_scanners -> symbol)
- sec_004: tool_selection_error at step 3 (run_scanners -> symbol)
