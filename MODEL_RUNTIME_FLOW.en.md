[简体中文](MODEL_RUNTIME_FLOW.md) | **English**

# EVO Runtime and Trajectory Regression Evaluation

The baseline and candidate versions review the same labeled task set. Each review report stores `execution.trajectory`. After collecting reports from both versions, run trajectory regression evaluation and the release gate as a separate step.

```mermaid
flowchart TD
    A["Same review tasks<br/>API diff or GitHub PR"] --> B["Run baseline and candidate versions separately"]

    subgraph REVIEW["Review workflow for each version"]
        B --> C["Create task and save diff"]
        C --> D["Planning: parse diff"]
        D --> E["Executing: rule scans and memory recall"]
        E --> F["Lead delegates assignments"]
        F --> G["Security and Correctness/Reliability review in parallel"]
        G --> H{"High risk?"}
        H -- Yes --> I["Lead assesses results; one worker revision if needed"]
        H -- No --> J["Merge candidate findings"]
        I --> J
        J --> K["Critic challenges findings"]
        K --> L["Lead makes final decision"]
        L --> M["Apply evidence gates and generate report"]
        M --> N["Save report and execution.trajectory"]
    end

    subgraph REGRESSION["Separate trajectory regression evaluation"]
        N --> O["Collect reports from both versions by task_id"]
        O --> P["export: convert reports to standard trajectories"]
        Q["Labeled JSONL task set<br/>Expected tools, arguments, answer, critical flag"] --> R
        P --> R["Evaluate both versions task by task"]
        R --> S["Check tool chain, per-role order, arguments, execution errors, and answer coverage"]
        S --> T["Aggregate success rate, latency, tokens, cost, and category metrics"]
        T --> U["Find regressions: baseline passes, candidate fails"]
        U --> V["Locate first error: tool selection, arguments, execution, or final answer"]
        V --> W["Apply thresholds from release-gate.yaml"]
        W --> X{"Gate result"}
        X --> Y["PASS"]
        X --> Z["WARNING"]
        X --> AA["BLOCK: exit code 1 can stop CI"]
    end
```

A task passes only if its expected tools, asserted arguments, answer text, and execution checks all pass. Parallel workers' tool sequences are aligned separately by role. The trajectory gate does not run automatically after a normal review; it needs reports from both versions and the same labeled JSONL task set.

```powershell
python -m evoagent.trajectory export --reports baseline_reports.json --output baseline_results.json
python -m evoagent.trajectory export --reports candidate_reports.json --output candidate_results.json
python -m evoagent.trajectory gate --tasks tasks.jsonl --baseline baseline_results.json --candidate candidate_results.json --config release-gate.yaml
```
