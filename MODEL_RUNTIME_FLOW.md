[**简体中文**](MODEL_RUNTIME_FLOW.md) | [English](MODEL_RUNTIME_FLOW.en.md)

# EVO 运行逻辑与轨迹回归评估

同一批标注任务分别由基线版本和候选版本执行。每次审查生成的报告保存 `execution.trajectory`；收集两个版本的报告后，再运行独立的轨迹回归评估和发布门禁。

```mermaid
flowchart TD
    A["同一批审查任务<br/>API diff 或 GitHub PR"] --> B["分别运行基线版本与候选版本"]

    subgraph REVIEW["每个版本的审查流程"]
        B --> C["创建任务并保存 diff"]
        C --> D["Planning：解析 diff"]
        D --> E["Executing：规则扫描、记忆召回"]
        E --> F["Lead 分派任务"]
        F --> G["Security 与 Correctness/Reliability 并行审查"]
        G --> H{"高风险？"}
        H -- 是 --> I["Lead 评估；必要时返工一次"]
        H -- 否 --> J["合并候选问题"]
        I --> J
        J --> K["Critic 质疑"]
        K --> L["Lead 最终决策"]
        L --> M["证据门控、生成报告"]
        M --> N["保存报告及 execution.trajectory"]
    end

    subgraph REGRESSION["独立的轨迹回归评估"]
        N --> O["按 task_id 收集两个版本的报告"]
        O --> P["export：转换为标准轨迹"]
        Q["JSONL 标注任务集<br/>预期工具、参数、答案、关键任务标记"] --> R
        P --> R["逐任务评估基线与候选版本"]
        R --> S["检查工具链、角色内顺序、参数、执行错误和答案覆盖"]
        S --> T["汇总成功率、延迟、Token、成本及任务类别指标"]
        T --> U["识别回归：基线成功且候选失败"]
        U --> V["首错归因：工具选择、参数、执行或最终生成"]
        V --> W["按 release-gate.yaml 检查阈值"]
        W --> X{"门禁结果"}
        X --> Y["PASS"]
        X --> Z["WARNING"]
        X --> AA["BLOCK：退出码 1，可阻断 CI"]
    end
```

单任务必须同时满足预期工具、预期参数、答案文本及无执行错误才算成功。并行角色的工具轨迹按角色分别对齐。普通审查完成后不会自动运行轨迹门禁；需要收集两个版本的报告，并提供同一份 JSONL 标注任务集。

```powershell
python -m evoagent.trajectory export --reports baseline_reports.json --output baseline_results.json
python -m evoagent.trajectory export --reports candidate_reports.json --output candidate_results.json
python -m evoagent.trajectory gate --tasks tasks.jsonl --baseline baseline_results.json --candidate candidate_results.json --config release-gate.yaml
```
