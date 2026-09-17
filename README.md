English | [**简体中文**](docs/README.zh-CN.md)

# TrajectEvo · Agentic Code Review & Self-Evolution Platform

> TrajectEvo (Trajectory + Evolution) layers **agent trajectory-level regression evaluation, first-error attribution, and a CI release gate** on top of multi-agent code review and prompt self-evolution.

- Reviews unified diffs and emits structured findings, fix suggestions, and test recommendations
- GitHub `pull_request` webhook (`opened`, `reopened`, `synchronize`)
- `agentic` mode: collaborative Lead, Security, Correctness/Reliability, and Critic agents
- SQLite persistence for task state, execution traces, and final reports
- JSON API and Markdown reports
- HMAC-SHA256 webhook signature verification and optional GitHub PR comment write-back
- Web console, task dashboard, and Prometheus metrics
- Fixed single-round multi-agent collaboration for normal tasks (2 Lead calls, 1 per worker, 1 Critic — 5 LLM calls total); high-risk tasks allow at most one worker rework round
- Closed-loop auto-fix via LLM unified patch, AST/CST transforms, sandboxed before/after test comparison, and Draft-PR-only delivery
- PostgreSQL and Redis for production deployments
- Failure-case feedback loop, prompt evaluation, and version activation/rollback
- Self-built agent runtime with persistent checkpoints, execution budgets, and task resume
- Agent Loop with a Tool Registry, parameter Schema validation, and structured Observations
- Tenant/repository-isolated memory backed by confirmed feedback, with retrieval and expiry cleanup
- Redis Streams ACK, worker leases, exponential-backoff retries, and a dead-letter queue
- Idempotent webhook delivery, replay time window, and comment upsert
- User login, RBAC, tenant/repository isolation, and an immutable admin audit log
- Dynamic Skill manifest validation, signature verification, and isolated process sandboxing
- Post-fix compile/test gating, canary rollout, and shadow traffic
- OpenTelemetry traces, Prometheus metrics, and persistent alerts
- **Trajectory-level regression**: runs two versions against the same task set, flags tasks that "passed on baseline but fail on candidate", and slices quality/cost by task category
- **First-error attribution**: aligns tool-call traces step by step to pinpoint the first tool-selection / argument / execution / generation divergence
- **YAML release gate**: emits PASS / WARNING / BLOCK; BLOCK exits with code 1 to stop CI directly (see `release-gate.yaml`)

## Quick Start

The project targets Python 3.11. Install the pinned runtime dependencies and configure a local administrator in the same PowerShell window:

```powershell
python -m pip install -r requirements.txt

$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$env:EVOAGENT_AUTH_REQUIRED = 'true'
$env:EVOAGENT_AUTH_SECRET = [Convert]::ToBase64String($bytes)
$env:EVOAGENT_BOOTSTRAP_ADMIN_USERNAME = 'admin'
$env:EVOAGENT_BOOTSTRAP_ADMIN_PASSWORD = '<replace with a password of at least 10 characters>'

python -m evoagent
```

Do not use the sample placeholders as real passwords or secrets. Environment variables only apply to the current PowerShell window and its child processes; after changing configuration, stop and restart EvoAgent. The service can start and serve health checks without a configured model, but a model must be configured (see [Model Configuration](#model-configuration)) before submitting reviews.

The bootstrap administrator is created only when the username does not already exist; restarting will not overwrite the password of an existing admin user.

The service listens on `127.0.0.1:8080` by default. After startup, open `http://127.0.0.1:8080/`; the frontend shows a login layer once a business API returns unauthorized. Login state is kept in the browser's `localStorage`; to log in again, sign out or clear site data.

API calls require a login and a Bearer token:

```powershell
$session = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/auth/login `
  -ContentType 'application/json' `
  -Body (@{username='admin'; password='<your-password>'} | ConvertTo-Json)
$headers = @{Authorization="Bearer $($session.access_token)"}
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/reviews `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    repository = 'demo/api'
    pull_request = 12
    mode = 'agentic'
    diff = "diff --git a/app.py b/app.py`n--- a/app.py`n+++ b/app.py`n@@ -1 +1,2 @@`n+password = 'secret'`n+eval(user_input)"
  } | ConvertTo-Json)
```

Query a task:

```powershell
Invoke-RestMethod -Headers $headers http://127.0.0.1:8080/v1/tasks/<task-id>
Invoke-WebRequest -Headers $headers http://127.0.0.1:8080/v1/tasks/<task-id>/report
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

## Trajectory-Level Regression & Release Gate

`evoagent/trajectory.py` performs version regression over standardized trajectories (lists of `TrajectorySpan`) that are decoupled from any concrete agent implementation: three per-task assertions (tool chain / arguments / answer), version metric comparison, first-error attribution, and a YAML gate.

An offline demo with no API key required (built-in code-review fixtures: baseline fully passes, regression is BLOCKed, fixed fully passes):

```powershell
python -m evoagent.trajectory demo
```

For real usage, persist the trajectory results produced by each of the two versions as JSON (keyed by task id), then run the offline comparison and gate:

```powershell
# Compare two versions and emit a Markdown regression report (--format json for structured output)
python -m evoagent.trajectory compare --baseline trajectory_examples/baseline.json --candidate trajectory_examples/regression.json

# Evaluate the release gate against release-gate.yaml thresholds; BLOCK exits with code 1 and can be wired into CI directly
python -m evoagent.trajectory gate --baseline trajectory_examples/baseline.json --candidate trajectory_examples/regression.json --config release-gate.yaml
```

The trajectories of the real multi-agent system are adapted via `ledger_to_spans(ExecutionLedger.summary())`: every tool call's arguments, success/failure, latency, and error are reconstructed into a standard tool span, so the gate evaluates the **real agent rather than a rule-based simulator**. Sample result files are in `trajectory_examples/`.

## Model Configuration

Official DeepSeek API (billed per token):

```powershell
$env:EVOAGENT_LLM_PROVIDER = 'deepseek'
$env:EVOAGENT_DEEPSEEK_API_KEY = '<deepseek-api-key>'
python -m evoagent
```

Use the rate-limited, availability-varying free DeepSeek model through OpenRouter:

```powershell
$env:EVOAGENT_LLM_PROVIDER = 'openrouter-deepseek-free'
$env:EVOAGENT_OPENROUTER_API_KEY = '<openrouter-api-key>'
python -m evoagent
```

If the designated free DeepSeek model is retired, either change `EVOAGENT_LLM_MODEL` to another current `:free` model on OpenRouter, or set the provider to `openrouter-free` to let free routing pick an available model automatically.

For any other OpenAI Chat Completions-compatible endpoint, use `custom`:

```powershell
$env:EVOAGENT_LLM_PROVIDER = 'custom'
$env:EVOAGENT_LLM_BASE_URL = 'https://example.com/v1'
$env:EVOAGENT_LLM_API_KEY = '<token>'
$env:EVOAGENT_LLM_MODEL = '<model-name>'
```

Secrets are read only from environment variables; never commit them to the repository.

On startup the project auto-loads `.env` from the project root and also supports `evoagent/.env`; system environment variables take precedence over `.env`. A recommended root `.env` (already ignored by `.gitignore`):

```env
EVOAGENT_LLM_PROVIDER=deepseek
EVOAGENT_DEEPSEEK_API_KEY=your-real-api-key
```

## GitHub Webhook

The project receives PR events through "GitHub repository webhook + public tunnel + fine-grained PAT"; creating or installing a GitHub App is not required:

```text
GitHub pull_request event
        │
        ▼
https://<public-domain>/webhooks/github
        │  public tunnel
        ▼
http://127.0.0.1:8080/webhooks/github
        │
        ▼
EvoAgent creates an asynchronous review task
```

### 1. Configure EvoAgent

Generate a webhook secret and, if needed, configure a GitHub fine-grained personal access token:

```powershell
$webhookBytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($webhookBytes)
$env:EVOAGENT_GITHUB_WEBHOOK_SECRET = [Convert]::ToBase64String($webhookBytes)

# Required for private repos, PR comment write-back, or auto-fix; optional when reviewing public repos without write-back.
$env:EVOAGENT_GITHUB_TOKEN = '<GitHub fine-grained PAT>'

# Disabled by default. When true, update or create a PR comment after the review completes.
$env:EVOAGENT_AUTO_POST_REVIEW = 'true'

python -m evoagent
```

The webhook secret verifies the HMAC-SHA256 signature in the GitHub request header and must not be reused as the login `EVOAGENT_AUTH_SECRET`. Webhook requests carry no console Bearer token; `/webhooks/github` authenticates via signature rather than user login.

Grant the fine-grained PAT access only to the repositories you integrate with, with least-privilege permissions per feature:

- Read private-repo PR diffs: `Contents: Read`, `Pull requests: Read`;
- Write review comments: `Pull requests: Read and write`;
- Create auto-fix branches and commits: `Contents: Read and write`, `Pull requests: Read and write`.

When you only receive webhooks for public repos without comment write-back or auto-fix, no PAT is required. Secrets must be set before starting EvoAgent, and the service must be restarted after changing them.

### 2. Establish a Public Tunnel

GitHub cannot reach `127.0.0.1`, so forward a public HTTPS URL to local `http://127.0.0.1:8080`. Use any installed tunnel tool, for example:

```powershell
# Cloudflare Quick Tunnel
cloudflared tunnel --url http://127.0.0.1:8080

# or ngrok
ngrok http 8080
```

The command prints a public HTTPS URL such as `https://example.trycloudflare.com` or `https://example.ngrok-free.app`. Keep both EvoAgent and the tunnel process running. Temporary public URLs usually change when the tunnel restarts; when that happens, update the GitHub webhook Payload URL accordingly.

These quick tunnels expose both the console and the API on port 8080 to the public internet, so you must keep `EVOAGENT_AUTH_REQUIRED=true` with a strong admin password and a random `EVOAGENT_AUTH_SECRET`. For long-term deployment, use a reverse proxy to expose only `/webhooks/github` (and optionally `/health`) rather than exposing the entire console.

### 3. Add the Webhook in the GitHub Repository

In the target repository, go to **Settings → Webhooks → Add webhook** and fill in:

- **Payload URL**: `https://<public-domain>/webhooks/github`;
- **Content type**: `application/json`;
- **Secret**: exactly the same value as `EVOAGENT_GITHUB_WEBHOOK_SECRET`;
- **SSL verification**: keep enabled;
- **Which events would you like to trigger this webhook?**: choose **Let me select individual events** and check only **Pull requests**;
- **Active**: keep checked.

EvoAgent handles the `opened`, `reopened`, and `synchronize` PR actions; other `pull_request` actions are accepted but ignored. The service downloads the diff from the payload's `diff_url` and creates an asynchronous review task.

### 4. Verify the Connection

First confirm both the local service and the public URL pass health checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod https://<public-domain>/health
```

Then open a new PR, reopen a PR, or push a commit to a PR. In GitHub **Settings → Webhooks → Recent Deliveries** you should see `/webhooks/github` return `202`; the corresponding review task then appears in the console's task center. On failure, first check that the tunnel is still running, that the Payload URL contains `/webhooks/github`, that the secret matches, and that the PAT has permission on the target repository.

Results are stored only in the console by default. PR comments are written back only when `EVOAGENT_AUTO_POST_REVIEW=true`.

Auto-fix only covers rules that are deterministically safe, such as debug output, `shell=True`, and hardcoded Python credentials; results are always committed to a new `evoagent/fix-pr-*` branch and never modify the source branch directly.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/v1/auth/login` | Log in and obtain a short-lived tenant-bound Bearer token |
| `POST` | `/v1/reviews` | Create a synchronous review task |
| `POST` | `/v1/reviews?async=true` | Create an asynchronous review task |
| `GET` | `/v1/tasks/{id}` | Get status, trace, and report |
| `GET` | `/v1/tasks/{id}/report` | Get the Markdown report |
| `GET` | `/v1/tasks/{id}/feedback` | Get feedback history for a completed task |
| `POST` | `/v1/tasks/{id}/fix` | Create an auto-fix branch and commit |
| `POST` | `/v1/tasks/{id}/feedback` | Feed back false positives, missed findings, or bad fixes |
| `POST` | `/v1/tasks/{id}/cancel` | Request task cancellation |
| `POST` | `/v1/tasks/{id}/resume` | Resume a task from its latest checkpoint |
| `POST` | `/webhooks/github` | Receive GitHub PR webhooks |
| `POST` | `/v1/skills/reload` | Dynamically reload Skills |
| `POST` | `/v1/evolution/auto` | Generate and evaluate a prompt version from failure cases |
| `POST` | `/v1/evolution/propose` | Evaluate a specified prompt candidate |
| `GET/POST` | `/v1/evaluation/cases` | Query or add versioned evaluation samples |
| `GET` | `/v1/evolution/status` | Query model and evaluation-gate readiness |
| `GET` | `/v1/evolution/runs` | Query persisted old-vs-new version evaluation runs |
| `POST` | `/v1/skills/{name}/versions/{version}/activate` | Activate or roll back a version |
| `POST` | `/v1/skill-evolution/auto` | Generate, replay, and gate a Skill candidate from confirmed feedback |
| `POST` | `/v1/skill-evolution/propose` | Evaluate a given agent Skill `SKILL.md` artifact |
| `GET` | `/v1/skill-evolution/status?skill_name={name}` | Query Skill gate and active version |
| `GET` | `/v1/skill-evolution/runs` | Query Skill evolution runs and metrics |
| `GET` | `/v1/skill-evolution/{name}/versions` | Query the Skill artifact version chain |
| `POST` | `/v1/skill-evolution/{name}/versions/{version}/activate` | Activate or roll back a Skill artifact |
| `GET` | `/metrics` | Prometheus text metrics |
| `GET` | `/api/alerts` | Query tenant alerts |
| `GET` | `/api/audit` | Query tenant audit logs |
| `GET` | `/api/queue/dead-letters` | Query dead-letter tasks |
| `POST` | `/v1/queue/dead-letters/replay` | Replay a dead-letter task |
| `GET/POST` | `/api/deployments/llm-review`, `/v1/deployments/llm-review` | Query or configure canary/shadow rollout |

The `diff` in `POST /v1/reviews` defaults to a 1 MiB maximum; a single task defaults to at most 8 steps and 120 seconds. These can be tuned via environment variables; see `.env.example`.

## Architecture

```text
HTTP / GitHub Webhook
        │
        ▼
 ReviewService ── TaskStore (SQLite / PostgreSQL)
        │
        ▼
        ReviewHarness (EvoAgent Runtime / checkpoint / resume / budget / trace)
        │
        ├── DiffParser
        ├── Redis Streams / ACK / lease / retry / DLQ
        └── Agentic Lead/Workers
              ├── Lead: assigns the task once and judges risk; requests at most one rework round for high-risk tasks; final synthesis
              ├── Security Worker: inputs / permissions / sensitive data / dangerous call chains
              ├── Correctness/Reliability Worker: state / exceptions / concurrency / resources / compatibility
              ├── Critic Worker: one-shot blind review, counterexamples, and evidence challenges
              └── Gates: format, evidence, confidence, and release gates
```

## Skills: Purpose and Authoring

The directories under `skills/` are agent Skill packages, not tool-connector directories. They are currently centered on the review domain because the product goal of this project is code review. A Skill's job is to inject a domain's judgment criteria, investigation steps, and output constraints into the model — not to execute SQL, call GitHub, or open network connections itself.

A Skill package contains at least one `SKILL.md`:

```text
skills/<skill-name>/
└── SKILL.md                 # YAML metadata + full operating instructions
```

The runtime reads the `SKILL.md` frontmatter:

- `name`: must match the directory name and use only lowercase letters, digits, and hyphens;
- `description`: used only for directory discovery and routing; it should state concisely when to enable the Skill;
- `allowed-tools`: the set of tool names the Skill permits a worker to use. It is a permission declaration and does not install or connect any tool; the effective permission is also intersected with the role's permissions.

The Markdown body after the frontmatter is the core of the Skill. It is injected into the relevant worker's context after the Lead selects the Skill, so it should be written as an executable review protocol rather than a few tag-style prompts. It is recommended to include at least: task boundaries, in/out-of-scope conditions, a step-by-step investigation flow, an edge-case checklist, evidence and severity criteria, false-positive exclusion rules, remediation and verification requirements, and stable output fields.

A Skill may attach text resources (such as spec excerpts, query templates, or checklists). The runtime includes resources in the Skill package and registers `read_skill_resource` only for selected Skills; resources cannot escape the package directory via path traversal or symlinks. Real tool connections are provided by the runtime's `RepositoryToolSuite`, external integrations, or plugins — a Skill only declares "which already-existing tools may be used".

The runtime flow is:

```text
Scan SKILL.md files on disk
        ↓ expose only name/description to the Lead
Lead selects the relevant Skill
        ↓ inject instructions and narrow allowed-tools
Worker gathers evidence with repository tools
        ↓
Finding gate validates format, evidence, confidence, and release conditions
```

Hardcoding Skill content into the global context is therefore not equivalent: doing so would make every task carry the full set of domain rules, increasing context length and cross-rule interference. The Skill mechanism enables per-task selection, tool-permission narrowing, hot-reloadable versions, and candidate evaluation/gating/rollback via `/v1/skill-evolution/*`. Rules that are only a sentence or two and have no independent selection value are better placed in the global prompt; only a group of rules with a clear boundary, scenario-based activation, or a need to evolve independently should become a Skill.

The built-in Skills in this repository already follow this structure with complete Mission, Review procedure, Checklist, Evidence/severity, Remediation/verification, and Output contract sections. To add a new Skill, copy one of the existing directories, replace those sections for the new domain, and call `POST /v1/skills/reload` so new tasks load the latest version.
