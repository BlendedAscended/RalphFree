# RalphFree

## Stack
- TBD — inspect project files to determine stack

## Key Paths
- Inspect root directory for entry points

## Notes
- Project details to be documented on first deep dive

## Obsidian Vault

Cross-tool knowledge base: `~/Documents/Obsidian Vault`
- Project status: [[Projects/RalphFree]]
- Daily notes: `Daily/YYYY-MM-DD.md`
- Export LLM outputs: `Outputs/LLM/`


# Cross-Tool Rules (Shared by ALL AI tools)

## 1. Changelog Rule

After any code change, feature, bug fix, or architecture update, you MUST:
1. Ensure `CHANGES.md` exists in the project root
2. Add a new entry at the very top using this format:

```markdown
## YYYY-MM-DD HH:MM (UTC) — one-line summary

**Changes:**
- What changed
- Key files modified / created / deleted
- Breaking changes, migration notes, client impact
```

Applies to ALL tools (OpenCode, Claude Code, Hermes, Codex, Copilot, Gemini). No exceptions.

**MANDATORY: Read-then-prepend protocol.** Before writing `CHANGES.md`:
1. ALWAYS read the entire file first to get existing content.
2. Prepend your new entry at the very top, above all older entries.
3. Use `write_file` with the full combined content (new entry + all existing content).
4. NEVER use `write_file` with only your new entry — this destroys history.
5. The file grows indefinitely. Do not truncate, summarize, or prune it.
6. If the file exceeds 200 lines, still prepend — do not compact. A separate compact summary may be generated for agent context, but the full `CHANGES.md` is append-only.

This prevents the observed failure where OpenCode (kimi-k2.6) replaced the entire file with a single entry, destroying all prior history.

---

<!-- universal -->

## 2. Agent Coordination (Concurrent Editing)

San runs multiple AI tools simultaneously. Before editing ANY file:
1. Check `{project_root}/WORKING.lock`. If it exists and is <30 min old, ask San before proceeding.
2. Read `{project_root}/AGENT_REGISTRY.md` to see what other agents are doing.
3. While working: write your own `WORKING.lock` with target files and expiry.
4. When done: delete `WORKING.lock`. Mark `AGENT_REGISTRY.md` entry as `completed`.

## 3. Formatting Hard Stops

- No em dashes. Use colon, period, or comma.
- No filler adverbs: "seamlessly", "effortlessly", "robustly", "leveraging", "utilizing".
- No preamble. Lead with the deliverable.
- Active voice only. No "it should be noted that."
- Code: `snake_case` (Python), `camelCase` (JS/TS), `PascalCase` (React components).

## 4. Verification (Auto-Run After Edits)

| Trigger | Action |
|---|---|
| Edit `.php` | Run `php -l` on file |
| Edit `.jsx`/`.js` | Check bracket balance |
| Edit JS/JSX in Vite project | Run `npm run build` before done |
| About to `git commit` | Verify no secrets staged |

## 5. Skill Activation

- Load `sandeep-universal-style` at conversation start.
- Load named skills only when relevant to the task.
- Do not bulk load every skill.

## 6. Reference Files

- Full rules: `~/.claude/CLAUDE.md` (150 lines, memory/scope/safety)
- OpenCode rules: `~/.config/opencode/AGENTS.md` (135 lines)
- Cross-tool rules: `~/.claude/context/cross-tool-rules.md`
- Vault (single source of truth): `~/Documents/Obsidian Vault/CLAUDE.md` (306 lines)


---

## Who I Am

**Name:** San Ghotra  
**Company:** Veldon Lab (founder)  
**Role:** Full-stack builder — AI agents, browser extensions, healthcare tech, community platforms  
**Focus:** Building AI-augmented tools that compound in value over time  
**Style:** Direct, no fluff, terminal-first, ship fast, automate ruthlessly  

**Goals 2026:**
1. Ship Galaxy multi-agent platform as a product
2. Launch PriorZap denial-prevention in healthcare (real revenue)
3. Complete all active client projects (MLLC, Sanctuary, Aqwa)

---

---

## Current Projects (16 active)

### High Priority
| Project | What | Status |
|---------|------|--------|
| [[Samurai]] | Browser extension + Discord bot — vision analysis, MCP server | Building |
| [[Galaxy]] | Multi-agent AI platform — OpenClaw, n8n, Prometheus | Building |
| [[PriorZap]] | Real-time healthcare denial prevention at point of care | Building |
| [[Veldon Lab-2.0]] | Company website — Next.js 15, Sanity, Spline 3D | Building |
| [[MLLC]] | Nonprofit foundation — WordPress + Preact admin portal | Phase 5 |

### Medium Priority
[[RT]], [[Refree]], [[Sanctuary]], [[Aqwa]], [[Galaxy-Products]], [[Jack]], [[Gibby]], [[Zenith]], [[TrustOneServices]]

### Low Priority
[[Galaxy-Personal]], [[RalphFree]]

**Currently stuck on:** Multi-agent concurrency control across tools.  
**Next milestone:** Galaxy agent lifecycle completion; Samurai extension MVP.

---

---

## How This Vault Works

```
Vault Root
├── CLAUDE.md          ← THIS FILE — read first every session
├── Dashboard.md       ← Navigation hub
├── Inbox/             ← Raw captures (articles, tweets, voice notes, quick saves)
├── Notes/             ← Processed content (articles, highlights, research)
├── Ideas/             ← My own thinking, observations, voice note transcripts
├── Projects/          ← Active work — one note per project
├── Knowledge/         ← Evergreen concepts, people, tools, processes
├── Resources/         ← External references (articles, videos, books, papers)
├── Daily/             ← Daily notes (YYYY-MM-DD.md)
├── Queue/             ← Task files for agent processing
├── Generated/         ← Agent outputs, operations log
├── Outputs/           ← AI-generated content (LLM exports, charts, diagrams)
├── Attachments/       ← Images, PDFs, media files
├── System/            ← Schemas, templates reference
├── Templates/         ← Note templates
└── Archive/           ← Completed/deprecated, Notion export raw
```

---

---

## What I Want From AI Tools

1. **Surface connections I haven't seen.** Find links between projects, ideas, and notes.
2. **Challenge my assumptions.** Don't agree with me — find the gap in my thinking.
3. **When I ask what to focus on** — answer from vault context, not generically.
4. **Flag contradictions.** If something I saved earlier contradicts something new, call it out.
5. **Be concise.** No preamble. No fluff. Direct answers. Tables over paragraphs.
6. **Save durable insights.** Update this vault. Compound the knowledge.

---

---

## Behavioral Contract — 12 Rules

> Each rule prevents a specific observed failure mode. Source: Forrest Chang's 6-week, 30-codebase study built on Karpathy's January 2026 thread. Compliance holds ~76% for 12 rules; past 14 rules drops to 52%.
> **Any rule that hasn't prevented a real failure in 4 weeks gets dropped.** This is a living contract, not a museum.

### Before Writing Code
| # | Rule | Prevents |
|---|------|----------|
| R1 | **State assumptions explicitly.** If uncertain, ask. Stop when confused. Name what's unclear. | Silent wrong assumptions |
| R2 | **Read before writing.** Before adding code, read exports, callers, shared utilities. | Code conflicts 30 lines away |
| R3 | **Minimum code that solves the problem.** Nothing speculative. No abstractions for single-use code. Would a senior engineer call it overcomplicated? Simplify. | Over-engineering |

### While Writing Code
| # | Rule | Prevents |
|---|------|----------|
| R4 | **Touch only what you must.** Don't refactor adjacent code. Match existing style. Clean up only your own mess. | Orthogonal damage |
| R5 | **Model only for judgment calls.** Use for: classification, drafting, summarization. NOT for: routing, retries, deterministic transforms. If code can answer, code answers. | Flaky AI-driven business logic |
| R6 | **Match conventions, even if you disagree.** snake_case → snake_case. class components → class components. If you think the convention is harmful, surface it — don't fork silently. | Convention drift across 16 projects |
| R7 | **Surface conflicts, don't average them.** If two patterns contradict, pick one (more recent/tested). Explain why. Flag the other for cleanup. | Incoherent blended code |

### Verifying Work
| # | Rule | Prevents |
|---|------|----------|
| R8 | **Define success, then iterate.** Don't follow steps. Define what done looks like and loop until verified. | Aimless iteration |
| R9 | **Tests verify intent, not just behavior.** A test that can't fail when business logic changes is wrong. | Shallow tests passing while production breaks |
| R10 | **Fail loud.** "Completed" is wrong if anything was skipped silently. "Tests pass" is wrong if any were skipped. Default to surfacing uncertainty, not hiding it. | Silent failures compounding across 7 tools |

### Session Management
| # | Rule | Prevents |
|---|------|----------|
| R11 | **Hard token budgets.** Per-task: 4,000 tokens. Per-session: 30,000 tokens. Summarize and start fresh when approaching the limit. Surface the breach. | Agent spirals, 90-minute loops |
| R12 | **Checkpoint every significant step.** Summarize what was done, what's verified, what's left. Don't continue from a state you can't describe back. If you lose track, stop and restate. | Multi-step refactors losing state |
| R13 | **Drain the backlog before closing.** Before ending any session: scan for features, ideas, or changes discussed but not implemented. Write each one to `BACKLOG.md` in the project root (create if absent). Format: `- [ ] [YYYY-MM-DD] [tool] Item — one-line context`. Never let a discussed idea die in session context. | Ideas discussed in one session, never surfaced again |

### Exceptions
- **Prototypes / spikes:** R3 and R9 are relaxed — speculative code and shallow tests are acceptable.
- **Single-file scripts:** R2, R4, and R10 are relaxed.
- **Trivial tasks:** R1 and R12 are relaxed.

### Rule Expiration — 4-Week Auto-Prune
After each Weekly Synthesis (Monday 8am), Hermes reviews the 12 rules against the vault's work log:
- **If a rule prevented zero real failures in 4 weeks** → flag for removal.
- **If a new failure mode appears** → propose a new rule.
- Target: stay at 10-14 rules. Add one → drop one.
- **Failure tracking:** `Knowledge/rules-fired.md` — `YYYY-MM-DD | R# | what the rule prevented`.

---

<!-- universal -->

## Formatting Hard Stops

These apply to ALL output across ALL tools. No exceptions.

- No em dashes. Restructure the sentence.
- No hyphens in compound modifiers (write "real time" not "real-time").
- No filler adverbs: "really", "truly", "incredibly", "genuinely".
- No bullet lists in final deliverables (emails, letters, pitch scripts) unless explicitly requested.
- No preamble before a deliverable. Output first.
- No post-summary after a deliverable. End at the last line.
- One clarifying question max per response.
- Active voice in all operational and instructional writing.
- End every response with a concrete next step or ask. Never end with "let me know if you need anything."

---

---

## Safety Rules — Always Apply

- Never fabricate facts, IDs, dates, credentials, metrics, or environment behavior.
- Preserve schema integrity and relational integrity.
- Prefer native APIs over UI automation.
- Keep outputs implementation-oriented. No filler.
- Apply only rules relevant to the task. No scope bleed.
- Never commit secrets (`.env`, `credentials.json`, API keys).
- Never auto-resolve merge conflict markers. Stop and ask.
- Never paste secret literals (API keys, tokens, passwords) into shell command text. Pull them from OpenBao or env vars at runtime. Inline secrets trip content scanners (Hermes Tirith) and force approval prompts that cannot be allowlisted.

### Deletion / Cleanup Permission Rules

Interpret "do not delete without permission" as protecting durable user/project data, not temporary caches or rebuildable artifacts.

Never delete user-created source files, project files, configs, credentials, databases, uploads, or production data without explicit permission.

Safe cleanup actions do NOT require permission:
- deleting build/cache folders: `.next/cache`, `.turbo`, `.vite`, `node_modules/.cache`, `.cache`
- clearing framework cache: Next.js cache, Vercel build cache, npm/pnpm/yarn cache
- restarting local dev servers, Docker containers, or non-production services
- pruning stopped Docker containers/images only when they are clearly temporary and not named production data volumes

Always ask permission before:
- deleting files outside known cache/build folders
- deleting Docker volumes
- deleting databases, buckets, uploads, logs needed for audit, or backups
- deleting `.env`, secrets, config files, migrations, package files, lockfiles
- running `rm -rf` on broad paths like project root, home directory, `/var`, `/etc`, `/opt`, or VPS folders
- changing production infrastructure, DNS, Vercel project settings, or live database state

When unsure, ask. But do not ask for routine cache cleanup or service restart unless it affects production.

---

---

## Agent Write Rules

- Write to vault `Inbox/` only unless explicitly told to write elsewhere.
- When writing to `Knowledge/`, ensure the note is evergreen (not time-bound).
- When updating `Projects/`, add a timestamp line: `Last updated: YYYY-MM-DD`.
- Never overwrite vault content without reading it first.
- When exporting to `Outputs/`, include source context (which tool, which session, which prompt).

---

---

## Concurrent Agent Protocol (ALWAYS ACTIVE — AUTOMATIC)

San runs multiple AI tools simultaneously: OpenCode, Claude Code, Codex, Hermes, VS Code Copilot, Gemini CLI, OpenClaw. Every session MUST follow the lock protocol before editing any file.

**Automatic behavior:** If the working directory contains a `.git` folder or `AGENTS.md`, this is a project repo. You MUST create `WORKING.lock` before making ANY edit. No exceptions.

**Before any edit:**
1. Check `{project_root}/WORKING.lock`. If it exists and is <30 min old, another agent is active. Ask San before proceeding.
2. Read `{project_root}/AGENT_REGISTRY.md` to see what other agents are doing.

**While working:**
- Write your own `WORKING.lock` with your target files and expiry time.
- Append your task to `AGENT_REGISTRY.md` with status `in_progress`.

**When done:**
- Delete `WORKING.lock`.
- Mark your `AGENT_REGISTRY.md` entry as `completed`.

**Exclusive locks (no shared editing):** `functions.php`, `app.js`, `package.json`, `.htaccess`
**Soft locks (read OK, write waits):** `AdminPortal.jsx`, any file >1000 lines.

### Git Hook Enforcement (Cross-Tool Protection)

Hooks to implement as git hooks:
- `pre-commit`: Check for `WORKING.lock` before commits. Check for secrets (`.env`, `credentials.json`, API keys). Run lint/typecheck if configured.
- `pre-push`: Verify no secrets in staged files. Verify no merge conflict markers.
- `commit-msg`: Enforce conventional commit format if project uses it.

Each active repo gets hooks via `setup.sh hooks`. Hook scripts in `ai-config-transfer/hooks/`, symlinked into `.git/hooks/`.

---

---

## Cross-Tool Architecture

| Tool | Lane | Model |
|---|---|---|
| OpenCode | Daily coding | Mimo V2.5 Pro (Go), MiniMax M3 fallback |
| OpenClaw | Orchestration, scheduling, HITL gate | pm-san: Qwen 3.6 Plus (2h), design-lead: Qwen 3.6 Plus (4h), daemon: GLM 5.1 (6h) |
| Claude Code | Hard refactors, architecture, complex debugging | Opus 4.6 (escalation only) |
| Hermes | Knowledge compounding, cron automation | Mimo V2.5 Pro default, GLM 5.1 cached reads |
| Codex | Extended coding sessions, sandboxing | OpenAI models (subscription) |
| VS Code Copilot | Inline assist | GitHub Copilot |
| Gemini CLI | Supplementary, large context. MacDaddy only; MacAttack runs Antigravity IDE | Gemini models |

Cross-tool config:
| Tool | Lane | Model |
|---|---|---|
| OpenCode | Daily coding | Mimo V2.5 Pro (Go), MiniMax M3 fallback |
| Claude Code | Hard refactors, escalation | Opus 4.6 (escalation only) |
| Codex | Extended sessions, sandboxing | OpenAI models (subscription) |
| VS Code Copilot | Inline assist | GitHub Copilot |
| Gemini CLI | Supplementary, large context. MacDaddy only; MacAttack runs Antigravity IDE | Gemini models |
| Hermes | Knowledge compounding, cron | Mimo V2.5 Pro default, GLM 5.1 cached |
| OpenClaw | Orchestration, HITL gate | Qwen 3.6 Plus, GLM 5.1 |

Escalation: OpenCode Go → OpenCode Zen (pay-per-use) → Claude Code (Opus 4.6) → Codex.

### Runtime Ownership (decided 2026-07-03)

Three tiers. Full rationale in `~/.claude/memory/decisions.md`.

| Tier | Rule | Examples |
|---|---|---|
| T1: AI CLIs | Vendor native installers only. Never `npm i -g`. | claude (native), opencode, hermes, codex (brew cask), mempalace (uv) |
| T2: npm utility globals | One tree only: Hermes node (`~/.local/bin/node` and `npm` symlinks stay). | agnix, rulesync, pnpm, higgsfield |
| T3: project runtimes | Homebrew node or per project pins. nvm retired (MacDaddy migration pending). | vflow2.0, samurai builds |

### Config Sync (Critical)

All 7 tools share configs via symlinks. Single source of truth paths:
- **Instructions:** `~/Documents/Obsidian Vault/CLAUDE.md` (this file) propagated to all tools
- **Skills:** `~/.config/opencode/skills/` → symlinked to `~/.claude/skills/`, `~/.hermes/skills/*`, `~/.codex/skills/*`
- **MCP servers:** Must be added to ALL 3 configs: `opencode.json`, `~/.claude/settings.json`, `~/.hermes/config.yaml`
- **Git hooks:** All repos symlink `.git/hooks/` → `~/Projects/Project26/ai-config-transfer/hooks/git/`
- **Cross-tool rules:** `~/.claude/context/cross-tool-rules.md` — changelog, WORKING.lock, formatting, verification, git safety

When updating this file, the sync script (`~/.hermes/scripts/sync-rules.py`) propagates changes to all derived files. If you edit a derived file directly, the script merges NEW content back into this file.

### MCP Servers

All AI tools share the same MCP server set. If you add a new MCP server, add it to ALL tool configs:
- **OpenCode**: `~/.config/opencode/opencode.json` → `mcp` key (command as array)
- **Claude Code**: `~/.claude/settings.json` → `mcpServers` key (command + args)
- **Hermes**: `~/.hermes/config.yaml` → `mcp_servers` key (command + args in YAML)
- **Codex**: No MCP support (uses project-level AGENTS.md symlinked to `~/.claude/CLAUDE.md`)

Current shared MCP servers:
| Server | Command | Purpose |
|--------|---------|---------|
| mempalace | `mempalace-mcp` | Persistent memory palace (knowledge graph) |
| fetch | `npx -y @modelcontextprotocol/server-fetch` | HTTP fetch/web scraping |
| filesystem | `npx -y @modelcontextprotocol/server-filesystem` | Filesystem access (Agents + Obsidian) |
| git | `npx -y @modelcontextprotocol/server-git` | Git operations |
| time | `npx -y @modelcontextprotocol/server-time` | Time/timezone utilities |

### AI Provider Aliases

Provider keys load through shell aliases named `deepseek`, `qwen`, and `openai` defined in `~/.bashrc`. The canonical secret store is OpenBao on MacDaddy (`http://100.82.2.87:8200`, paths under `secret/data/shared/providers/*`; schema in ai-config-transfer `openbao/SECRET-ROUTING.md`, helpers `secret-run` and `openbao-projects`). This file never carries key material.

### oMLX Serving

Local inference at `127.0.0.1:8087`, API key: `mlx`. Max 20GB RAM per model.
One large model at a time (24GB total). Loaded: qwen2.5-14b (default), qwen3-30b (for heavy analysis, needs `/no_think` in system prompt, 24GB). Fallback: deepseek-chat, kimi-k2.5.
Commands: `ollama list` (loaded), `mlx_lm.server --model ... --port 8087`.

---

---

## Voice and Style

Load and apply `sandeep-universal-style` on every response without exception. It governs formatting, banned patterns, tone, and output discipline across all task types.

**Self-healing rule.** When San corrects a phrase, word choice, or formatting pattern:
1. Apply the correction immediately in the current response.
2. Identify which rule in `sandeep-universal-style/SKILL.md` it maps to (or is missing from).
3. Propose a concrete line-level update to the skill file.
4. Log the correction to `~/.claude/memory/style-corrections.md` with date and description.

Never silently absorb a style correction. Every correction is a rule candidate.

---

---

## Scope Routing

Apply only rules relevant to the active task. No scope bleed.

| Task type | Load |
|---|---|
| Engineering (code, schema, infra) | `docs/ai/CODE-STANDARDS.md` + engineering section of `sandeep-universal-style` |
| Design and frontend | `veldonlab-design-system`, `frontend-blueprint`, `design-director` |
| Career and pitches | Domain skill (`aiml` / `tech` / `health` / `finance` / `cloud`) + `resume-alignment` / `intro-interview` |
| Content and outreach | `opt-email`, `referral`, voice section of `sandeep-universal-style` |
| Data, Supabase, infra | Supabase MCP, `data:*` skills, AWS MCP |
| Documents (.docx / .pptx / .xlsx / .pdf) | Read skill file before generating |
| Brain dump / task triage | Task Classification Rules in `~/.claude/context/pipelines.md` |

---

---

## Scope Discipline

- Do not let resume, outreach, or writing rules distort coding tasks.
- Do not let coding style rules distort non-engineering writing tasks.
- Infer intent from the existing codebase, repeated user preferences, and accepted implementations.

---

<!-- universal -->

## Skill Auto-Triggers

Load without being asked.

| Signal | Skill |
|---|---|
| Every response from San | `sandeep-universal-style` |
| `/intro-interview` or "interview script" | `intro-interview` |
| Resume + JD in same request | `resume-alignment` |
| "prompt for AI video" | `ai-video-director` |
| `.docx` requested | `docx` |
| `.pptx` / "deck" / "slides" | `pptx` |
| `.xlsx` / "spreadsheet" | `xlsx` |
| `.pdf` creation, merging | `pdf` |
| Figma URL + design task | `design-director` |
| "capture design from URL" | `capture-design` |
| "which matrix" / Veldon Lab design | `veldonlab-design-system` |
| Domain pitch (aiml / tech / health / finance / cloud) | Matching domain skill |
| `/opt-email` / "polish this email" | `opt-email` |
| `/referral` / "write a referral letter" | `referral` |
| "OPT compliant" / F-1 STEM OPT question | `opt-compliant` |
| Multiple agents / concurrent editing / "another terminal" | `agent-coordination` |
| "brainstorm" / "what are our options" / before multi-day build | `brainstorming` |
| "finish this" / "wrap up branch" / "ready to commit" | `finish-branch` |
| `/graphify` | `graphify` |
| Any 127.0.0.1:3456 gateway error / routing 500/502/ConnectionRefused / OpenCode key 429-401 / model invalid after update / Claude or Hermes routing broken | `ai-routing-doctor` |

---

---

## Memory Protocol

**Session start:** Read `~/.claude/memory/MEMORY.md`. Load topic files relevant to the task domain. If `MEMORY.md` is absent: create `~/.claude/memory/` and initialize a blank index.

**During session:** After any significant decision, preference, or project update: write/update the relevant topic file in `~/.claude/memory/`. One file per topic, `kebab-case.md`. Absolute dates only. When San says "remember this": write immediately.

**Ambient learning:** Document consistent but undocumented patterns as `learned-convention` in `~/.claude/memory/conventions.md`.

**Decision logging:** Significant architectural/strategic decisions → `~/.claude/memory/decisions.md` with date and one-line rationale.

**Consolidation trigger:** Run `consolidate-memory` when 5+ new facts accumulate or any topic file has stale dates. Generate a volatile `hot.md` at the end of every session containing the immediate working context. Read this first on the next session start.

**Index rule:** `MEMORY.md` stays under 200 lines. Format: `- [Title](file.md) — one-line hook`.

---

<!-- universal -->

## Changelog Rule

After any code change, feature, bug fix, or architecture update, you MUST:
1. Ensure `CHANGES.md` exists in the project root
2. Add a new entry at the very top using this format:

```
## YYYY-MM-DD HH:MM (UTC) - one-line summary

**Changes:**
- What changed
- Key files modified / created / deleted
- Breaking changes, migration notes, client impact
```

---

## Verification (Auto-Run After Edits)

| Trigger | Automatic Action |
|---|---|
| Edit `functions.php` or any `.php` file | Run `php -l` on the file before declaring done |
| Edit any `.jsx` or `.js` file | Run bracket balance check `()[]{}` on modified files |
| Edit JS/JSX in a Vite project | Run `npm run build` before declaring done |
| Edit component with `return` statement | Verify it does not return `null` without placeholder (if project rule exists) |

### State Safety (Auto-Check During Edits)

| Trigger | Automatic Action |
|---|---|
| Edit auth state file | Scan for new `useState` additions near auth logic; flag fragmentation |
| Edit save/mutation flow | Verify success callback is NOT inside catch block |
| Edit settings mutation | Verify hash/token gets updated after mutation (if project uses optimistic locking) |
| Add new REST endpoint | Verify permission callback is strict (not `__return_true`) |
| Add new admin module | Verify it appears in any required registry/map (if project has one) |

### Convention Enforcement (Auto-Check)

| Trigger | Automatic Action |
|---|---|
| Edit JSX in Preact project | Verify `class` not `className`, `h()` not `React.createElement` |
| Add new route | Verify route is wrapped in error boundary (if project rule exists) |
| Add user-visible text | Check if bilingual/i18n variant needed |
| Add image/media field | Verify it uses project's asset resolver function |

### Documentation (Auto-Maintain)

| Trigger | Automatic Action |
|---|---|
| Any user-facing change | Append entry to `CHANGES.md` (or project's changelog) before declaring done |
| Remove dead code/fields | Update field map/reference docs if they exist |
| Fix a bug from known-issues list | Mark issue as resolved with date |

### Git Safety (Auto-Guard)

| Trigger | Automatic Action |
|---|---|
| Uncommitted changes exist at session start | Warn user and suggest stash or commit before editing |
| Edit creates merge conflict markers | Stop immediately and ask user; never auto-resolve |
| About to run `git commit` | Verify no secrets (`.env`, credentials) are staged |

---

---

## Knowledge & Context

- Vault: `~/Documents/Obsidian Vault` (`OBSIDIAN_VAULT_PATH`)
- Read `CLAUDE.md` at vault root before vault operations
- Context files: `~/.claude/context/` (projects.md, engineering.md, pipelines.md, business.md)
- All tools read/write vault for cross-tool persistence
- Use absolute paths. `[[WikiLinks]]` for connectivity

---

<!-- universal -->

## Integration Points

- **Hermes Daily Brief:** Weekday 6am → `Inbox/`
- **Hermes Weekly Synthesis:** Monday 8am → `Outputs/LLM/`
- **Cross-tool context:** `~/.claude/context/` — project registry, engineering, pipelines, business
- **Skill files:** `~/.config/opencode/skills/` — symlinked to `~/.claude/skills/`, `~/.hermes/skills/*`, `~/.codex/skills/*`
- **MCP servers:** mempalace, fetch, filesystem, git, time — shared across OpenCode, Claude Code, Hermes

---

<!-- universal -->

## Merged Learnings from Projects

### Workflow & Automation Rules

- **Plan before executing.** Produce architecture docs, PRDs, and implementation plans before writing code. If San says "just plan don't execute," produce the plan only.
- **Discord is the universal HITL gate.** Every automated pipeline touching external output routes through Discord for approval before execution.
- **Google Sheets is the boundary layer between CLI tools and n8n.** CLI writes rows. n8n reads rows. They communicate exclusively through the sheet.
- **Always implement DLQ / Circuit Breaker patterns** for failed data pipeline jobs.
- **Always implement a "Fast Test Mode"** (mock data bypassing LLM or external API) during feature development.
- **Always create a detailed Implementation Plan** offering multiple prototype options before starting major UI builds.
- **Do not interrupt long-running processes** (like AI generations) just to check logs. Observe non-destructively.
- **Immediate Credential Rotation** if `.env` is accidentally committed: rewrite git history, push forcefully, rotate all affected API keys.
- **Docker Container Syncing:** Always rebuild the container (`docker compose up --build -d`) rather than assuming changes take effect locally.
- **n8n Debuggability:** Use "Code in JavaScript" nodes (with `$http.request`) for MCP calls. Guarantees full payload visibility.
- **n8n Connectivity:** Use native HTTP Request nodes instead of custom Python inside Code nodes. The Python sandbox is restrictive.
- **The Content Factory Pipeline:** FETCH (source data) → GENERATE (content packaging) → SAVE (to target DB or Notion).
- **Ensure safe retries** with idempotency keys.
- **Prioritize quick wins** to maintain momentum. Ship velocity over perfection.
- **Close open loops before opening new ones.**

### For Engineering Work

Apply `docs/ai/CODE-STANDARDS.md`. That file is the source of truth for:
- Backend and frontend standards
- Type safety
- Parsing and schema safety
- Async and integration patterns
- AI structured output conventions
- Maintenance and self-healing standards review

---



---

## Memory Layer

Three-tier memory model. All layers are additive. L0 is never modified or trimmed by this section.

**L0 — existing CLAUDE.md auto-load (~15.6K words).** Unchanged. Rules + identity + project registry. Source of truth. Stays as-is.

**L1 — `mempalace wake-up` on session start (~789 tokens).** Returns essential context from the top rooms of the current project's wing. Already installed. Add invocation to every session opener for OpenCode, Hermes, and Codex.

**L2 — on-demand queries (zero token cost until invoked).** Surface via MCP or CLI from any tool:
- `mempalace search "..."` for decision and conversation recall
- `graphify query "..."` for code structure questions
- `graphify path A B` for dependency traces

When to use which:
- Decision recall, past rationale, conversation search → `mempalace search`
- Code structure, call graphs, community clusters → `graphify query`
- Bidirectional: any tool can write a decision via `mempalace hook`, any tool can read it next session

Session start protocol: run `mempalace wake-up` before accepting task instructions. If the tool has MCP access, use the MCP tools directly. If not, use CLI.

---

<!-- universal -->



---

## Instruction Hierarchy

1. **Vault `CLAUDE.md`** — `~/Documents/Obsidian Vault/CLAUDE.md` — READ FIRST. Identity, behavioral contract (12 rules), safety rules, formatting hard stops, agent coordination protocol. Single authoritative source for ALL AI tools.
2. This file — Global config for ALL AI tools (Claude Code, OpenCode, Gemini CLI, Hermes, Codex, Copilot)
3. `~/.claude/CLAUDE.local.md` — device-specific overrides, never committed
4. `instructions` loaded from `~/.config/opencode/opencode.json` (OpenCode)
5. Repo local `AGENTS.md` — project-specific overrides when present
6. Repo local `CLAUDE.md` — project-specific overrides (fallback)
7. `docs/ai/CODE-STANDARDS.md` — engineering tasks only
8. `~/.claude/memory/MEMORY.md` — living session context, read at start
9. Task specific skill files loaded on demand

Context imports (load on session start):
@~/.claude/context/projects.md
@~/.claude/context/engineering.md
@~/.claude/context/pipelines.md
@~/.claude/context/business.md

---

<!-- universal -->

---

---

## Cross-Tool Context Sharing

Shared files (one edit propagates to all):
- `~/.claude/CLAUDE.md` — master control plane. OpenCode reads via `opencode.json` relative path.
- `~/.claude/context/` — project registry, engineering, pipelines, business context
- `~/.claude/memory/` — session memory, learned conventions, decisions
- `~/.claude/skills/` — symlinked to `~/.config/opencode/skills/`

Derived files (core + tool-specific tails, synced by script):
- `~/.config/opencode/AGENTS.md` — core + `tails/opencode.md` (Agent Modes, Ralph Commands, Skill Activation)
- `~/.gemini/GEMINI.md` — core + `tails/gemini.md` (Gemini-specific MCP config)
- `.github/copilot-instructions.md` — per repo, from global AGENTS.md + project context

Canonical shared files:
- `~/Documents/Obsidian Vault/CLAUDE.md` — vault instruction layer (identity + 12-rule behavioral contract + safety + agent protocol)
- `~/.claude/CLAUDE.md` — master control plane (stack, memory, triggers, scope routing)
- `~/.claude/context/` — project registry, engineering, pipelines, business
- `~/.claude/skills/` — skill files (symlinked to `~/.config/opencode/skills/`)
- `~/.config/opencode/AGENTS.md` — OpenCode rule file (core + OpenCode-specific tail)
- `~/.codex/AGENTS.md` — Codex instructions (symlinked to `~/.claude/CLAUDE.md`)
- `~/.hermes/skills/` — Hermes skills (15 shared skills symlinked from `~/.config/opencode/skills/`)
- `~/.codex/skills/` — Codex skills (15 shared skills symlinked from `~/.config/opencode/skills/`)

Skill symlink chain: `~/.claude/skills` → `~/.config/opencode/skills` → `~/.hermes/skills/*` + `~/.codex/skills/*`

Git hooks (agent coordination): All repos with `.git` should have symlinks in `.git/hooks/` pointing to `~/Projects/Project26/ai-config-transfer/hooks/git/` (pre-commit, commit-msg, pre-push).

---

<!-- universal -->

---

---

## Second Brain

Vault: `~/Documents/Obsidian Vault` (`OBSIDIAN_VAULT_PATH` env var).
Read `CLAUDE.md` at vault root before any vault operation. It contains identity, 12-rule behavioral contract, 16 active projects, formatting hard stops, safety rules, and agent coordination protocol.
Key automated workflows: Daily Brief (weekday 6am → `Inbox/`), Weekly Synthesis (Mon 8am → `Outputs/LLM/`).
Hermes has full vault access. Other tools read vault via CLAUDE.md.
Hermes handles automated workflows (Daily Brief, Weekly Synthesis). All tools read/write vault for cross-tool persistence.
File paths: absolute only. Use `[[WikiLinks]]` for connectivity.

---

---

---

---

## Code Style

- Python: snake_case, UPPER_SNAKE constants, PascalCase classes.
- JS/TS: camelCase vars/functions, PascalCase components.
- Comments explain WHY, not WHAT.
- Error handling explicit. Never swallow errors silently.
- Infrastructure: Docker Compose, Hetzner VPS, Cloudflare Tunnels, Supabase, n8n, DeepSeek V3.

---

---

---

## Maintenance

- **This file:** Version-bump and re-date when project registry, scope, or core rules change. The sync script propagates core sections to all derived files, appending tool-specific tails.
- **Rules:** Weekly Synthesis reviews rule effectiveness. Drop any rule unused for 4 weeks.
- **Style corrections:** Log to `~/.claude/memory/style-corrections.md`. Map to `sandeep-universal-style` skill.
- **Decision logging:** Significant decisions → `~/.claude/memory/decisions.md` with date and one-line rationale.
- **Memory:** Run consolidation when 5+ new facts accumulate or any topic file has stale dates.
- **Derived files sync:** When this file changes, run `python3 ~/.hermes/scripts/sync-rules.py` or wait for the 6-hour Hermes cron to propagate. Tool-specific tails live in `~/.hermes/scripts/tails/`.

---