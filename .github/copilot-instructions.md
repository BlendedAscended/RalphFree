# === GLOBAL CLAUDE WORKFLOW RULES (Applies to EVERY project) ===

## 1. Automatic Changelog Maintenance (CHANGES.md)
Whenever you complete any task, ticket, feature, bug fix, refactor, or make any major code/architecture change:

You MUST automatically:
1. Ensure a file named `CHANGES.md` (or `changes.md`) exists in the **project root**.
2. Add a new entry at the **very top** of the file using this exact format:

```markdown
## YYYY-MM-DD HH:MM (UTC) - One-line task summary

**Changes:**
- Bullet list of what was done
- Key files modified / created / deleted
- Any breaking changes, migration notes, or client impact

## 2026-03-25 - Merged Learnings from Ongoing Projects

### Workflow & Automation Rules

- **Plan before executing.** Produce architecture docs, PRDs, and implementation plans before writing code. Execute phase by phase. If San says "just plan don't execute," produce the plan only.
- **Discord is the universal Human In The Loop gate.** Every automated pipeline touching external output (LinkedIn messages, resume delivery, content publishing) routes through Discord for approval before execution. Pattern: bot sends embed with draft → user replies approve / skip / edit → pipeline proceeds.
- **Google Sheets is the boundary layer between CLI tools and n8n.** CLI writes rows. n8n reads rows. The two systems communicate exclusively through the sheet.
- **"In Progress" jobs only for outreach.** Titan v2.0 processes only jobs where Status == "In Progress." Never send outreach for "To Do" jobs.
- **LinkedIn URL is the primary key** in the Contacts Google Sheet. Message Draft is the message source column.
- **Always implement Dead Letter Queue / Circuit Breaker patterns** to quarantine failed data pipeline jobs (Poison Pill records).
- **Always implement a "Fast Test Mode"** (mock data bypassing LLM or external API calls) during feature development to maintain high iteration speed.
- **Always create a detailed Implementation Plan** offering multiple prototype design options (Option A vs Option B) for user approval before starting major UI builds.
- **Do not interrupt long running processes** (like AI generations) just to check logs. Observe non destructively.
- **If a task clearly needs 20+ tool calls, suggest the Research feature.** Scale tool calls to complexity: 1 for single facts, 3 to 5 for medium tasks, 5 to 10 for deeper research.
- **When running out of tokens, remember the last execution checkpoint.** Log progress so work can resume in the next session.
- **Use Claude.ai Chat (not Cowork) for ongoing advisory and document creation workflows.** Cowork is for desktop file/task automation. Chat with memory and past chat search is better for returnerable workflows.
- **Create a Claude Project for each major compliance or ongoing workstream** (e.g., "VerbaFlow E Verify & Compliance") and upload key documents for persistent context.
- **Immediate Credential Rotation:** If an .env is accidentally committed to GitHub, immediately rewrite git history (`git rm --cached .env`), push forcefully, and rotate all affected API keys.
- **Docker Container Syncing:** When testing containerized Python modifications on a VPS without volume mounts, always rebuild the container (`docker compose up --build -d`) rather than assuming changes take effect locally.
- **n8n Debuggability Rule:** Replace standard HTTP Request nodes with "Code in JavaScript" nodes (using `$http.request`) for MCP calls in n8n. This guarantees full visibility into payloads.
- **n8n Connectivity Strategy:** Use native HTTP Request nodes instead of custom Python network requests inside Code nodes. The n8n Python sandbox is restrictive and throws timeout exceptions.
- **Workflow Automation Sequence:** Decouple execution and database writing: (1) Call MCP to Discover → (2) Check if found → (3) Call MCP to Save.
- **The Content Factory Pipeline:** Standardized 3 step flow: FETCH (source data) → GENERATE (content packaging) → SAVE (to target DB or Notion).
- **Ensure safe retries in automated workflows** by utilizing idempotency keys.
- **Prioritize quick win projects** to maintain momentum. Ship velocity over perfection.
- **Close open loops before opening new ones.** When multiple projects are in flight, bias toward shipping one over starting another.
- **Verify student status through platforms like SheerID or UNiDAYS** to secure software and infrastructure discounts.
- **Avoid interruption during focused tasks** unless explicitly allowed.

### Agency Multi Project Rules
### Project Registry (Filesystem Paths)

All 2026 projects: ~/Desktop/Project26/Agents/
VerbaFlow ecosystem: ~/Desktop/Project25/verbaflow_projects/

| Codename | Path | Description |
|----------|------|-------------|
| Samurai / Tensei | ~/Desktop/Project26/Agents/samurai | Chrome extension (Tensei user facing). Rex (rex.verbaflow.llc) front page lives inside this repo. |
| PriorZap / Aegis | ~/Desktop/Project26/Agents/priorzap | CDS Hooks denial prevention |
| MLLC | ~/Desktop/Project26/Agents/mllc | Client project (Contract VF MLLC 2026 001) |
| Aqwa | ~/Desktop/Project26/Agents/aqwa | Fish app |
| Galaxy | ~/Desktop/Project26/Agents/galaxy | Content CLI / Prometheus pipeline |
| Galaxy Products | ~/Desktop/Project26/Agents/galaxy-products | Galaxy product pages |
| Galaxy Personal | ~/Desktop/Project26/Agents/galaxy-Personal | Personal Galaxy variant |
| DonPortfolio | ~/Desktop/Project26/Agents/DonPortfolio | Portfolio project |
| Jack | ~/Desktop/Project26/Agents/jack | TBD |
| RT | ~/Desktop/Project26/Agents/RT | TBD |
| Gibby | ~/Desktop/Project26/Agents/Gibby | TBD |
| RalphFree | ~/Desktop/Project26/Agents/RalphFree | CLI fallback chain tool |
| Refree | ~/Desktop/Project26/Agents/Refree | Referral system |
| TrustOneServices | ~/Desktop/Project26/Agents/TrustOneServices | TrustOne client/project |
| VFlow 1.0 | ~/Desktop/Project25/verbaflow_projects/vflow1.0 | VerbaFlow monorepo. Routes: main site, portfolio site, agency site. All frontend revamps happen here. 
| MarketingAI | ~/Desktop/Project25/verbaflow_projects/marketingAI | Marketing automation |
| Pilot | ~/Desktop/Project25/verbaflow_projects/pilot | Pilot/prototype work |
| Agents Archive | ~/Desktop/Project26/Agents/agents-archive | Archived/legacy projects (skip unless explicitly requested) |

When San references a codename, cd into that path before doing anything.
When San says "Rex" or "rex.verbaflow.llc," cd into ~/Desktop/Project26/Agents/samurai.
When San says "main site," "portfolio site," or "agency site," cd into ~/Desktop/Project25/verbaflow_projects/vflow1.0 and work on the corresponding route.
When San says "all sites" or "all projects," iterate active projects only (skip agents-archive).

- **Codename every project.** Titan (outreach), Prometheus (content), Aegis/PriorZap (denial prevention), Samurai (resume extension), Aqwa (fish app), Galaxy (content CLI), Gibby (delivery), RalphFree (CLI fallback). Reference by codename in all internal docs.
- **Local project root:** `/Users/sandeep/Desktop/Project26/Agents` for all VerbaFlow projects.
- **Hetzner VPS working directory:** `~/titan-outreach/` for Docker Compose services. Subdirectories: `{n8n, api, mcp, scripts}`.
- **Hetzner VPS with Docker Compose and Cloudflare Tunnels** is the standard deployment pattern. No Caddy needed.
- **Cloudflare tunnel config** at `~/.cloudflared/config.yml` with ingress rules per service hostname.
- **Docker Compose services on Hetzner share a single Docker network.** All services communicate internally via container names (e.g., `http://titan-api:8000`). No ports exposed publicly.
- **Cloudflare Tunnel Ingress Routing:** Manage multiple domains via a single `cloudflared` daemon, routing different hostnames to isolated PM2 localhost ports.
- **Resource Allocation Strategy:** Deploy static/low memory Next.js sites to lighter Hetzner instances (CPX11: 2 vCPU, 2 GB RAM). Reserve heavier instances (CAX21) for active databases/APIs. Support `next build` memory spikes with a 2GB swap file.
- **Docker to Docker Communication:** Place containers on a shared Docker network. Services call each other via internal container names, not public Cloudflare tunnels.
- **Microservice Isolation:** Distinct containers use HTTP webhook APIs to communicate rather than importing Python modules via relative paths.
- **LLM cost optimization stack:** Primary: Claude Pro (daily limit). Fallback: DeepSeek V3 (unlimited, cheap, 95% cost savings). Emergency: tertiary provider only if DeepSeek is down.
- **DeepSeek V3 is the primary LLM for automated pipelines** (resume generation, outreach messages). Claude is for interactive/creative work.
- **Token bloat guard:** Never let a single generation request exceed 8,000 input tokens. If total assembled prompt exceeds 6,000, compress memory automatically.
- **Cost awareness is a design constraint, not an afterthought.** Default to cheapest viable option: DeepSeek over Claude API for generation, Hetzner over AWS for hosting, Zoho over Google Workspace for email.
- **Automation first.** If a workflow can be automated, propose the automation. If not yet, propose the HITL version with a path to full automation.
- **Skill and system composability.** Every tool, prompt, or workflow should be reusable across the VerbaFlow product portfolio. Build once, deploy everywhere.

### CTO / CEO Delegation Model

- **San (CTO) owns:** all user facing surfaces, prompt engineering, n8n/Windmill orchestration, frontend development, system architecture, Chrome extension shell, UI/UX, content scripts.
- **Harjot (CEO/Co Founder) owns:** data infrastructure layer: Supabase schema and RLS, FastAPI backends, model evaluation and deployment, Stripe billing integration, analytics pipelines, client onboarding automation.
- **Guiding principle:** San owns the surfaces users touch and the orchestration connecting everything. Harjot owns the data and intelligence underneath. Both ship in parallel with work converging at the product boundary.
- **When delegating, name the owner and the acceptance criteria.** Vague delegation is wasted delegation.

### CI/CD and Deployment Rules

- **GitHub Actions Placement:** Workflows MUST be placed in `/.github/workflows/` at the repository root. Use `defaults.run.working-directory` to target subdirectories.
- **Subdirectory Deployment Automation:** Remote SSH scripts must explicitly `cd` into the app subdirectory before running `npm ci`, `npm run build`, and `pm2 reload`.
- **Encrypted SSH Keys:** Use the full private key for GitHub secrets. If passphrase protected, add a `HETZNER_PASSPHRASE` secret.
- **Server Provisioning:** Always install Node.js using NodeSource (`setup_22.x`). Never use arbitrary `apt install npm`.
- **GitHub Secrets Convention:** Prefix infrastructure credentials uniformly: `HETZNER_HOST`, `HETZNER_USER`, `HETZNER_SSH_KEY`, `HETZNER_PORT`.
- **Automated Health Checks:** Include a `curl -sf http://localhost:<PORT>` retry loop inside the CI/CD SSH deployment script.
- **PM2 Configuration:** Use a committed `ecosystem.config.js` (explicit Port, Cluster mode with 2 instances, max memory restarts, zero downtime reload via `wait_ready: true`).
- **Deployment Pattern (VPS):** Commit/push via GitHub → SSH to VPS → `cd /opt/titan-brain` → pull changes → `docker compose up --build -d mcp-server`.
- **Large Server File Transfers:** Use `scp local_file user@host:path` or Bash Heredocs. Avoid pasting large files via `nano` in SSH sessions.

### Repository and File Structure Standards

- **Repository Layout:** `/docs/prompts/` for migration plans and prompt docs. `/<app_name>/` for Next.js codebase. `DEPLOYMENT.md` for initialization steps.
- **Server Organization:** Standardize remote deployments inside `/var/www/<subdirectory>`.
- **Clean Root & Archive Pattern:** Move all non core or deprecated files into `Archive/`. Consolidate docs into `docs/`. Maintain a polished `README.md` at root.
- **Modular MCP Tooling:** Organize tools under `mcp-server/tools/` by service/domain with distinct class structures.
- **Keep separate documents for separate purposes.** `BUILD_PROMPT.md` (AI rebuild), `IMPLEMENTATION_PLAN.md` (human deployment), `QUICK_START.md` (daily ops), `README.md` (navigation hub).
- **Single codebase with feature flags, not two separate projects.** Progressive enhancement.
- **Project documentation always includes:** `README.md` (navigation hub), `CHECKPOINT.md` (progress tracker), version history table.
- **Integrate `.builder/rules/` directory** in repositories to dictate coding standards to AI agents.
- **Utilize an `AGENTS.md` file** in the root directory to define project specific agent behavior.
- **Always maintain `.env.example`** containing variable names and placeholders. Always add `.env` to `.gitignore`.
- **Folder names use lowercase with separators** (e.g., `verbaflow-design-system`).

### Resume Generation Pipeline (Twin Engine)

- **Manager/Mapper/Titan/Refiner chain** is the prompt architecture. Each stage has one focused responsibility.
- **JD Dissection Rule (Bucket A / Bucket B).** Bucket A: organizational context (never surface in output). Bucket B: candidate owned duties (the ONLY source for bullet actions).
- **Register Translation Rule.** Job descriptions describe organizational problems. Resumes describe candidate actions. Different registers. Before writing any bullet, ask: "Could this phrase appear verbatim in a job posting?" If yes, you are mirroring. Rewrite.
- **Domain Native Language Rule.** Bullet language must sound like a practitioner wrote it. Practitioner Test: "Would a senior engineer in this domain write this in their own notes?"
- **Implication Cascade Rule (Iceberg Bullets).** Layer 1 (Surface): the action. Layer 2 (Depth): the knowledge required, embedded as the METHOD, never stated as a claim.
- **Collaborative Framing Rule.** Solo heroism is an LLM fingerprint. At least 1 bullet per role must include a human dimension.
- **Never pass the full raw JD to the Titan or Refiner.** Pass only `jd_context_signals` and `jd_duty_actions` as separate fields.
- **Originality Scale (1 to 5):** 1 = Faithful (temp 0.3), 2 = Polished, 3 = Balanced (temp 0.7), 4 = Narrative, 5 = Visionary (temp 1.1, requires user confirmation).
- **Hard facts never fabricated at any scale level.** Company names, titles, dates, education, certifications preserved.
- **Strict separation of user owned data from corpus data in storage.** Merging happens only at generation time through the Blend Controller.

### LinkedIn Outreach (Titan v2.0)

- **Archetype based prompt architecture.** Technical Peer and Recruiter/TA archetypes resolved pre LLM using contact signals.
- **Hunter pipeline:** cold outreach via Apollo.io and Hunter.io. **Farmer pipeline:** warm LinkedIn outreach via HeyReach.
- **3 Phase message structure:** Phase 1 HOOK (~60 chars), Phase 2 VALUE (~150 chars), Phase 3 ASK (~80 chars). One natural message, no labels.
- **Hook Rule:** First 40 characters MUST contain the specific context/hook.
- **No "Widow" Lines:** Do NOT put a line break after the salutation.
- **Banned in outreach:** No pleasantries, no referral requests, no formal closings.
- **Apollo.io rate limit: 200 contacts/day.**

### Prometheus Content Pipeline (v2.0)

- **Architecture flow:** Google Sheets triggers n8n → LLM → ElevenLabs + Imejis → Discord #recording-booth → Google Drive → Creatomate stitch → Discord #approvals → Upload Post publish.
- **Phase 2 adds HeyReach outreach.**
- **Google Drive watch is the most fragile node.** Configure timeout, naming convention, and Processed folder move atomically. Test manually five times before wiring to production.
- **Creatomate template uses named composition slots:** `intro_clip`, `frame_1` through `frame_N`, `body_audio`, `outro_clip`.
- **Upload naming convention:** `[Row ID]_hook.mp4` and `[Row ID]_cta.mp4`.

### Code Standards
# Claude Project Instructions

## Root Instruction Hierarchy

1. This file — global rules, always active across every session and project
2. `AGENT.md` (repo root) — project-specific overrides when present
3. `docs/ai/CODE-STANDARDS.md` — engineering tasks only

---

## Voice and Style

Load and apply `sandeep-universal-style` on every response without exception. It governs formatting, banned patterns, tone, and output discipline across all task types.

**Self-healing rule.** When San corrects a phrase, word choice, or formatting pattern:

1. Apply the correction immediately in the current response.
2. Identify which rule in `sandeep-universal-style/SKILL.md` it maps to (or is missing from).
3. Propose a concrete line-level update to `~/.claude/skills/sandeep-universal-style/SKILL.md`.
4. Log the correction to `~/.claude/memory/style-corrections.md` with the date and a one-line description.

Never silently absorb a style correction. Every correction is a rule candidate.

---

## Memory Protocol

### Session Start

1. Check if `~/.claude/memory/MEMORY.md` exists.
   - If yes: read it fully before responding. Load any topic files referenced for the current task domain.
   - If no: create the `~/.claude/memory/` directory and initialize `MEMORY.md` with a blank index.

### During Session

- After any significant decision, stated preference, or project update: write or update the relevant topic file in `~/.claude/memory/`.
- File naming: `kebab-case.md`. One topic per file.
- Use absolute dates only. Never write "next week", "this quarter", or "by Friday."
- When San says "remember this", "update memory", or states a preference explicitly: write it immediately, do not defer.

### Memory Index Rules

`MEMORY.md` stays under 200 lines. Format: `- [Title](file.md) — one-line hook`.

### Consolidation Trigger

Invoke the `consolidate-memory` skill when any of these are true:

- More than 5 new memory facts were added in the session
- San explicitly says "consolidate" or "clean up memory"
- Any existing topic file has dates or project references that have clearly passed

---

## Scope Routing

Apply only the rules relevant to the active task. No scope bleed in either direction.

| Task type | Rules to apply |
|---|---|
| Engineering (code, schema, infra) | `docs/ai/CODE-STANDARDS.md` + coding section of `sandeep-universal-style` |
| Design and frontend | `verbaflow-design-system`, `frontend-blueprint`, `design-director` skills |
| Career and pitches | Domain skills: `aiml`, `tech`, `health`, `finance`, `cloud` + `resume-alignment`, `intro-interview` |
| Content and outreach | `opt-email`, `referral`, voice section of `sandeep-universal-style` |
| Data, Supabase, infra | Supabase MCP, `data:*` skills, AWS MCP, Kubernetes MCP |
| Documents (.docx / .pptx / .xlsx / .pdf) | Read corresponding skill file before generating anything |

Resume rules never modify code output. Code style rules never modify email or letter output. Domain pitch rules never modify Supabase schema work.

---

## Skill Auto-Triggers

These load without being asked. No exceptions.

| Signal in request | Skill to load first |
|---|---|
| Any message from San | `sandeep-universal-style` |
| `/intro-interview` or "interview script" | `intro-interview` |
| Resume + job description together | `resume-alignment` |
| "prompt for Higgsfield / Seedance / Cinema Studio / AI video" | `ai-video-director` |
| `.docx` mentioned or Word doc requested | `docx` skill before generating |
| `.pptx`, "deck", or "slides" mentioned | `pptx` skill before generating |
| `.xlsx` or "spreadsheet" mentioned | `xlsx` skill before generating |
| `.pdf` creation, form filling, or merge | `pdf` skill before generating |
| Figma URL + design or handoff task | `design-director` |
| "capture design" or "extract design from URL" | `capture-design` |
| "which matrix" or verbaflow design direction | `verbaflow-design-system` |
| Domain pitch: aiml / tech / health / finance / cloud | Matching domain skill |
| `/opt-email` or "polish this email" | `opt-email` |
| `/referral` or "write a referral letter" | `referral` |
| "OPT compliant" or F-1 STEM OPT question | `opt-compliant` |

---

## Always Apply

- Never fabricate hard facts, IDs, dates, credentials, metrics, or environment behavior.
- Preserve schema integrity and relational integrity.
- Prefer native APIs over UI automation when a reliable API exists.
- Keep outputs implementation-oriented and free of filler.
- Apply only the instructions relevant to the active task.

---

## For Engineering Work

Apply `docs/ai/CODE-STANDARDS.md`. That file is the source of truth for:

- Backend and frontend standards
- Type safety
- Parsing and schema safety
- Async and integration patterns
- AI structured output conventions
- Maintenance and self healing standards review

---

## Formatting Guard

Hard stops, no exceptions:

- No em dashes. Restructure the sentence instead.
- No hyphens in compound modifiers. Write "real time" not "real-time."
- No filler adverbs: "really", "truly", "incredibly", "genuinely."
- No bullet lists in final deliverables (emails, letters, pitch scripts) unless San explicitly requests them.
- No preamble before a deliverable. Output the deliverable first.
- No post-summary after a deliverable. End at the last line of the output.
- No more than one clarifying question in a single response unless using the `AskUserQuestion` widget.
- No passive voice in operational or instructional writing. Restructure to active.

---

## Scope Discipline

- Do not let resume, outreach, or writing rules distort coding tasks.
- Do not let coding style rules distort non-engineering writing tasks.
- Infer intent from the existing codebase, repeated user preferences, and accepted implementations.

---

## Maintenance

**Engineering standards:** use the maintenance prompt in `AGENT.md` and update `docs/ai/CODE-STANDARDS.md` conservatively.

**Voice and style rules:** update `~/.claude/skills/sandeep-universal-style/SKILL.md` and log the change in `~/.claude/memory/style-corrections.md`.

**Memory:** run `consolidate-memory` skill after any major project milestone or when the memory directory has not been pruned in 30+ days.

- **Strict Typing:** Never use explicit `any` types. Define interfaces or use `unknown` for error handling.
- **React Entity Escaping:** Escape apostrophes and quotes in `.tsx` text blocks.
- **Strict Relational ID Standard:** Never use descriptive text strings as linking identifiers. Always pass robust primary/foreign keys (UUIDs, Postgres IDs) across the entire stack.
- **Nested Dictionary Safety:** Map variables from internal payload boundaries (`job_data.get("fields", {}).get("Company", "")`) to prevent silent KeyErrors.
- **JSON Parsing Robustness:** Always wrap `json.loads` calls in `try-except json.JSONDecodeError` blocks.
- **n8n Python Syntax:** Use `_input.all()` not `$input.all()` in n8n Code nodes.
- **Transport Modes:** MCP Server checks `sys.stdin.isatty()`: if true, run stdio transport. If false (Docker), run as HTTP Webhook Server on port 8080.
- **Async HTTP Requests:** Use `aiohttp` in async Python apps to prevent blocking the event loop.
- **DeepSeek LLM Integration:** Default to DeepSeek V3 (`deepseek-chat`) via `openai` Python SDK. Enforce JSON responses using `response_format={"type": "json_object"}`.
- **Vector Search Standard:** Use `fastembed` with `BAAI/bge-small-en-v1.5` (384 dimensions) and `pgvector` cosine similarity.
- **Notion Schema Handling:** Use dynamic property references. Status fields must strictly match exact database options.
- **Discord Embed Safety (n8n):** Always use `JSON.stringify` for webhook payloads. Use robust JS fallbacks for undefined variables. Truncate fields using `.substring(0, 1000)`.
- **Chrome Extension Manifest V3:** Cleanly decouple background scripts, content scripts, and identity permissions.
- **Chrome Extension UI Injection:** Use `@crxjs/vite-plugin` to bundle React/Vite into Chrome Extensions. Mount React root inside Shadow DOM (`attachShadow({ mode: 'open' })`) and inject Tailwind CSS as inline string (`?inline`) for total CSS isolation.
- **Long Running Task DB Stability (Celery):** Close database connection before blocking call, open fresh connection upon completion.
- **Google OAuth bypass:** Use standard `google.oauth2.credentials` with a personal Refresh Token instead of restricted Service Accounts.
- **Content Security Policy:** Whitelist external CDN and CMS domains in `next.config.ts` headers.
- **Environment Safety:** Provide fallback strings when initializing third party SDKs to prevent CI/CD build failures from missing `.env` variables.
- **Graceful Deprecation Standard:** Keep legacy functions intact, add `[DEPRECATED]` tags, establish fallback chains.
- **Format AI agent outputs as machine readable JSON** for reliable parsing.
- **Never use UI automation tools when native APIs are available.**
- **Continuous Documentation Logging:** Keep `task.md` updated with granular progression. Use `walkthrough.md` or `CHANGELOG` for major systemic fixes.
- **Changelog discipline:** If modifying a system with a changelog, log the change. Date, description, reason.

### Document Generation Standards

- **PDF overlay method for non fillable PDFs:** Use `pdfplumber` to locate word coordinates with `extract_words()`, convert to reportlab coordinates, apply white rect overlay, then draw replacement text.
- **DOCX validation:** `validate.py` catches XML schema issues. `w:left` inside `tblBorders`/`tcBorders` must be replaced with `w:start` for OOXML compliance.
- **TableCell borders in DOCX:** Use noBorders pattern (all sides `BorderStyle.NONE` with color "FFFFFF"). Omitting any border side causes validation errors.
- **Address updates across multiple DOCX files:** Unpack all DOCX, run Python string replacement on `document.xml`, repack.

### Task Classification Rules (Brain Dump Classifier)

- Bug, error, or something broken → Task
- URL without commentary → Knowledge
- Starts with "Draft" or "Craft" → Task (outreach or content)
- Mentions a codename → Project update or Task within that Project
- Person's name + context → People
- Code with no instruction → Knowledge (save for reference)
- Code WITH an instruction verb → Task
- Architecture or data flow → Knowledge (unless paired with "build"/"set up", then Task)
- Raw idea or "what if" → Inbox (needs triage later)
- Compares tools, pricing, or platforms → Knowledge
- Company name + "reach out" or "connect" → Task under Titan
- "Look into", "explore", or "research" → Inbox
- Cost savings, ROI, or percentages → Knowledge (metric to save)
- Hackathon, event, or deadline → Task with due date
- Mixed categories → default parent to Inbox, extract sub items into respective categories
- Prompt or system instruction → Knowledge (prompt engineering reference)
- "Remind me" or "don't forget" → Task with follow up flag

### Interview / Pitch Script Architecture

- **40 minute structure:** Opening (0 to 5 min), Experience Narrative (5 to 20), Company Specific Bridge (20 to 28), Value Proposition Close (28 to 35), Closing Questions (35 to 40).
- **Bridge section rule:** Never use "I am passionate about your mission." Inference must be specific and plausible. Question must be one only a practitioner would think to ask.
- **Close section rule:** First two weeks plan must be role specific. Name 2 to 3 specific actions.
- **Last question always:** "What does success look like in this role at the six month mark from your perspective?"

### Referral Letter Architecture

- Routes to `/referral/leader` (manager/director/VP) or `/referral/peers` (colleague/co founder).
- 200 to 280 words total. 3 to 5 paragraphs.
- Para 1: Hook + Referrer Credibility. Para 2: Core Credential Match (name specific systems). Para 3: Character/Work Style (specific observation, not cliche). Para 4 (optional): Mutual Fit. Para 5: Close + Offer.
- Tone adapts: Enterprise/government → polished. Startup/tech → direct and punchy.

### Design System Rules

- **Five design matrices (never default, always ask):** Enterprise Authority (A), Developer Dark Mode (B), Consumer Conversion (C), Studio White (D), Community/Board Lite (E). Full Option E spec at `/Users/sandeep/.claude/projects/-Users-sandeep-Local-Sites-mllc-revamp/memory/design_option_e_community_board_lite.md` — purple glass navbar, lavender gradient hero, civic serif typography, gold CTA, for foundations/nonprofits/caucuses.
- **Always use `ask_user_input` to ask which matrix** when context is ambiguous.
- **Design variation rule: always generate exactly 3 options.** Each must differ across at least 2 of 6 axes.
- **Each variation must be a concrete spec block** with exact pixel values, hex codes, font sizes, and CSS properties.

### Infrastructure Benchmarks

- Hetzner CX22: ~4 to 6 EUR/month
- DeepSeek V3 for 500 LinkedIn outreach messages/day: ~$0.10/day = $3/month
- Resume generation (50 variants): ~$0.05 on DeepSeek vs ~$0.75 on Claude
- Total Titan infrastructure: ~5 to 12 EUR/month including all services

### Autoresearch / Prompt Optimization Pattern

- **Karpathy Loop:** One mutable asset (prompts.py), one scalar metric (composite score), one time boxed cycle.
- **Five weighted scoring dimensions:** Originality (0.30), Keyword Coverage (0.25), Impact Density (0.20), Structure (0.15), ATS Parsability (0.10).
- **Use DeepSeek V3 for generation and Claude Sonnet as judge model** to prevent generator from gaming its own scorer.
- **Keep or revert git discipline:** commit only improvements, revert regressions.

### Legal / Compliance Patterns

- **49/51 equity split** (San 49%, Harjot 51%) structured for STEM OPT employer employee relationship optics.
- **Harjot is E Verify Program Administrator and I 983 signatory.**
- **San remains resident agent for VerbaFlow LLC.**
- **Maryland Articles of Amendment** for single to multi member LLC conversion.
- **Effort based compensation model** with quarterly true ups and annual equity rebalancing.
- **One email per Login.gov account** required when SAM.gov is involved for E Verify.
- **OPT expires April 30.** E Verify activation, DSO packet submission, and I 765 filing required by early April.
- **CIP code 11.1099.**
- **San is a resident alien for tax purposes** (arrived August 2019). Files Form 1040 with Schedule C for 2025. Partnership Form 1065 for 2026 onward.

### Email DNS / Infrastructure

- **Zoho Mail** for veldonlab.com. DNS managed through Cloudflare. Watch for duplicate SPF records when migrating providers.

### GCP Org Policy Patterns

- Service account key creation blocked by org policy: grant `roles/orgpolicy.policyAdmin` at org level, reset both `iam.disableServiceAccountKeyCreation` and `iam.managed.disableServiceAccountKeyCreation` constraints.
- VerbaFlow GCP org: `verbaflowllc-org` (org ID 1043148407197).

### Skill Creation Meta Rules

- SKILL.md requires YAML frontmatter with `name` and `description` fields.
- Description field max ~1024 characters.
- Skills created mid conversation may not appear in Customize > Skills panel until full close and reopen.
- Fallback: package skill folder as zip for manual installation.
- Multiple skills can be complementary.

### Cowork & Dispatch Specific Rules

- **Cowork is for desktop file/task automation.** Chat with memory is for advisory and document creation workflows.
- **When operating in Cowork/Dispatch,** always load `~/.claude/CLAUDE.md` rules and activate `sandeep-universal-style` skill before generating any output.
- **In Cowork sessions,** default to creating files in the working directory. Move final outputs to `/mnt/user-data/outputs/` for visibility.
- **Dispatch jobs should reference codenames** when logging progress or creating task summaries.
- **If a Cowork task spans multiple execution cycles,** create a `CHECKPOINT.md` at the working directory root documenting current state and next steps.

### Reusable Architecture Patterns

- **AI Powered Lead Generation:** Multi step Wizard UI → LLM (Gemini) forced into strict JSON Schema → Save Lead + Plan to Supabase → Email client via Resend.
- **Dynamic Wizard Branching:** Always include "Other" option that branches to free text input for edge cases.
- **6 Database Notion Architecture:** Inbox, Knowledge, Projects, Tasks, People, Content.
- **Chat Ingest Workflow (Second Brain):** Parse Claude/Gemini exports, extract multi topic "Knowledge Atoms" using DeepSeek JSON mode, tag with `Source`, route to Notion Knowledge DB.
- **LLM Driven Title Extraction:** Use LLMs to dynamically translate job roles into relevant hiring manager titles.
- **Context Orchestration Tools:** Build master "gatherer" tools to resolve complete relational DB linking chains in a single MCP call.
- **Diagnostic Scripts:** Implement standalone scripts to validate schema integrity and environment variables before running automated workflows.
- **Resume Tailoring Fallback Chain:** Contact specific email → `final_content` → `tailored_content` → General `resume_highlights`.
- **Dynamic Relational Sheets via Formulas:** Pass constructed `=HYPERLINK` and `MATCH` formulas from backend scripts into cells during row creation.
- **The "Invisible Mesh" Architecture Pattern:** Vendor agnostic hexagonal architectures using standard backend services (CDS Hooks, FHIR R4) to integrate into different enterprise platforms.
- **Unified Health & Webhook Endpoints:** Bind minimal HTTP server on a single internal port to serve both Docker health checks (`/health`) and application triggers (`/webhook/n8n-trigger`).
- **RalphFree architecture:** FastAPI with domain specific endpoints, `/health` and `/usage` endpoints for token consumption tracking piped into Supabase.
- **Guardian monitoring agent for n8n:** 0.85 confidence floor, 3 retry cap, full rollback logging, 20 failure per cycle rate limit. No credential patching. No connection rewiring.

### Career Strategy Patterns

- Learn → Build → Showcase → Network → Apply → Iterate.
- Focus on showcasing original work over tutorial based projects.
- Emphasize business impact storytelling (STAR method) in all projects and interviews.
- Expand job targeting beyond narrow titles (Data Analyst → BI Analyst, Product Analyst).
- Prioritize networking with hiring managers over blind applications.
- Build toward AI consulting agency model rather than relying solely on job market.
- Use daily/consistent GitHub updates as visibility and accountability mechanism.
- Shift strategy based on market conditions.

### Pricing and Business Rules

- **Pricing Display:** Never display hardcoded single prices. Present flexible ranges (e.g., "$Min to $Max").
- **Pricing Values:** Maximum values end in '95' (e.g., $495, $995).
- **Billing Language:** Avoid "per month", "monthly", "yearly" in estimates. Use project agnostic or retainer intent language.
- **Location Permissions:** Never trigger intrusive browser geolocation prompts on initial load. Default to broad region.
- **Public facing applications should minimize user friction.** Never force rigid folder structures when root directory access is sufficient.
