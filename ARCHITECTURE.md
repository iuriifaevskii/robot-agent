# Architecture intent

These are boundaries for future implementation, not a description of completed code.
Follow [ROADMAP.md](ROADMAP.md) in order and introduce each component only when its
feature is requested. Keep Python, PydanticAI, OpenAI API and SQLite as the foundation.

## Start small

Keep terminal I/O separate from agent logic. Add identity resolution when introducing
the fixed profile; add session, storage and memory functions with their roadmap steps.
Prefer small modules and functions over a framework, plugin system or extra agents.
Do not create empty hardware adapters, unused classes or speculative database tables.
Keep paid services and dependencies to the existing stack. Handle deterministic work
locally; add a cloud call only for a user-visible capability that needs it, and measure
its API use before enabling speech or vision.

| Responsibility | First implementation | Later evolution |
| --- | --- | --- |
| Identity resolution | Return one fixed profile | Manual selection, then face recognition |
| Input/output | Terminal text | Microphone, speaker and screen |
| Conversation control | User/session context, turn routing | Activation, inactivity and presence events |
| Agent | PydanticAI calling OpenAI | Same agent on Pi with bounded allowed tools |
| Storage | SQLite when persistence is added | Same ownership model for more users |
| Memory/context | Explicit facts, summaries, recent turns | Select relevant facts as volume grows |
| Device control | None during the agent milestone | Validated commands to Arduino and local display |
| Perception | None during the agent milestone | Local face/object observations, Hailo where compatible |

Hardware events and terminal commands should feed the same conversation operations.
The agent and database must not depend on camera frames or Arduino connections.

## Code layout

Application code lives in one import package at the repository root; run it with
`python -m robot_agent`. A flat layout keeps a small CLI simple and needs no install
step. Reconsider a `src/` layout if the package grows past roughly ten modules or
is installed as a distribution on the Pi.

```text
robot_agent/
  __main__.py   entry point, calls cli.main()
  cli.py        terminal input/output, configuration checks, turn handling
                and the local commands
  agent.py      PydanticAI agent and model calls
  identity.py   resolves the current user profile
  tools.py      allowed agent tools (current date and time)
  storage.py    the only module that runs SQL
  sessions.py   create, list and resume sessions for a user_id
  memory.py     explicit user memories
  context.py    pure selection of summary, recent turns and relevant facts
  usage.py      pure daily and session budget decisions and messages
tests/          one test module per module or feature
```

Planned modules not created yet. Create each only with the roadmap step that needs it:

| Module | Responsibility | Why it does not exist yet |
| --- | --- | --- |
| `chat.py` | Conversation turns for the current user and session | Turn handling is still in `cli.py` |
| `commands.py` | Local command dispatcher | `/time`, session and memory commands live in `cli.py`; extract a shared dispatcher when voice needs them (#41) |
| `config.py` | Typed settings loaded from the environment | Configuration checks are still inline in `cli.py` |

Dependency direction: `cli` (and later voice/screen adapters) calls conversation
operations; those call `agent`, `memory`, `sessions` and `storage`. Lower modules never
import `cli` or print. Only `cli` reads the environment and talks to the terminal.
Split a module into a subpackage only when it no longer fits one clear responsibility.

## Current implementation notes

A snapshot as of PR #44, for the places where the code differs from the intent
below. See [ROADMAP.md](ROADMAP.md) for per-step status.

- Storage has `users`, `sessions`, `messages`, `summaries`, `memories`, `active_sessions`,
  `daily_usage` and `session_usage`. `active_sessions` stores each user's selected
  session (set by `/new` and `/resume`) so a restart reopens it. `daily_usage` holds
  device-wide daily counters; `session_usage` holds one row of counters per session,
  read and written only for a session the current user owns. Per-user usage is not
  stored. No face reference table exists.
- Memories keep a `source` column for provenance. Summaries keep `covered_through`
  as their coverage position.
- Summaries are opt-in and triggered by a turn count (`AGENT_SUMMARY_AFTER_TURNS`),
  not by a token threshold. There is no summary at the end of a session.
- The model request cap per turn bounds tool-call rounds indirectly; `tool_calls_limit`
  is not used.
- Local commands (`/time`, session and memory commands) run in `cli.py` before any
  model request. Natural-language time questions still go to the model and its tool.
- Failure handling: a failed summary is skipped with a warning. A failed or
  interrupted chat request (including Ctrl+C) prints a message, saves no turn and
  returns to the prompt; nothing is retried. Ctrl+C at the prompt ends the chat. A
  failure to save a completed turn, an unopenable database and invalid configuration
  still end the process.

## Identity and ownership

- `user_id` is a stable internal identifier, independent of the person's name,
  face embedding, camera tracking ID and current session.
- Initially one resolver returns a fixed ID such as `local-user` and a display name.
  Pass that identity explicitly rather than embedding the constant in every operation.
- When storage arrives, create the default profile once and reuse it after restarts.
  Later enrollment can attach a face to this same profile without losing its history.
- Manual selection and recognition resolve identities before entering the agent.
  The model must not choose or override the ID used by data-access tools.
- A session belongs to exactly one user. Messages and summaries belong to a session;
  long-term memories belong to a user.
- Check ownership for listing, resuming, searching, reading, updating and deleting.
  Bind tools to trusted application context, not model-supplied owner identifiers.
- Guest interactions use an isolated temporary identity/session; they must not fall
  back to the fixed development user. Do not retain a guest's personal memory by default.

## Logical SQLite model

Introduce these entities with their features. This is not a prescribed migration
or final SQL schema; use explicit keys/relationships and add indexes when needed.

| Entity | Ownership and purpose | Introduced with |
| --- | --- | --- |
| User | Stable `user_id`, editable display name | Persistence of the fixed profile |
| Session | `session_id`, owning `user_id`, lifecycle timestamps/state | Session persistence |
| Message | Ordered serialized model messages linked to `session_id` | History persistence |
| Summary | Session-linked summary with the last covered message position | Context compaction |
| Memory | `memory_id`, owning `user_id`, content and provenance | Explicit memory tools |
| Usage | Session/user usage plus device-wide daily counters | API limits |
| Face reference | Local enrollment data linked to an existing `user_id` | Face recognition |

Preserve PydanticAI's supported serialization, including tool requests and results.
Session message order must be explicit and deterministic. Summaries need a coverage
position so rebuilding context does not duplicate or omit already summarized turns.
Retain raw saved history when a summary request fails; never delete it first.

## Sessions and memory

The terminal establishes a user and session before a conversational turn.
`/new` creates a new session for the same user; `/resume` checks ownership;
`/exit` ends the interaction cleanly. Later voice commands invoke these same operations.

Keep three distinct kinds of information:

1. Recent complete conversation turns, including tool calls/results.
2. A compact summary of older session content.
3. Explicitly saved user facts that may be used across sessions.

A summary can describe tentative plans or unresolved questions. It must not silently
promote them into confirmed personal facts. Start with explicit save/update/delete
requests and a way to inspect remembered facts. Preserve provenance for corrections.

## Context and usage budgets

Construct model input from instructions, the selected session summary, a bounded
number of recent complete turns and relevant memories belonging to the current user.
Keep complete tool exchanges together when trimming history. Do not send the entire
database, every past session or continuous camera footage.

For the first text-agent milestone, bound model input before each paid request: keep a
configurable number of recent complete turns, reject an oversized current prompt before
sending it when the provider supports pre-request token counting, and set a provider-side
maximum output-token setting. Do not invoke a model to summarize history just to enforce
a basic limit. Keep the selected model configurable; do not hardcode a premium model.

PydanticAI token and cost limits use reported model usage and may be checked after a
response has already been billed. Treat a `cost_limit` as a best-effort guard, not a hard
billing ceiling. Pair per-call limits with request/tool-call limits. A cumulative
session/day budget needs persisted counters and belongs with SQLite usage tracking.

Summarize when a configured context threshold is approached, and at meaningful session
completion if new content remains. Do not summarize every turn or run an idle LLM loop.
If summarization fails or the budget is exhausted, keep history and use a bounded
fallback context or explain that the conversation cannot continue within the limit.

Track session and device-wide daily usage, including summaries and any cloud speech or
vision calls. Persist counters across restarts; preflight requests conservatively and
reconcile with returned usage. Exact thresholds, model choices and rates belong to
their implementation issues.

## Local actions and hardware

Known deterministic commands go through a local dispatcher before a model request.
This avoids an API call for answers or actions the application can handle locally.
General conversation uses OpenAI. Natural-language tool calls may request only
explicitly allowed actions, with arguments validated by the application; never execute
arbitrary model text.

Pi owns the application, camera/audio/display orchestration and SQLite. Arduino owns
servo execution with calibrated bounds, speed limits, acknowledgements and defined
link-loss behavior. Stop commands take priority over ongoing conversational work.
Movement must not depend on a successful cloud response.

Display and motion controllers react to application states without extra model calls.
"Processing" means waiting for work to finish, not displaying hidden model reasoning.
Use local eyes/blinking/idle animations and a small set of allowed expressions.

## Security and privacy

- Treat terminal text, future speech transcripts and camera/model observations as
  untrusted input. They cannot change system rules, grant tool permissions or select
  a user's data. Prompt text is never authorization.
- Keep the OpenAI key in local environment configuration. Never place it in prompts,
  logs, SQLite content, commits or model-visible tool results. Log only the minimum
  operational details needed to diagnose failures; do not log raw audio, video or
  transcripts by default.
- Keep raw audio, video and face references local by default. Send only the data needed
  for an explicitly selected cloud feature; do not upload continuous audio/video or
  retain raw media unless a feature specifically requires it and defines deletion.
- Model output must never go directly to a shell, arbitrary network call or actuator.
  Tools and local actions use an allowlist, validated arguments and trusted application
  context. Arduino enforces movement bounds and safe link-loss behavior independently.
- Before persisting transcripts, face data or other personal data, define who can read
  it, how long it is kept, and how it can be deleted.

## Recognition and conversation lifecycle

Face detection answers whether/where a face is present. Recognition attempts to
resolve a registered `user_id`. Neither alone proves that someone wants to converse
or that a visible face is the person currently speaking.

Once the later voice/vision stages exist:

1. Observe presence and a stable identity candidate.
2. Begin a session after deliberate activation; resolve ambiguity explicitly or
   use guest mode without personal memory.
3. Bind the session to that user. Brief tracking loss does not end it immediately.
4. Do not reassign ownership when another person appears. An explicit handover ends
   or pauses the old session and starts/selects a session for the new user.
5. Goodbye or configurable inactivity/absence ends the interaction, saves any needed
   summary and returns to idle. A later encounter can start a new session with the
   same user's long-term memory.

Start with one active conversation. Keep face enrollment local and separate from
general conversation history. Choose recognition thresholds and speech/vision models
after hardware validation; do not assume every model runs on the Hailo accelerator.

## Verification as features are implemented

Future behavior issues should cover user isolation, stable identity after restart,
session ownership, complete tool histories, summary coverage/failure, input/output
limits, budget exhaustion, untrusted inputs, data deletion and local commands without
OpenAI. Hardware issues add explicit device scenarios.
Use fake API/model and hardware boundaries in automated tests; live API calls remain
opt-in. This documentation change itself introduces no runtime behavior.
