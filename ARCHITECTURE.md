# Architecture intent

These are boundaries for future implementation, not a description of completed code.
Follow [ROADMAP.md](ROADMAP.md) in order and introduce each component only when its
feature is requested. Keep Python, PydanticAI, OpenAI API and SQLite as the foundation.

## Start small

Keep terminal I/O separate from agent logic. Add identity resolution when introducing
the fixed profile; add session, storage and memory functions with their roadmap steps.
Prefer small modules and functions over a framework, plugin system or extra agents.
Do not create empty hardware adapters, unused classes or speculative database tables.

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

Summarize when a configured context threshold is approached, and at meaningful session
completion if new content remains. Do not summarize every turn or run an idle LLM loop.
If summarization fails or the budget is exhausted, keep history and use a bounded
fallback context or explain that the conversation cannot continue within the limit.

Limit input context, output size, tool rounds, session usage and device-wide daily
usage. Track summary calls and any cloud speech/vision calls too. Persist counters
across restarts; preflight requests conservatively and reconcile with returned usage.
Exact thresholds, model choices and rates belong to their implementation issues.

## Local actions and hardware

Known deterministic commands go through a local dispatcher. General conversation
uses OpenAI. Natural-language tool calls may request only explicitly allowed actions,
with arguments validated by the application; never execute arbitrary model text.

Pi owns the application, camera/audio/display orchestration and SQLite. Arduino owns
servo execution with calibrated bounds, speed limits, acknowledgements and defined
link-loss behavior. Stop commands take priority over ongoing conversational work.
Movement must not depend on a successful cloud response.

Display and motion controllers react to application states without extra model calls.
"Processing" means waiting for work to finish, not displaying hidden model reasoning.
Use local eyes/blinking/idle animations and a small set of allowed expressions.

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
session ownership, complete tool histories, summary coverage/failure, budget exhaustion
and local commands without OpenAI. Hardware issues add explicit device scenarios.
Use fake API/model and hardware boundaries in automated tests; live API calls remain
opt-in. This documentation change itself introduces no runtime behavior.
