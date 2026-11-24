# Roadmap: agent first, robot head later

Build and use the agent on a computer first. Add physical robot capabilities
incrementally after the agent, sessions and memory work well.

This is an ordered product plan. An item being listed does not authorize
implementing it. Track actual work in GitHub issues, with one observable outcome
per implementation issue. The **Status** notes in stage 1 are a snapshot as of
PR #44; the linked issues and PRs remain the source of truth.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the boundaries to preserve.

## Product goal

A stationary robot head that recognizes familiar people, talks through OpenAI API,
remembers each person's explicitly saved information, turns toward the speaker,
and displays expressive eyes and interaction states. It handles simple commands
locally and identifies a single object presented to its camera on request.

The application runs on Raspberry Pi; conversational model inference uses OpenAI API.
Planned hardware: Raspberry Pi 5, Pi Camera, a Hailo AI HAT, microphone, speaker,
small screen, and an Arduino connected to the Pi to control the head servo.
Confirm the exact HAT model and supported workloads before choosing vision models.

## 1. Usable text agent on a computer

Complete this milestone before introducing hardware dependencies. Each numbered
step is a candidate for a separate future issue; split a step if necessary.

1. Project environment, configuration and minimal entry point.
   Status: done (#1, #6).
2. One OpenAI request and a printed response, with clear configuration/API errors.
   Status: done (#1, #4).
3. One fixed local profile with a stable `user_id` (for example `local-user`).
   Resolve it in one place and pass it into application operations. No registration,
   switching UI, camera or recognition is required yet.
   Status: done (#15).
4. Interactive terminal chat with `/exit` and clean handling of empty input.
   Status: done (#19).
5. In-memory conversation history tied to the current user and session;
   preserve tool calls and results as well as text.
   Status: done (#21).
6. Bound per-turn model use before adding more model-driven features: retain only a
   configurable number of recent complete turns, reject oversized input before sending
   it where supported, and set a provider-side output-token limit. Do not call a
   summarizer just to enforce these limits.
   Status: done (#25).
7. One simple PydanticAI tool: current date and time, to exercise model tool calling.
   Keep exact deterministic shortcuts (such as a later `/time` command) local so they
   do not need an OpenAI request.
   Status: done (#27 tool, #41 `/time`).
8. SQLite persistence for the default user, sessions and messages, with restoration
   after restart. Reuse the same user ID; never create a new identity on each run.
   Status: done (#29).
9. Session commands: `/new`, `/sessions`, `/resume`. Scope listing and resume to
   the current user. A new session starts a fresh conversation, not a new profile.
   Status: done (#31).
10. Persistent user memory: save, search, update, delete and inspect remembered facts.
    Initially store long-term facts only when explicitly requested.
    Status: done (#33): `/memory save|list|search|update|delete`.
11. Conversation summaries and relevant context: summary + recent complete turns +
    relevant user memories. Summarize near a context threshold and at the end of
    a meaningful session; avoid repeatedly summarizing unchanged content.
    Status: partly done. Matching saved facts are added to a request when enabled
    (#35). Rolling session summaries are opt-in and triggered by a turn count, not a
    token threshold (#37). Not done: a summary at the end of a session.
12. Persistent usage limits: session and daily request/token budgets, tool-call rounds,
    and summary/speech/vision costs. Persist counters so restarting or opening a new
    session does not reset the daily limit. Per-turn input/output limits are step 6.
    Status: partly done. Opt-in daily request/token budget, persisted in SQLite and
    counting chat and summary requests (#39). A cap on model requests per chat turn
    (#43); `tool_calls_limit` is not used. Not done: session-level budgets.
    Speech and vision costs belong to their stages.
13. Route supported deterministic commands locally, without asking the model to
    decide every time. Include date/time and session commands; reuse these operations
    later from voice and hardware adapters.
    Status: partly done. `/time`, `/new`, `/sessions`, `/resume` and `/memory` run
    locally (#31, #33, #41), but the code lives in `cli.py`, so other adapters cannot
    reuse it yet. Extracting a shared dispatcher waits for a second adapter (see
    #41). Natural-language time questions still go to the model and its tool.
14. Stabilize interrupted requests, failed summary generation and restart recovery.
    Keep saved history if summarization fails; show clear errors without retry loops.
    Status: partly done. A failed summary is skipped with a warning and the chat
    continues (#37). A completed turn is saved in one SQLite transaction, so a failed
    request leaves no partial turn. A failed or interrupted chat request (including
    Ctrl+C) prints a message and returns to the prompt without saving or retrying;
    Ctrl+C at the prompt ends the chat (#45). Not done: recovering an unsaved line.

**Done when:** a terminal conversation survives restart, sessions remain separate,
explicit memories are available in later sessions for the same user, and each model
request stays within configured input, output and usage limits. No hardware is needed.

## 2. Multiple user IDs without a camera

- Add explicit profile creation/selection for development and testing.
- Keep the initial fixed profile and its existing history unchanged.
- Verify isolation using two users: listing/resuming sessions, summaries and all
  memory tools must only access the selected user's data.
- Changing the selected user ends or pauses the current session and selects or
  creates that user's session; it never reassigns the old session's owner.
- Keep one active conversation. Do not add accounts, passwords or a web service.

**Done when:** switching between two local profiles yields separate histories and
memories. Recognition can later select a profile without changing the storage model.

## 3. Raspberry Pi runtime and hardware feasibility

- Run the same agent and SQLite storage on Raspberry Pi 5.
- Add startup/restart handling, useful logs and basic health/error reporting.
- Check camera, microphone, speaker and screen individually.
- Confirm the Hailo board model, software compatibility and supported vision models.
- Measure local speech/vision latency and resource use before choosing implementations.
  Hailo acceleration is optional per workload, not assumed for every model.

**Done when:** the text agent works on Pi with persistent data, and hardware/model
choices have been validated enough to begin integration.

## 4. Voice conversation

- Start with a button or explicit recording action: audio -> text -> agent -> speech.
- Choose speech recognition and synthesis after the Pi feasibility checks;
  prefer local processing where quality and latency are acceptable.
- Handle silence, unrecognized speech, audio failures and recording limits.
- Start with turn-taking; prevent the robot from treating its own speech as user input.
- Send audio to a cloud service only when the selected speech feature needs it; avoid
  continuous upload and account for the cost of each cloud speech path.
- Keep terminal input available for development and recovery.
- Treat text streaming as an optional latency improvement, not a prerequisite. First
  measure time to first audible response. For a separate STT -> agent -> TTS pipeline,
  stream text only if the chosen TTS accepts incremental input and it measurably starts
  speech sooner; buffer chunks into natural phrases and keep the final response/history
  complete. Text streaming to the terminal alone does not add voice or reduce token use.
- A full speech-to-speech realtime API is a separate architecture and cost decision;
  compare it with the turn-based pipeline before choosing it.

**Done when:** one user can hold a spoken conversation without a keyboard after
starting recording, with understandable feedback when audio processing fails.

## 5. Screen expressions and personality

- Local states: idle, listening, processing, speaking, confused/error, sleeping.
- Animate eyes, blinking and small idle variations without model calls.
- Show processing while waiting for a response; this is an application state,
  not a request to expose or generate the model's internal reasoning.
- Add quiet mode, clear offline feedback and a small consistent set of reactions.
- Trigger standard reactions from application events; optional model-selected
  expressions must come from an allowed set and remain nonessential to operation.

**Done when:** the screen reflects actual activity and remains responsive while
the agent is waiting, speaking or unavailable.

## 6. Head movement through Arduino

- Connect Pi to Arduino and define a small command/acknowledgement protocol.
- Start with one servo axis: left/right, center and stop. Add tilt only later.
- Calibrate mechanical limits, bounded speed and safe startup/link-loss behavior.
- Add local commands for turning, centering and stopping; do not execute arbitrary
  model-produced motor values or silently repeat physical actions.
- Coordinate eye direction with head movement. Face following arrives in stage 7.

**Done when:** the head performs allowed movements reliably and handles a lost
connection without uncontrolled or repeated motion.

## 7. Local face recognition and personal sessions

- Detect a face and track its position separately from recognizing its identity.
- Enroll a known user explicitly and associate face data with an existing `user_id`,
  or create a profile deliberately. Keep face data local.
- Match registered faces locally using a validated compatible model; use Hailo
  acceleration where supported.
- Use confidence thresholds and stable observations. Ask or use guest mode when
  identity is uncertain; never guess which personal memory to load.
- Follow the selected person's face with bounded, smooth head movement.
- Seeing a person may trigger attention; only a deliberate interaction starts a
  conversation. Greet by name once per encounter, not once per frame.
- Do not switch the active conversation's owner when another face appears.
- Isolate guest conversations; do not load a known user's personal memory for a guest.

**Done when:** two enrolled users can have separate conversations using recognition,
while unknown/uncertain faces and temporary tracking loss do not mix personal data.

## 8. Natural interaction and local commands

- Add a wake phrase and local speech activity detection after basic voice works.
- Start a session on deliberate activation using the resolved user identity.
- Keep brief pauses or temporary loss of a face within the same session.
- End on goodbye, explicit command or configurable inactivity/absence timeout.
- Save a final summary when appropriate, then return to idle/sleep.
- Handle a different person deliberately taking over without changing the old
  session's owner. Continue to support only one active speaker/conversation.
- Extend local commands: stop, volume, quiet mode, sleep/wake, turn and center.
- For commands advertised as offline, speech/keyword recognition must also be local.
  A local action after cloud transcription is not an entirely offline feature.

**Done when:** users can begin and end conversations without a recording button,
and essential local commands still work when OpenAI is unavailable.

## 9. Recognize a presented object

- Start on explicit request: "What is this?" while one object is held in front
  of the camera. Do not continuously ask the cloud to analyze video.
- Recognize a limited documented set of categories using a validated local model.
- On low confidence, say it is unknown or ask the user to reposition the object.
- Send only the needed label/confidence to the conversational agent when relevant.
- Distinguish recognizing an object from proving which hand/person holds it;
  multi-object hand association is later work.

**Done when:** supported objects can be named on request, and uncertain observations
produce useful feedback rather than confident guesses.

## 10. Integrated robot-head MVP

- Assemble the stationary head, screen, camera, audio and servo into a usable unit.
- Validate the full scenario: recognize -> activate -> talk with personal memory ->
  local command -> identify a presented object -> goodbye -> save -> next encounter.
- Check startup, restarts, network loss, API limits, audio feedback, identity ambiguity,
  movement limits, long-running resource use and responsiveness together.
- Keep idle animations and basic local behavior available without OpenAI.

**Done when:** the complete interaction works repeatedly with more than one registered
profile, one conversation at a time, without losing or mixing user data.

## Beyond the robot-head MVP

- Wheels, drive motors, navigation and obstacle avoidance.
- More servo axes and richer coordinated gestures.
- Interrupting the robot naturally while it speaks.
- Multiple simultaneous speakers and reliable active-speaker identification.
- Arbitrary scene understanding, hand/object association and optional cloud vision.
- Suggested long-term memories with user confirmation and autonomous conversation starts.
- Fully local conversational models, if later desired and supported by the hardware.

## Turning the roadmap into issues

Create implementation issues only as their stage is approached. Describe the user
outcome, prerequisites, scope, acceptance criteria and behavior test scenarios.
Do not bundle this whole roadmap into one implementation issue. Documentation work
records the plan; it does not implement the features it describes.
