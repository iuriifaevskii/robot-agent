# Roadmap: agent first, robot head later

Build and use the agent on a computer first. Add physical robot capabilities
incrementally after the agent, sessions and memory work well.

This is an ordered product plan, not a status tracker. An item being listed does
not mean it is implemented or authorize implementing it. Track actual work in
GitHub issues, with one observable outcome per implementation issue.
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
2. One OpenAI request and a printed response, with clear configuration/API errors.
3. One fixed local profile with a stable `user_id` (for example `local-user`).
   Resolve it in one place and pass it into application operations. No registration,
   switching UI, camera or recognition is required yet.
4. Interactive terminal chat with `/exit` and clean handling of empty input.
5. In-memory conversation history tied to the current user and session;
   preserve tool calls and results as well as text.
6. One simple tool: current date and time.
7. SQLite persistence for the default user, sessions and messages, with restoration
   after restart. Reuse the same user ID; never create a new identity on each run.
8. Session commands: `/new`, `/sessions`, `/resume`. Scope listing and resume to
   the current user. A new session starts a fresh conversation, not a new profile.
9. Persistent user memory: save, search, update, delete and inspect remembered facts.
   Initially store long-term facts only when explicitly requested.
10. Conversation summaries and bounded context: summary + recent complete turns +
    relevant user memories. Summarize near a context threshold and at the end of
    a meaningful session; avoid repeatedly summarizing unchanged content.
11. Usage limits: input context, response size, tool-call rounds, session API usage
    and daily API usage. Include summaries in the budget; persist usage counters
    so restarting or opening a new session does not reset the daily limit.
12. Route supported simple commands locally, without asking the conversational
    model to decide every time. Begin with time/date and session commands.
13. Stabilize interrupted requests, failed summary generation and restart recovery.
    Keep saved history if summarization fails; show clear errors without retry loops.

**Done when:** a terminal conversation survives restart, sessions remain separate,
explicit memories are available in later sessions for the same user, and long
conversations stay within configured limits. No hardware is needed.

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
- Account for cloud speech costs if cloud recognition or synthesis is selected.
- Keep terminal input available for development and recovery.

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
