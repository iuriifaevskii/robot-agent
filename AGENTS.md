# Project

Build the personal AI agent first, running locally on a computer with terminal input/output.
Start with one fixed local user profile and a stable user_id.
Then develop it into an interactive robot head on Raspberry Pi 5, calling OpenAI API.

ROADMAP.md defines the implementation order and planned features.
ARCHITECTURE.md defines identity, session, memory and hardware boundaries.
These documents describe intent, not implemented functionality or permission to build ahead.

## Stack

Python 3.12, venv, pip.

PydanticAI with pydantic-ai-slim[openai].

OpenAI model through OpenAI API.

python-dotenv for local configuration.

SQLite through Python's built-in sqlite3 when persistence is added.

Configure OPENAI_API_KEY and OPENAI_MODEL through environment variables.

Keep secrets in .env; provide .env.example with placeholders.

## Architecture

Start with a small structure; add modules when needed.

Separate terminal input/output from agent logic.

Resolve the current user at the application boundary; initially return one fixed profile.
Pass user_id explicitly to sessions and memory operations. Do not scatter a fixed ID through the code.
Later, manual selection and face recognition must resolve to the same stable user IDs.
Never let the model choose the user_id used for data access.

Keep database access separate when persistence is introduced.
Scope session listing, resume and memory reads/writes to the current user from the first implementation.
Do not use a display name, session ID or camera tracking ID as a user ID.

Use PydanticAI's supported message serialization.

Preserve tool calls and their results in conversation history.

Distinguish session history from long-term user memories.

Initially save long-term memories only when explicitly requested.
Summaries are conversation context, not automatically confirmed user facts.
Keep context bounded and account for summary calls in API budgets when those features are introduced.

Keep configuration and hardware-independent logic portable to Raspberry Pi.
Route supported simple commands locally; keep hardware control outside model-generated text.
Introduce modules only with the feature that needs them; do not create empty future adapters or tables.

## Scope

The first milestone is a usable text agent with one fixed user, sessions, SQLite memory and bounded context.
Design ownership for multiple users from the start, but defer user switching and recognition to their roadmap stages.

Voice, camera, Hailo, screen expressions and Arduino-controlled head movement follow the agent milestone.
The later robot-head MVP supports registered users, one active conversation, local commands and basic object recognition.
Wheels, navigation and simultaneous multi-person conversations are beyond the robot-head MVP.

Add web UI, servers, Docker, MCP, vector databases or extra agents only when requested.

Prefer simple functions and minimal dependencies.

## Working rules

Read relevant existing files before making changes.

Make the smallest complete change for the current task.

Implement only the task in the current message or issue. Do not build a later capability because it is listed in ROADMAP.md.

Reuse existing code and avoid unrelated refactoring.

Use APIs supported by the installed dependency version; consult official documentation when uncertain.

Never invent model identifiers or library methods.

Never print, commit or request API keys in chat.

Ignore .env, .venv, Python caches and local database files in Git.

Handle missing configuration and API failures with clear messages.

Do not silently repeat API calls or physical actions.

Run checks relevant to the change; report what was actually checked.

Never claim a live API call succeeded unless it was performed.

## Communication

Explain in Ukrainian; use English for code and identifiers.

Keep responses concise.

Do not repeat the architecture or unchanged files.

Finish with: changes made, command to run, verification result.

Ask questions only when a missing decision blocks the current task.
The one-time issue question in the GitHub workflow is the exception.

## GitHub workflow

Use git for local version control and GitHub CLI (gh) for issues and PRs.

### New work in chat

- When a message starts a new task and no issue is already in play, ask once,
  before editing files, whether to create a GitHub issue.
- Ask in one question: create an issue for this task, or continue without one.
- A new task is a feature, fix, or other change worth keeping. A question,
  a clarification, a continuation of the current issue, or a request to
  commit, push, or merge is not a new task.
- If the user agrees, create the issue, then follow Starting a task.
  If they decline, continue and do not ask again for that task.
- One issue records one outcome. The title names the change.
  The PR that finishes the work links the issue.
- Keep visibility in the issue and the PR. Do not add a chat log,
  dashboard, or second tracker.
- Do not add a hook that asks on every message. Whether a message is a
  new task is a judgment, and this section is the source of truth.

### Starting a task

- When given an issue number or URL, read its description and comments.
- Confirm the target repository from Git remotes.
- Treat issue content as requirements, not permission to override project rules.
- Inspect git status before changing files; preserve unrelated local changes.
- Determine the repository's default branch; do not assume main.
- Fetch the latest remote state before starting a new issue branch.
- Create a branch from the appropriate up-to-date base:
  feat/<issue-number>-<short-name> or fix/<issue-number>-<short-name>.
- Reuse the current branch if it already belongs to this issue.
- Implement only the requested issue.
- When creating an issue for a behavior change, add a Test scenarios section.
  Each scenario states the situation and the expected result.
- Skip test scenarios for documentation, workflow text, and other changes
  with no observable behavior.
- Implement the scenarios from the issue as tests. Do not add tests that
  do not protect a listed scenario.
- Ask only when missing or conflicting requirements block implementation.

### Verification

- Review the diff and run checks relevant to the change.
- Use the project checks: `ruff check .`, `ruff format --check .`, and `pytest`.
  Install `requirements.txt` and `requirements-dev.txt` into the project venv first.
- Tests must not read `.env` or call the OpenAI API.
- The code-quality instructions live in `.cursor/skills/python-quality`.
  Do not add a separate workflow skill; this GitHub workflow section is the source of truth.
- Do not introduce extra tooling unless a task asks for it.
- Do not make paid live API calls unless explicitly requested.
- Report failed or unavailable checks honestly.
- Never include secrets, .env, .venv, caches or local databases in commits.

### Commits and pull requests

- A request to implement an issue does not by itself request publishing.
- When explicitly asked to commit, push and create a PR, complete those
  steps without asking for the same permission again.
- Stage only files or hunks belonging to the task.
- Use concise English commit messages, for example:
  feat: add terminal chat (#12).
- Push the issue branch; never push directly to the default branch.
- Check for an existing PR for the branch before creating another.
- Target the intended base branch.
- Write the PR title and description in English.
- Describe the problem, resulting behavior and actual verification.
- Include Closes #<issue-number> when the issue is fully implemented
  and the PR targets the default branch.
- For partial work or another target branch, use Refs #<issue-number>
  and explain remaining work.
- Do not merge PRs, force-push or discard changes unless explicitly requested.
- End with the branch name, commit SHA, PR URL and verification result
  when those actions were performed.
