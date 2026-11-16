# Project

Build a small personal AI agent running locally on a computer.
The user types messages in the terminal and receives text responses.
Later, the application may run on Raspberry Pi 5.

## Stack

Python 3.12, venv, pip.

PydanticAI with pydantic-ai-slim[anthropic].

Claude Haiku through Anthropic API.

python-dotenv for local configuration.

SQLite through Python's built-in sqlite3 when persistence is added.

Configure ANTHROPIC_API_KEY and ANTHROPIC_MODEL through environment variables.

Keep secrets in .env; provide .env.example with placeholders.

## Implementation order

Implement only the task requested in the current message:

1. Project environment and minimal entry point.
2. One agent request and printed response.
3. Interactive terminal chat with /exit.
4. In-memory conversation history.
5. One simple tool: current date and time.
6. SQLite history persistence and restore after restart.
7. Session commands: /new, /sessions, /resume.
8. Persistent memory tools: save, search, update, delete.
9. Conversation summaries to limit context size.

Do not implement later steps ahead of time.

## Architecture

Start with a small structure; add modules when needed.

Separate terminal input/output from agent logic.

Keep database access separate when persistence is introduced.

Use PydanticAI's supported message serialization.

Preserve tool calls and their results in conversation history.

Distinguish session history from long-term user memories.

Initially save long-term memories only when explicitly requested.

Keep configuration and hardware-independent logic portable to Raspberry Pi.

## Scope

MVP has one local user.

Voice, camera, Hailo, motors and Arduino are future work.

Add web UI, servers, Docker, MCP, vector databases or extra agents only when requested.

Prefer simple functions and minimal dependencies.

## Working rules

Read relevant existing files before making changes.

Make the smallest complete change for the current task.

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
