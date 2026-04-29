"""LangGraph ReAct agent for robot manipulation tasks.

Architecture:
  User directive → reason → act → observe → reason → ... → respond

The agent runs a ReAct (Reason + Act) loop:
  1. reason   — LLM thinks about what to do next, chooses a capability
  2. act      — execute the chosen capability via the registry
  3. The result feeds back into reason for the next step

The loop terminates when the LLM decides the task is done or max_iterations
is reached.  Every step is recorded in the trace for the UI.

Memory:
  Conversation history is persisted per thread_id in a PostgreSQL database via
  LangGraph's AsyncPostgresSaver checkpoint system.  If the DB is unreachable
  the agent falls back to stateless (single-turn) mode with a warning.

  Compaction: when the estimated token count of accumulated messages exceeds
  COMPACT_TOKEN_THRESHOLD, older messages are summarised and replaced so the
  LLM context stays manageable.
"""

from .graph import run_directive, astream_directive
__all__ = ["run_directive", "astream_directive"]