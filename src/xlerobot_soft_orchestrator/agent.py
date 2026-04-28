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

from __future__ import annotations

import asyncio
import logging
import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict, AsyncGenerator, Callable

from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

from .capability import TOOLS
from .config import Settings, configure_langsmith, get_settings, resolve_model

logger = logging.getLogger(__name__)

_FRAME_B64_KEYS = ("frame_b64", "base_frame_b64", "wrist_frame_b64")

# Token threshold before a compaction summary is triggered (rough char/4 estimate)
COMPACT_TOKEN_THRESHOLD = 25_000
# Number of most-recent messages to keep verbatim after compaction
COMPACT_KEEP_RECENT = 8


def _strip_frame_b64_from_tool_messages(messages: list) -> list:
    """Remove all *_frame_b64 keys from tool results before they re-enter the context window.

    Base64 image strings can be 300k+ tokens and are only needed by the UI renderer.
    """
    import json
    cleaned = []
    for msg in messages:
        if getattr(msg, "type", None) == "tool":
            try:
                payload = json.loads(msg.content)
                if any(k in payload for k in _FRAME_B64_KEYS):
                    for k in _FRAME_B64_KEYS:
                        if k in payload:
                            payload[k] = "<stripped>"
                    from langchain_core.messages import ToolMessage
                    msg = ToolMessage(
                        content=json.dumps(payload),
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
            except (json.JSONDecodeError, TypeError):
                pass
        cleaned.append(msg)
    return cleaned


# ---------------------------------------------------------------------------
# Protocols & shared types
# ---------------------------------------------------------------------------

class LLMLike(Protocol):
    def invoke(self, input_data: Any) -> Any: ...
    async def ainvoke(self, input_data: Any) -> Any: ...


class TraceEvent(TypedDict):
    step: int
    stage: str
    capability: str
    reasoning: str
    args: dict[str, Any]
    result_summary: str


class AgentState(TypedDict):
    directive: str
    # add_messages handles appending new messages and RemoveMessage-based compaction
    messages: Annotated[list, add_messages]
    trace: Annotated[list[TraceEvent], operator.add]
    step: int
    done: bool
    final_response: str


@dataclass
class AgentRunResult:
    final_response: str
    trace: list[TraceEvent]
    steps_taken: int
    graph_mermaid: str


# ---------------------------------------------------------------------------
# System prompt  (stable — capability docs live in capabilities.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a robot manipulation agent controlling an XLErobot arm.
Your job is to translate a high-level human directive into a sequence of robot
capability calls that accomplish the task.

=== HOW TO WORK ===
You operate in a ReAct loop: Reason → Act → Observe → Reason → ...

Before calling ANY capability you must:
1. Call list_capabilities to see what is available.
2. Call read_capability("<id>") for each capability you plan to use.
3. Then call run_capability("<id>", {args}) to execute it.

=== RESPONSE FORMAT ===
You can call tools to perform actions.
When you need to take an action, call the appropriate tool.
Before calling any tool, include a brief explanation of your reasoning in plain text. This must be included in the message content.
Do not output JSON. Use tool calls instead.

The "args" field:
  - For list_capabilities: {}
  - For read_capability: {"capability_id": "<id>"}
  - For run_capability:   {"capability_id": "<id>", "params": {<capability args>}}
  - If your model uses "kwargs" or "params" wrappers for tool inputs, that is
    also accepted and will be normalized.

When the task is complete, stop calling tools and provide a concise final
summary in plain text describing success or failure.

=== CAMERA RULES ===
You have two distinct camera views:
  - base camera  → wide-angle scene overview (object presence, workspace state)
  - wrist camera → close-up end-effector view (grasp quality, held object)

Use the correct camera for the task:
  - observe_with_base_camera:  scene understanding, "what's on the table?"
  - observe_with_wrist_camera: grasp confirmation, "what's in the gripper?"
  - observe_with_both_cameras: when you need full context simultaneously
  - yolo_base_camera:          locate objects on the table by bounding box
  - yolo_wrist_camera:         detect objects in close range or verify grip

Always pass a specific question when calling an observe capability.
Do not rely on frame images in your reasoning — the VLM description is your observation.

=== VLA POLICY EXECUTION ===
To run a manipulation policy:
1. Call start_vla_policy with the appropriate policy_id.
   This call BLOCKS until the model is loaded and the robot is already executing —
   no extra wait needed for model download.
2. The SUCCESS response includes a `typical_wait_time` field (in seconds).
   Use that value as your first wait duration — do NOT default to 30 seconds.
3. Call observe_with_base_camera or observe_with_wrist_camera to check success.
4. If not successful, call wait again (using typical_wait_time / 2 as a reasonable
   re-check interval) and re-observe. Repeat until success or max attempts.
5. Call stop_vla_policy when the task is confirmed complete.

Never call start_vla_policy twice in a row — a policy is already running if SUCCESS was returned.

=== SAFETY RULES ===
- Always observe the scene before triggering any policy.
- Never invent object positions — use values from yolo_base_camera results.
- If a capability returns an error, read its doc again and correct your args.
- Do not loop more than {{MAX_ITERATIONS}} times.
"""

tool_node = ToolNode(TOOLS)


def _extract_text_content(content: Any) -> str:
    """Normalize AI message content into plain text for final summaries."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(p for p in parts if p).strip()
    return ""


def _extract_done_summary_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Back-compat shim: if model emits run_capability(DONE), terminate gracefully."""
    for call in tool_calls:
        if call.get("name") != "run_capability_tool":
            continue
        call_args = call.get("args") or {}
        cap_id = str(call_args.get("capability_id", "")).strip().upper()
        if cap_id != "DONE":
            continue
        payload = call_args.get("args") or call_args.get("kwargs") or {}
        if isinstance(payload, dict) and isinstance(payload.get("summary"), str):
            return payload["summary"].strip()
        if isinstance(call_args.get("summary"), str):
            return call_args["summary"].strip()
        return "Task completed."
    return ""


def should_continue_factory(max_iterations: int) -> Callable[[AgentState], str]:
    def should_continue(state: AgentState) -> str:
        if state.get("done"):
            return "END"
        if state.get("step", 0) >= max_iterations:
            return "END"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "END"

    return should_continue


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------

def _estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "") or block.get("thinking", "")))
                else:
                    total += len(str(block))
    return total // 4


async def _compact_if_needed(messages: list, summary_llm: Any) -> list | None:
    """Return a list of add_messages-compatible updates (RemoveMessage + summary) or None."""
    if _estimate_tokens(messages) <= COMPACT_TOKEN_THRESHOLD:
        return None
    if len(messages) <= COMPACT_KEEP_RECENT:
        return None

    to_remove = messages[:-COMPACT_KEEP_RECENT]

    lines: list[str] = []
    for m in to_remove:
        role = getattr(m, "type", "message")
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in content
            )
        lines.append(f"[{role}] {str(content)[:400]}")

    summary_response = await summary_llm.ainvoke([
        SystemMessage(
            content=(
                "Summarize this robot manipulation conversation concisely (3-5 sentences): "
                "what tasks were requested, what actions were taken, what was observed, "
                "and what the outcomes were."
            )
        ),
        HumanMessage(content="\n".join(lines)),
    ])
    summary_text = _extract_text_content(summary_response.content)

    removes = [RemoveMessage(id=m.id) for m in to_remove if getattr(m, "id", None)]
    summary_msg = SystemMessage(content=f"[Prior Conversation Summary]\n{summary_text}")
    return removes + [summary_msg]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def reason_node(
    state: AgentState,
    llm: LLMLike,
    summary_llm: LLMLike,
    max_iterations: int,
) -> dict[str, Any]:
    # Compact conversation history if it exceeds the token threshold
    compact_updates = await _compact_if_needed(state["messages"], summary_llm)
    if compact_updates:
        ids_to_remove = {u.id for u in compact_updates if isinstance(u, RemoveMessage)}
        summaries = [u for u in compact_updates if not isinstance(u, RemoveMessage)]
        messages = [
            m for m in state["messages"]
            if getattr(m, "id", None) not in ids_to_remove
        ]
        messages = summaries + messages
        logger.info("Compacted conversation: removed %d messages, added summary.", len(ids_to_remove))
    else:
        messages = state["messages"]
        compact_updates = []

    system = SYSTEM_PROMPT.replace("{{MAX_ITERATIONS}}", str(max_iterations))
    clean_messages = _strip_frame_b64_from_tool_messages(messages)
    llm_messages = [SystemMessage(content=system)] + clean_messages

    response = await llm.ainvoke(llm_messages)

    tool_calls = getattr(response, "tool_calls", [])

    # Extract thinking/reasoning from the response.
    # - Ollama (reasoning=True): lands in additional_kwargs["reasoning_content"]
    # - Vertex structured thinking: lands as a {"type": "thinking"} block in content list
    reasoning = ""
    additional = getattr(response, "additional_kwargs", {}) or {}
    if additional.get("reasoning_content"):
        reasoning = additional["reasoning_content"]
    elif isinstance(response.content, list):
        parts: list[str] = []
        for b in response.content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "thinking" and isinstance(b.get("thinking"), str):
                parts.append(b["thinking"])
            elif b.get("type") == "reasoning" and isinstance(b.get("reasoning"), str):
                parts.append(b["reasoning"])
            elif b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
        reasoning = "\n\n".join(filter(None, parts))
    elif isinstance(response.content, str):
        reasoning = str(response.content)

    if not reasoning:
        response_meta = getattr(response, "response_metadata", {}) or {}
        usage_meta = response_meta.get("usage_metadata", {}) if isinstance(response_meta, dict) else {}
        thoughts_tokens = usage_meta.get("thoughts_token_count") if isinstance(usage_meta, dict) else 0

        usage = getattr(response, "usage_metadata", {}) or {}
        output_details = usage.get("output_token_details", {}) if isinstance(usage, dict) else {}
        reasoning_tokens = output_details.get("reasoning") if isinstance(output_details, dict) else 0

        has_vertex_signature = bool(additional.get("__gemini_function_call_thought_signatures__"))
        token_count = reasoning_tokens or thoughts_tokens
        if has_vertex_signature or token_count:
            token_note = f" ({token_count} reasoning tokens)" if token_count else ""
            reasoning = (
                "_Internal model thinking was used"
                f"{token_note}, but this provider does not expose plaintext "
                "thought content for this step._"
            )

    done_from_done_tool = _extract_done_summary_from_tool_calls(tool_calls)
    final_response = ""
    done = False

    if not tool_calls:
        final_response = _extract_text_content(getattr(response, "content", ""))
        done = True

    if done_from_done_tool:
        final_response = done_from_done_tool
        done = True

    next_step = state["step"] + 1
    if next_step >= max_iterations and not done:
        final_response = (
            "Stopped after reaching max iterations before a clear completion "
            "summary was produced."
        )
        done = True

    trace_event = TraceEvent(
        step=state["step"] + 1,
        stage="reason",
        capability=tool_calls[0]["name"] if tool_calls else "final_answer",
        reasoning=reasoning,
        args=tool_calls[0]["args"] if tool_calls else {},
        result_summary=final_response if final_response else "",
    )

    # Combine compaction updates with the new AI response message
    new_msgs = [] if done_from_done_tool else [response]
    messages_update = compact_updates + new_msgs

    return {
        "messages": messages_update,
        "trace": [trace_event],
        "step": next_step,
        "done": done,
        "final_response": final_response,
    }


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def get_vertex_llm(settings: Settings, model_name: str) -> ChatVertexAI:
    return ChatVertexAI(
        model_name=model_name or settings.vertex_model,
        project=settings.vertex_project_id,
        location=settings.vertex_location,
        temperature=0,
        include_thoughts=True,
    )


def get_ollama_llm(settings: Settings, model_name: str) -> ChatOllama:
    return ChatOllama(
        model=model_name or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        reasoning=True,
    )


def get_openai_llm(settings: Settings, model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or settings.openai_model,
        api_key=settings.openai_api_key or None,
        temperature=0,
    )


def get_anthropic_llm(settings: Settings, model_name: str) -> ChatAnthropic:
    kwargs: dict = dict(
        model=model_name or settings.anthropic_model,
        api_key=settings.anthropic_api_key or None,
    )
    if settings.anthropic_thinking_budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": settings.anthropic_thinking_budget}
        kwargs["temperature"] = 1
    else:
        kwargs["temperature"] = 0
    return ChatAnthropic(**kwargs)


_PROVIDER_FACTORIES = {
    "vertex": get_vertex_llm,
    "ollama": get_ollama_llm,
    "openai": get_openai_llm,
    "anthropic": get_anthropic_llm,
}


def get_llm() -> tuple[LLMLike, LLMLike]:
    """Return (tool_bound_llm, raw_llm).

    raw_llm is used for compaction summaries — it does not have robot tools bound,
    so it cannot accidentally call capabilities during summarisation.
    """
    settings = get_settings()
    configure_langsmith(settings)

    provider = settings.llm_provider.strip().lower()
    model_name = resolve_model(settings)

    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        supported = ", ".join(sorted(_PROVIDER_FACTORIES))
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Choose one of: {supported}.")

    raw = factory(settings, model_name)
    bound = raw.bind_tools(TOOLS)
    return bound, raw


# ---------------------------------------------------------------------------
# Checkpointer — module-level singleton, lazy-initialised
# ---------------------------------------------------------------------------


async def _get_checkpointer(conn_string: str) -> Any:
    """Return an AsyncPostgresSaver, or None if the DB is unreachable.
    
    Note: We do NOT cache the checkpointer because asyncio locks are bound to 
    their event loop. In Streamlit and other frameworks that create new event loops
    per request, reusing a cached checkpointer causes "bound to a different event loop" errors.
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            conn_string,
            max_size=5,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        logger.info("Memory DB connected: %s", conn_string.split("@")[-1])
        return saver

    except Exception as exc:
        logger.warning("Memory DB unavailable (%s) — running without persistence.", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    llm: LLMLike,
    summary_llm: LLMLike,
    max_iterations: int = 10,
    checkpointer: Any = None,
):
    async def _reason_node(s: AgentState) -> dict[str, Any]:
        return await reason_node(s, llm, summary_llm, max_iterations)
    
    workflow = StateGraph(AgentState)
    workflow.add_node(
        "reason",
        _reason_node,
    )
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "reason")
    workflow.add_edge("tools", "reason")
    workflow.add_conditional_edges(
        "reason",
        should_continue_factory(max_iterations),
        {"tools": "tools", "END": END},
    )
    return workflow.compile(checkpointer=checkpointer)


async def astream_directive(
    directive: str, thread_id: str | None = None
) -> AsyncGenerator[dict[str, Any], None]:
    settings = get_settings()
    llm, summary_llm = get_llm()

    checkpointer = await _get_checkpointer(settings.get_memory_db_url())
    graph = build_graph(llm, summary_llm, settings.max_iterations, checkpointer)

    run_thread_id = thread_id or str(uuid.uuid4())

    # Only the new user message is passed as input; the checkpointer merges it
    # with the existing conversation history for this thread.
    input_state = {
        "directive": directive,
        "messages": [HumanMessage(content=directive)],
        "trace": [],
        "step": 0,
        "done": False,
        "final_response": "",
    }
    config = {"configurable": {"thread_id": run_thread_id}}

    async for update in graph.astream(input_state, config=config):
        yield update


def run_directive(directive: str, thread_id: str | None = None) -> AgentRunResult:
    return asyncio.run(_run_async_wrapper(directive, thread_id))


async def _run_async_wrapper(directive: str, thread_id: str | None = None) -> AgentRunResult:
    settings = get_settings()
    llm, summary_llm = get_llm()

    checkpointer = await _get_checkpointer(settings.get_memory_db_url())
    graph = build_graph(llm, summary_llm, settings.max_iterations, checkpointer)

    run_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_thread_id}}

    res = await graph.ainvoke(
        {
            "directive": directive,
            "messages": [HumanMessage(content=directive)],
            "trace": [],
            "step": 0,
            "done": False,
            "final_response": "",
        },
        config=config,
    )
    return AgentRunResult(
        final_response=res.get("final_response", ""),
        trace=res.get("trace", []),
        steps_taken=res.get("step", 0),
        graph_mermaid="",
    )
