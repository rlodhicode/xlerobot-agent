import asyncio
import logging
import uuid
import json
import os
from typing import Any, AsyncGenerator, Callable

from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langsmith import Client

from ..config import get_settings
from ..capabilities.registry import TOOLS
from .state import AgentState, AgentRunResult, TraceEvent, SYSTEM_PROMPT
from .memory import get_checkpointer, compact_if_needed, _extract_text_content
from .llm_factory import get_llm, LLMLike
from .callbacks import ThreadTokenAggregator

logger = logging.getLogger(__name__)

_FRAME_B64_KEYS = ("frame_b64", "base_frame_b64", "wrist_frame_b64")

def _strip_frame_b64_from_tool_messages(messages: list) -> list:
    """Remove all *_frame_b64 keys from tool results before they re-enter the context window.

    Base64 image strings can be 300k+ tokens and are only needed by the UI renderer.
    """
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


def _update_thread_metadata_in_langsmith(thread_id: str, token_aggregator: ThreadTokenAggregator) -> None:
    """Update thread metadata in LangSmith with aggregated token usage."""
    try:
        if not os.getenv("LANGCHAIN_TRACING_V2") or not os.getenv("LANGCHAIN_API_KEY"):
            logger.debug("LangSmith tracing not enabled, skipping thread metadata update")
            return

        # Get LangSmith client
        ls_client = Client()
        summary = token_aggregator.get_summary()
        if summary["total_tokens"] == 0:
            logger.debug("No tokens to report, skipping LangSmith update")
            return

        # list_runs requires at least one of: session, id, parent_run, trace, reference_example
        # Use session (project name) + thread_id tag filter instead
        project = os.getenv("LANGCHAIN_PROJECT", "xlerobot-pro")
        runs = list(ls_client.list_runs(
            project_name=project,
            filter=f'and(eq(thread_id, "{thread_id}"))',
        ))

        if not runs:
            logger.debug("No runs found for thread %s", thread_id)
            return

        # Find root run and attach token summary as feedback
        root_run = next((r for r in runs if r.parent_run_id is None), runs[0])
        ls_client.create_feedback(
            run_id=root_run.id,
            key="token_usage",
            score=summary["total_tokens"],
            value=summary,
            comment=f"input={summary['total_input_tokens']} output={summary['total_output_tokens']} calls={summary['call_count']}",
        )
        logger.info("Posted token usage to LangSmith run %s: %s", root_run.id, summary)

    except Exception as e:
        logger.warning("Failed to update thread metadata in LangSmith: %s", e)

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

async def reason_node(
    state: AgentState,
    llm: LLMLike,
    max_iterations: int,
) -> dict[str, Any]:
    messages = state["messages"]

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
            reasoning = f"_Internal model thinking was used{token_note}, but this provider does not expose plaintext thought content for this step._"

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
        final_response = "Stopped after reaching max iterations before a clear completion summary was produced."
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
    messages_update = [] if done_from_done_tool else [response]

    return {
        "messages": messages_update,
        "trace": [trace_event],
        "step": next_step,
        "done": done,
        "final_response": final_response,
    }

tool_node = ToolNode(TOOLS)

def build_graph(llm: LLMLike, max_iterations: int = 10, checkpointer: Any = None, callbacks: list = None):
    if callbacks is None:
        callbacks = []
        
    async def _reason_node(s: AgentState) -> dict[str, Any]:
        return await reason_node(s, llm, max_iterations)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("reason", _reason_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "reason")
    workflow.add_edge("tools", "reason")
    workflow.add_conditional_edges(
        "reason",
        should_continue_factory(max_iterations),
        {"tools": "tools", "END": END},
    )
    return workflow.compile(checkpointer=checkpointer)

async def astream_directive(directive: str, thread_id: str | None = None) -> AsyncGenerator[dict[str, Any], None]:
    settings = get_settings()
    
    # Create token aggregator callback
    token_aggregator = ThreadTokenAggregator()
    callbacks = [token_aggregator]
    
    llm = get_llm(callbacks=callbacks)
    checkpointer = await get_checkpointer(settings.get_memory_db_url())
    graph = build_graph(llm[0], settings.max_iterations, checkpointer, callbacks=callbacks)
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

    try:
        async for update in graph.astream(input_state, config=config):
            yield update
    finally:
        # Update thread metadata with aggregated token usage
        # Run this in a separate task to avoid blocking the stream
        try:
            _update_thread_metadata_in_langsmith(run_thread_id, token_aggregator)
        except Exception as e:
            logger.warning(f"Failed to update thread metadata: {e}")

def run_directive(directive: str, thread_id: str | None = None) -> AgentRunResult:
    return asyncio.run(_run_async_wrapper(directive, thread_id))

async def _run_async_wrapper(directive: str, thread_id: str | None = None) -> AgentRunResult:
    settings = get_settings()
    
    # Create token aggregator callback
    token_aggregator = ThreadTokenAggregator()
    callbacks = [token_aggregator]
    
    llm = get_llm(callbacks=callbacks)
    checkpointer = await get_checkpointer(settings.get_memory_db_url())
    graph = build_graph(llm[0], settings.max_iterations, checkpointer, callbacks=callbacks)
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
    
    # Update thread metadata with aggregated token usage
    try:
        _update_thread_metadata_in_langsmith(run_thread_id, token_aggregator)
    except Exception as e:
        logger.warning(f"Failed to update thread metadata: {e}")
    
    return AgentRunResult(
        final_response=res.get("final_response", ""),
        trace=res.get("trace", []),
        steps_taken=res.get("step", 0),
        graph_mermaid="",
    )
