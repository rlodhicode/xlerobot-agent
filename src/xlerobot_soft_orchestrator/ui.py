"""Streamlit UI for the XLErobot ReAct orchestrator."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
import traceback
import uuid
from pathlib import Path

import streamlit as st

# Setup Pathing
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xlerobot_soft_orchestrator.agent import astream_directive

# ---------------------------------------------------------------------------
# Rendering Helpers
# ---------------------------------------------------------------------------

def _parse_tool_result(result_str: str) -> dict | None:
    """Try to parse a tool result string as JSON. Returns dict or None."""
    try:
        return json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return None


def render_tool_status(
    step_num: int,
    cap_id: str,
    args: dict,
    result_summary: str,
    is_running: bool = False,
):
    """Renders a tool call as a status card.

    If the result JSON contains a 'frame_b64' key with a non-empty base64 PNG,
    the image is displayed inline inside the card.
    """
    label = f"🛠️ Step {step_num}: {cap_id}"
    state = "running" if is_running else "complete"

    status = st.status(label, state=state)
    with status:
        st.write("**Input (Args):**")
        st.json(args)

        if not is_running:
            st.write("**Observation Result:**")

            parsed = _parse_tool_result(result_summary)

            if parsed is not None:
                frame_b64: str       = parsed.pop("frame_b64",       "") or ""
                base_frame_b64: str  = parsed.pop("base_frame_b64",  "") or ""
                wrist_frame_b64: str = parsed.pop("wrist_frame_b64", "") or ""

                def _show_image(b64: str, caption: str) -> None:
                    try:
                        st.image(base64.b64decode(b64), caption=caption, use_container_width=True)
                    except Exception as exc:
                        st.warning(f"Could not decode {caption}: {exc}")

                if base_frame_b64 and wrist_frame_b64:
                    col_base, col_wrist = st.columns(2)
                    with col_base:
                        _show_image(base_frame_b64, "Base camera")
                    with col_wrist:
                        _show_image(wrist_frame_b64, "Wrist camera")
                elif base_frame_b64:
                    _show_image(base_frame_b64, "Base camera")
                elif wrist_frame_b64:
                    _show_image(wrist_frame_b64, "Wrist camera")
                elif frame_b64:
                    _show_image(frame_b64, f"Camera frame — {parsed.get('camera_info', '')}")

                if parsed.get("error"):
                    st.error(f"⚠️ {parsed['error']}")

                detected = parsed.get("detected")
                if detected:
                    st.write(f"**Detected objects ({parsed.get('count', len(detected))}):**")
                    rows = []
                    for obj in detected:
                        rows.append({
                            "ID": obj.get("id", ""),
                            "Label": obj.get("label", ""),
                            "X (m)": round(obj.get("x", 0), 4),
                            "Y (m)": round(obj.get("y", 0), 4),
                            "Z (m)": round(obj.get("z") or 0, 4),
                            "Conf": round(obj.get("confidence", 0), 3),
                        })
                    st.dataframe(rows, use_container_width=True)

                _exclude = {"detected", "count", "error", "frame_b64", "base_frame_b64", "wrist_frame_b64"}
                meta = {
                    k: v for k, v in parsed.items()
                    if k not in _exclude
                    and v not in (None, "", [], {})
                }
                if meta:
                    with st.expander("Details", expanded=False):
                        st.json(meta)

            else:
                if "ERROR" in result_summary.upper():
                    st.error(result_summary)
                else:
                    st.code(result_summary, language="text")

    return status


def render_reasoning(reasoning: str, expanded: bool = False):
    """Renders the 'Thinking' expander block."""
    if reasoning and reasoning.strip():
        with st.expander("💭 Thinking", expanded=expanded):
            st.markdown(reasoning)


def extract_reasoning_from_message_content(content: object) -> str:
    """Extract model reasoning from structured content blocks."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking" and isinstance(block.get("thinking"), str):
            parts.append(block["thinking"])
        elif block_type == "reasoning" and isinstance(block.get("reasoning"), str):
            parts.append(block["reasoning"])
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def render_past_turn(turn: dict, expand_thinking: bool = False) -> None:
    """Render a completed conversation turn from history."""
    st.chat_message("human").write(turn["directive"])
    with st.chat_message("ai"):
        for i, step in enumerate(turn["steps"]):
            if step.get("reasoning"):
                render_reasoning(step["reasoning"], expanded=False)
            render_tool_status(i + 1, step["cap_id"], step["args"], step["result_summary"])
        if turn["final_response"]:
            st.info(f"**Final Summary:** {turn['final_response']}")


# ---------------------------------------------------------------------------
# Thread state — thread_id lives in the URL (?thread=<uuid>)
# Conversation history lives in session_state (ephemeral, within browser session)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="XLErobot-Pro Orchestrator", layout="wide", page_icon="🤖")

# Sync thread_id between URL and session_state.
# URL is the source of truth so the thread is bookmarkable / shareable.
_url_thread = st.query_params.get("thread")
if _url_thread:
    if st.session_state.get("thread_id") != _url_thread:
        # URL changed (e.g. user navigated to a shared link) — reset conversation.
        st.session_state.thread_id = _url_thread
        st.session_state.conversation = []
else:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    st.query_params["thread"] = st.session_state.thread_id

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    expand_thinking = st.toggle("Expand thinking by default", value=False)

    st.divider()
    st.subheader("Camera")
    st.caption(
        "Set `USE_REALSENSE=true` in your `.env` to use the Intel RealSense. "
        "Otherwise OpenCV camera index 0 is used.\n\n"
        "Set `OPENCV_CAMERA_INDEX=<n>` to pick a different USB camera."
    )

    st.divider()
    st.subheader("Thread")
    turn_count = len(st.session_state.conversation)
    st.caption(f"**ID:** `{st.session_state.thread_id}`")
    st.caption(f"{turn_count} turn{'s' if turn_count != 1 else ''}")
    if st.button("New Thread", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.thread_id = new_id
        st.session_state.conversation = []
        st.query_params["thread"] = new_id
        st.rerun()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("🤖 XLErobot-Pro Orchestrator")
st.caption("ReAct loop: Reason → Act → Observe → Reason")

async def main():
    # Render all past turns in this thread first
    for turn in st.session_state.conversation:
        render_past_turn(turn, expand_thinking=expand_thinking)

    directive = st.chat_input(
        placeholder='e.g. "Look at the workspace and tell me what you see."'
    )

    if directive:
        st.chat_message("human").write(directive)

        current_steps: list[dict] = []
        final_response = ""
        pending_reasoning = ""
        pending_args: dict = {}

        with st.chat_message("ai"):
            try:
                last_step_rendered = 0

                async for update in astream_directive(directive, thread_id=st.session_state.thread_id):
                    node_name = list(update.keys())[0]
                    node_data = update[node_name]

                    if node_name == "reason":
                        reasoning = ""
                        if "trace" in node_data:
                            event = node_data["trace"][-1]
                            reasoning = event.get("reasoning", "")

                        if not reasoning and "messages" in node_data:
                            for msg in node_data["messages"]:
                                content_obj = getattr(msg, "content", "")
                                reasoning = extract_reasoning_from_message_content(content_obj)
                                if reasoning:
                                    break
                                raw = str(content_obj or "")
                                m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
                                if m:
                                    reasoning = m.group(1).strip()
                                    break

                        pending_reasoning = reasoning
                        render_reasoning(reasoning, expanded=expand_thinking)

                        if "messages" in node_data:
                            for msg in node_data["messages"]:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for call in msg.tool_calls:
                                        pending_args = call.get("args", {})
                                        render_tool_status(
                                            step_num=last_step_rendered + 1,
                                            cap_id=call["name"],
                                            args=pending_args,
                                            result_summary="Executing capability...",
                                            is_running=True,
                                        )

                    elif node_name == "tools":
                        if "messages" in node_data:
                            for msg in node_data["messages"]:
                                if msg.type == "tool":
                                    render_tool_status(
                                        step_num=last_step_rendered + 1,
                                        cap_id=msg.name,
                                        args={},
                                        result_summary=msg.content,
                                        is_running=False,
                                    )
                                    current_steps.append({
                                        "cap_id": msg.name,
                                        "args": pending_args,
                                        "result_summary": msg.content,
                                        "reasoning": pending_reasoning,
                                    })
                                    pending_args = {}
                                    pending_reasoning = ""
                                    last_step_rendered += 1

                    if "final_response" in node_data and node_data["final_response"]:
                        final_response = node_data["final_response"]
                        st.info(f"**Final Summary:** {final_response}")

            except Exception as exc:
                st.error(f"Agent error during execution: {exc}")
                st.code(traceback.format_exc())

        st.session_state.conversation.append({
            "directive": directive,
            "steps": current_steps,
            "final_response": final_response,
        })


if __name__ == "__main__":
    asyncio.run(main())
