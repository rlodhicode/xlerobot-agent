import logging
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

logger = logging.getLogger(__name__)

# Token threshold before a compaction summary is triggered (rough char/4 estimate)
COMPACT_TOKEN_THRESHOLD = 25_000
# Number of most-recent messages to keep verbatim after compaction
COMPACT_KEEP_RECENT = 8

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

async def compact_if_needed(messages: list, summary_llm: Any) -> list | None:
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

async def get_checkpointer(conn_string: str) -> Any:
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
