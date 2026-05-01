"""LangSmith callback handlers for aggregating token usage at thread level."""

from typing import Any, Dict, Optional, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import json
import logging

logger = logging.getLogger(__name__)


class ThreadTokenAggregator(BaseCallbackHandler):
    """Aggregates token usage across all LLM calls in a thread."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _ingest_usage(self, usage: dict) -> bool:
        """Try known usage dict shapes. Returns True if any tokens were found."""
        # OpenAI shape: {"prompt_tokens": N, "completion_tokens": N}
        # Anthropic/Vertex shape: {"input_tokens": N, "output_tokens": N}
        inp = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        out = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
        if inp or out:
            self.total_input_tokens += inp
            self.total_output_tokens += out
            self.call_count += 1
            return True
        return False

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            # Path 1: llm_output (common for older LangChain OpenAI wrapper)
            llm_output = getattr(response, "llm_output", {}) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage_metadata") or {}
            if usage and self._ingest_usage(usage):
                return

            # Path 2: iterate generations looking for usage_metadata on the message
            for gen_list in (getattr(response, "generations", None) or []):
                for gen in gen_list:
                    # ChatGeneration has a .message with usage_metadata
                    msg = getattr(gen, "message", None)
                    usage = getattr(msg, "usage_metadata", None) or {}
                    if usage and self._ingest_usage(usage):
                        return
                    # Some providers put it directly on the generation
                    usage = getattr(gen, "usage_metadata", None) or {}
                    if usage and self._ingest_usage(usage):
                        return

            # Path 3: response-level usage_metadata (Anthropic)
            usage = getattr(response, "usage_metadata", None) or {}
            if usage:
                self._ingest_usage(usage)

        except Exception as e:
            logger.debug("Error extracting token metadata: %s", e)

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated token usage."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": self.call_count,
        }

    def reset(self) -> None:
        """Reset aggregation."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
