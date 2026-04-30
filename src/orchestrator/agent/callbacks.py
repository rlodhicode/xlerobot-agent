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
        self.total_cost = 0.0
        self.call_count = 0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM call ends."""
        try:
            # Try to get usage metadata from response
            if hasattr(response, "llm_output"):
                output = response.llm_output or {}
                usage = output.get("usage_metadata", {}) or {}
                if usage:
                    self.total_input_tokens += usage.get("input_tokens", 0)
                    self.total_output_tokens += usage.get("output_tokens", 0)
                    self.call_count += 1

            # Also try direct usage_metadata attribute
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata or {}
                if usage:
                    self.total_input_tokens += usage.get("input_tokens", 0)
                    self.total_output_tokens += usage.get("output_tokens", 0)
                    self.call_count += 1

            # Try generations (for newer versions)
            if hasattr(response, "generations") and response.generations:
                for generation_list in response.generations:
                    for generation in generation_list:
                        if hasattr(generation, "usage_metadata"):
                            usage = generation.usage_metadata or {}
                            if usage:
                                self.total_input_tokens += usage.get("input_tokens", 0)
                                self.total_output_tokens += usage.get("output_tokens", 0)
                                self.call_count += 1

        except Exception as e:
            logger.debug(f"Error extracting token metadata: {e}")

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
        self.total_cost = 0.0
        self.call_count = 0
