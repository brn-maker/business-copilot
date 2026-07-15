"""
Model Router: Routes requests to free OpenRouter models with fallbacks.

Free models and the openrouter/free router often go offline or point at
retired IDs (e.g. google/gemma-2-9b-it:free). We pin concrete :free model
IDs and try several in order until one succeeds.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Union

from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI


# Concrete free models only — do NOT use openrouter/free as primary.
# It can resolve to retired IDs with "No endpoints found".
# Catalog: https://openrouter.ai/models?max_price=0
FREE_MODEL_CHAIN: List[str] = [
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
]


class ModelRouter:
    """Routes LLM requests to free OpenRouter models with automatic fallback."""

    FAST_MODELS = {
        "auto": FREE_MODEL_CHAIN[0],
        "gemma": "google/gemma-4-26b-a4b-it:free",
        "gemini": "google/gemma-4-26b-a4b-it:free",  # alias
        "gpt_oss": "openai/gpt-oss-20b:free",
        "cohere": "cohere/north-mini-code:free",
        "nemotron": "nvidia/nemotron-nano-9b-v2:free",
        "llama": "meta-llama/llama-3.2-3b-instruct:free",
    }

    SYNTHESIS_MODELS = {
        "auto": FREE_MODEL_CHAIN[0],
        "gemma_pro": "google/gemma-4-26b-a4b-it:free",
        "gemini_pro": "google/gemma-4-26b-a4b-it:free",  # alias
        "gpt_oss": "openai/gpt-oss-20b:free",
        "llama_large": "meta-llama/llama-3.3-70b-instruct:free",
        "qwen": "qwen/qwen3-next-80b-a3b-instruct:free",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_openrouter: bool = True,
        default_fast_model: str = "auto",
        default_synthesis_model: str = "auto",
        free_model_chain: Optional[Sequence[str]] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.use_openrouter = use_openrouter
        self.default_fast_model = default_fast_model
        self.default_synthesis_model = default_synthesis_model
        self.free_model_chain = list(free_model_chain or FREE_MODEL_CHAIN)

        # Optional env override: comma-separated model IDs
        env_chain = os.getenv("OPENROUTER_FREE_MODEL_CHAIN", "").strip()
        if env_chain:
            self.free_model_chain = [m.strip() for m in env_chain.split(",") if m.strip()]

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or Streamlit secret."
            )

    def _chat_model(
        self,
        model_id: str,
        *,
        temperature: float,
        max_tokens: int,
    ):
        if self.use_openrouter:
            return ChatOpenRouter(
                model=model_id,
                openrouter_api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                app_url=os.getenv(
                    "OPENROUTER_APP_URL",
                    "https://business-copilot.streamlit.app",
                ),
                app_title=os.getenv(
                    "OPENROUTER_APP_TITLE",
                    "Coffee Analytics Copilot",
                ),
            )
        return ChatOpenAI(
            model=model_id,
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_fast_model(self, model_key: Optional[str] = None):
        """Get a free model for quick analysis (single model, no fallback)."""
        key = model_key or self.default_fast_model
        model_id = self.FAST_MODELS.get(key, self.free_model_chain[0])
        return self._chat_model(model_id, temperature=0.3, max_tokens=2000)

    def get_synthesis_model(self, model_key: Optional[str] = None):
        """Get a free model for synthesis (single model, no fallback)."""
        key = model_key or self.default_synthesis_model
        model_id = self.SYNTHESIS_MODELS.get(key, self.free_model_chain[0])
        return self._chat_model(model_id, temperature=0.7, max_tokens=2000)

    def get_model_for_task(self, task_type: str):
        if task_type in ["analysis", "forecasting", "data_processing"]:
            return self.get_fast_model()
        if task_type in ["synthesis", "recommendations"]:
            return self.get_synthesis_model()
        return self.get_fast_model()

    @staticmethod
    def extract_text(response: Any) -> str:
        """Normalize message content / reasoning-only free models to text."""
        if response is None:
            return ""
        content = getattr(response, "content", response)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif "text" in block:
                        parts.append(str(block.get("text", "")))
            text = "".join(parts).strip()
            if text:
                return text
        # Some free models return only reasoning with empty content
        extra = getattr(response, "additional_kwargs", None) or {}
        reasoning = extra.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            # Prefer not to show raw chain-of-thought; last line is often the answer
            lines = [ln.strip() for ln in reasoning.strip().splitlines() if ln.strip()]
            if lines:
                return lines[-1]
        return content if isinstance(content, str) else (str(content) if content else "")

    def invoke_with_fallback(
        self,
        messages: Union[str, Sequence[Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model_chain: Optional[Sequence[str]] = None,
    ) -> str:
        """
        Try free models in order until one returns non-empty text.

        Skips retired models (No endpoints found), rate limits, and empty replies.
        """
        chain = list(model_chain or self.free_model_chain)
        errors: List[str] = []

        for model_id in chain:
            try:
                model = self._chat_model(
                    model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = model.invoke(messages)
                text = self.extract_text(response)
                if text and text.strip():
                    return text.strip()
                errors.append(f"{model_id}: empty response")
            except Exception as e:
                msg = str(e)
                # Continue on common free-tier failures
                errors.append(f"{model_id}: {msg}")
                continue

        detail = "\n".join(f"- {e}" for e in errors[:8])
        raise RuntimeError(
            "All free OpenRouter models failed or returned empty responses.\n"
            f"Tried: {', '.join(chain)}\n"
            f"Errors:\n{detail}\n\n"
            "Free models are often rate-limited (~50 req/day without credits). "
            "Wait and retry, or add a small credit balance at https://openrouter.ai/settings/credits"
        )


def create_model_router() -> ModelRouter:
    """Factory function to create a ModelRouter with environment config."""
    return ModelRouter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        use_openrouter=True,
        default_fast_model=os.getenv("FAST_MODEL_KEY", "auto"),
        default_synthesis_model=os.getenv("SYNTHESIS_MODEL_KEY", "auto"),
    )
