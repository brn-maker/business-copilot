"""
Model Router: Intelligently routes requests to appropriate LLM based on task type.

Cost Optimization Strategy:
- Free-tier OpenRouter models only (IDs ending in :free, or openrouter/free)
- Smaller free models for data processing & analysis
- Larger free models for final synthesis & recommendations
- openrouter/free auto-router as default (survives free-model retirements)
"""

import os
from typing import Optional
from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI


class ModelRouter:
    """Routes LLM requests to cost-optimized free models on OpenRouter."""

    # Only :free models (or openrouter/free). Paid models will fail on $0 balance.
    # Free catalog changes often — prefer "auto" / openrouter/free when unsure.
    # See: https://openrouter.ai/models?max_price=0
    FAST_MODELS = {
        "auto": "openrouter/free",
        "llama": "meta-llama/llama-3.2-3b-instruct:free",
        "gemma": "google/gemma-4-26b-a4b-it:free",
        "gemini": "google/gemma-4-26b-a4b-it:free",  # alias for older config
        "nemotron": "nvidia/nemotron-nano-9b-v2:free",
        "gpt_oss": "openai/gpt-oss-20b:free",
    }

    SYNTHESIS_MODELS = {
        "auto": "openrouter/free",
        "llama_large": "meta-llama/llama-3.3-70b-instruct:free",
        "gemma_pro": "google/gemma-4-31b-it:free",
        "gemini_pro": "google/gemma-4-31b-it:free",  # alias for older config
        "qwen": "qwen/qwen3-coder:free",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_openrouter: bool = True,
        default_fast_model: str = "auto",
        default_synthesis_model: str = "auto",
    ):
        """
        Initialize ModelRouter.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            use_openrouter: Whether to use OpenRouter (default: True)
            default_fast_model: Default fast model key from FAST_MODELS
            default_synthesis_model: Default synthesis model key from SYNTHESIS_MODELS
                Prefer "auto" (openrouter/free) on free tier — specific free
                models are often rate-limited or temporarily offline.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.use_openrouter = use_openrouter
        self.default_fast_model = default_fast_model
        self.default_synthesis_model = default_synthesis_model

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
            )

        self._fast_model_cache = None
        self._synthesis_model_cache = None

    def get_fast_model(self, model_key: Optional[str] = None):
        """Get a fast free model for quick analysis tasks."""
        key = model_key or self.default_fast_model
        model_id = self.FAST_MODELS.get(key, self.FAST_MODELS["auto"])

        if self.use_openrouter:
            return ChatOpenRouter(
                model=model_id,
                openrouter_api_key=self.api_key,
                temperature=0.3,  # Lower temp for consistency in analysis
                max_tokens=4000,  # Prevent credit limit errors on free tier
            )
        return ChatOpenAI(
            model=model_id,
            api_key=self.api_key,
            temperature=0.3,
            max_tokens=4000,
        )

    def get_synthesis_model(self, model_key: Optional[str] = None):
        """Get a more capable free model for final synthesis and recommendations."""
        key = model_key or self.default_synthesis_model
        model_id = self.SYNTHESIS_MODELS.get(key, self.SYNTHESIS_MODELS["auto"])

        if self.use_openrouter:
            return ChatOpenRouter(
                model=model_id,
                openrouter_api_key=self.api_key,
                temperature=0.7,  # Higher temp for creative recommendations
                max_tokens=4000,  # Prevent credit limit errors on free tier
            )
        return ChatOpenAI(
            model=model_id,
            api_key=self.api_key,
            temperature=0.7,
            max_tokens=4000,
        )

    def get_model_for_task(self, task_type: str):
        """
        Get the appropriate model for a specific task type.

        Args:
            task_type: One of 'analysis', 'forecasting', 'synthesis', 'recommendations'
        """
        if task_type in ["analysis", "forecasting", "data_processing"]:
            return self.get_fast_model()
        elif task_type in ["synthesis", "recommendations"]:
            return self.get_synthesis_model()
        else:
            # Default to fast model
            return self.get_fast_model()


def create_model_router() -> ModelRouter:
    """Factory function to create a ModelRouter with environment config."""
    return ModelRouter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        use_openrouter=True,
        default_fast_model=os.getenv("FAST_MODEL_KEY", "auto"),
        default_synthesis_model=os.getenv("SYNTHESIS_MODEL_KEY", "auto"),
    )
