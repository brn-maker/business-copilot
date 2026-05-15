"""
Model Router: Intelligently routes requests to appropriate LLM based on task type.

Cost Optimization Strategy:
- Fast, cheap models (Llama 3.1 70B, Claude Haiku) for data processing & analysis
- Larger models only for final synthesis & recommendations
- Supports OpenRouter with fallback to OpenAI-compatible endpoints
"""

import os
from typing import Optional
from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI


class ModelRouter:
    """Routes LLM requests to cost-optimized models."""
    
    # Model configurations (OpenRouter model IDs)
    FAST_MODELS = {
        "llama": "meta-llama/llama-3.1-8b-instruct",
        "gemini": "google/gemma-2-9b-it",
        "mistral": "mistralai/mistral-nemo",
    }
    
    SYNTHESIS_MODELS = {
        "gemini_pro": "google/gemma-2-9b-it", # Capable model for synthesis
        "llama_large": "meta-llama/llama-3.1-70b-instruct",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_openrouter: bool = True,
        default_fast_model: str = "gemini",
        default_synthesis_model: str = "llama_large",
    ):
        """
        Initialize ModelRouter.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            use_openrouter: Whether to use OpenRouter (default: True)
            default_fast_model: Default fast model key from FAST_MODELS
            default_synthesis_model: Default synthesis model key from SYNTHESIS_MODELS
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
        """Get a fast, cheap model for quick analysis tasks."""
        key = model_key or self.default_fast_model
        model_id = self.FAST_MODELS.get(key, self.FAST_MODELS["gemini"])
        
        if self.use_openrouter:
            return ChatOpenRouter(
                model=model_id,
                openrouter_api_key=self.api_key,
                temperature=0.3,  # Lower temp for consistency in analysis
            )
        return ChatOpenAI(
            model=model_id,
            api_key=self.api_key,
            temperature=0.3,
        )
    
    def get_synthesis_model(self, model_key: Optional[str] = None):
        """Get a more capable model for final synthesis and recommendations."""
        key = model_key or self.default_synthesis_model
        model_id = self.SYNTHESIS_MODELS.get(key, self.SYNTHESIS_MODELS["llama_large"])
        
        if self.use_openrouter:
            return ChatOpenRouter(
                model=model_id,
                openrouter_api_key=self.api_key,
                temperature=0.7,  # Higher temp for creative recommendations
            )
        return ChatOpenAI(
            model=model_id,
            api_key=self.api_key,
            temperature=0.7,
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
        default_fast_model="gemini",
        default_synthesis_model="llama_large",
    )
