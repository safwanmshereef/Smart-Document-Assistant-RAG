import os
from typing import Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# We import the base class for type hints
from langchain_core.language_models.chat_models import BaseChatModel

# Importing Google Generative AI wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Importing ChatOllama with fallback imports for compatibility
try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        from langchain_community.chat_models.ollama import ChatOllama

# Allowed local models configuration for local Ollama setup
VALID_OLLAMA_MODELS = {"qwen3.5:9b", "gemma4:e4b"}

def get_llm(provider: str, model_name: str, **kwargs: Any) -> BaseChatModel:
    """
    LLM Factory function to return a LangChain ChatModel instance.

    Supported Providers:
    - "google": Google Gemini Models (e.g., gemini-1.5-flash, gemini-1.5-pro).
               Requires GOOGLE_API_KEY environment variable.
    - "ollama": Local Ollama Models. Currently validates and restricts models
                to "qwen3.5:9b" and "gemma4:e4b". Uses OLLAMA_BASE_URL.

    Args:
        provider: String indicating the provider ('google' or 'ollama').
        model_name: Name of the model to instantiate.
        **kwargs: Additional parameters (temperature, max_tokens, etc.) to pass
                  to the model initializer.

    Returns:
        An instance of BaseChatModel.

    Raises:
        ValueError: If provider is unknown, if the Ollama model name is invalid,
                    or if required environment configurations are missing.
    """
    provider_lower = provider.strip().lower()

    if provider_lower == "google":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set in the environment variables."
            )
        
        # Explicitly default to/use gemini-3.5-flash for free-tier rate limits
        target_model = model_name if model_name else "gemini-3.5-flash"
        if target_model in ("gemini-1.5-flash", "gemini-2.5-flash"):
            target_model = "gemini-3.5-flash"

        # Instantiate and return Google GenAI Chat Model
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=google_api_key,
            **kwargs
        )

    elif provider_lower == "ollama":
        if model_name not in VALID_OLLAMA_MODELS:
            raise ValueError(
                f"Unsupported local Ollama model: '{model_name}'. "
                f"Allowed models: {sorted(list(VALID_OLLAMA_MODELS))}"
            )
        
        # Read the OLLAMA_BASE_URL (fallback to standard local server)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Instantiate and return ChatOllama Model
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers are 'google' and 'ollama'."
        )
