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

# Allowed local models configuration optimized for VRAM/RAM constraints
VALID_OLLAMA_MODELS = {
    "llama3.2:3b", 
    "qwen3.5:4b",  
    "qwen3.5:9b", 
    "gemma4:e4b"
}

def get_llm(provider: str, model_name: str = None, **kwargs: Any) -> BaseChatModel:
    """
    LLM Factory function to return a LangChain ChatModel instance.

    Supported Providers:
    - "google": Google Gemini Models (e.g., gemini-3.5-flash, gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3.1-flash-lite).
                 Requires GOOGLE_API_KEY environment variable.
    - "ollama": Local Ollama Models. Restricted to hardware-safe configurations
                like "llama3.2:3b", "qwen3.5:4b", and "qwen3.5:2b". Uses OLLAMA_BASE_URL.

    Args:
        provider: String indicating the provider ('google' or 'ollama').
        model_name: Name of the model to instantiate. Defaults to optimized choices.
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
        
        # Default to the most cost-effective and capable API model
        if not model_name:
            model_name = "gemini-3.1-flash-lite"
            
        # Instantiate and return Google GenAI Chat Model
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            **kwargs
        )

    elif provider_lower == "ollama":
        # Default to your optimized local model
        if not model_name:
            model_name = "llama3.2:3b"

        if model_name not in VALID_OLLAMA_MODELS:
            raise ValueError(
                f"Unsupported local Ollama model: '{model_name}'. "
                f"Allowed hardware-optimized models: {sorted(list(VALID_OLLAMA_MODELS))}"
            )
        
        # Read the OLLAMA_BASE_URL (fallback to standard local server)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Extract or apply strict hardware guards to protect 8GB system RAM
        num_ctx = kwargs.pop("num_ctx", 4096)
        temperature = kwargs.pop("temperature", 0.1)

        # Instantiate and return ChatOllama Model
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            num_ctx=num_ctx,
            temperature=temperature,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers are 'google' and 'ollama'."
        )