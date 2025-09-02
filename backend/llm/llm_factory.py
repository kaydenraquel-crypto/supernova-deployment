"""LLM Factory with fallback support."""
import os
from typing import Optional
from .llm_base import LLM
from .claude_adapter import AnthropicLLM
from .openai_adapter import OpenAILLM

def get_llm(provider: Optional[str] = None) -> LLM:
    """
    Get LLM instance with fallback support.
    
    Priority:
    1. Specified provider (if valid)
    2. Environment variable LLM_PROVIDER
    3. Claude (if API key exists)
    4. OpenAI (if API key exists)
    5. Raise error if no provider available
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "auto")
        print(f"[DEBUG] LLM_PROVIDER from env: {provider}")
    
    # Try the specified provider first
    if provider == "anthropic":
        try:
            print("[INFO] Attempting to use Anthropic Claude...")
            return AnthropicLLM()
        except Exception as e:
            print(f"[WARN] Failed to initialize Anthropic: {e}")
            provider = "auto"  # Fall back to auto mode
    
    elif provider == "openai":
        try:
            print("[INFO] Attempting to use OpenAI GPT...")
            return OpenAILLM()
        except Exception as e:
            print(f"[WARN] Failed to initialize OpenAI: {e}")
            provider = "auto"  # Fall back to auto mode
    
    # Auto mode - try available providers
    if provider == "auto":
        # Try Claude first
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                print("[INFO] Using Anthropic Claude as LLM provider")
                return AnthropicLLM()
            except Exception as e:
                print(f"[WARN] Claude initialization failed: {e}")
        
        # Try OpenAI as fallback
        if os.getenv("OPENAI_API_KEY"):
            try:
                print("[INFO] Using OpenAI GPT as LLM provider (fallback)")
                return OpenAILLM()
            except Exception as e:
                print(f"[WARN] OpenAI initialization failed: {e}")
        
        raise ValueError(
            "No LLM provider available. Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "environment variable with a valid API key."
        )
    
    raise ValueError(f"Unknown LLM provider: {provider}")