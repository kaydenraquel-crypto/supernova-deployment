import os
import httpx
from typing import Any, Dict, List, Optional
from .llm_base import LLM, ToolSpec

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ANTHROPIC_API = "https://api.anthropic.com/v1/messages"

class AnthropicLLM(LLM):
    def __init__(self):
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.key = os.getenv("ANTHROPIC_API_KEY")
        if not self.key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required but not set")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    def chat(self, *, system: str, messages: List[Dict[str, str]], tools: Optional[List[ToolSpec]]=None) -> Dict[str, Any]:
        headers = {
            "x-api-key": self.key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Clean up messages to ensure proper format
        cleaned_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), str):
                cleaned_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg["content"]
                })
            elif isinstance(msg.get("content"), list):
                # Handle complex content format
                cleaned_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg["content"]
                })
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": cleaned_messages if cleaned_messages else messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        if tools:
            # Convert tools to proper Anthropic format with valid JSON Schema
            payload["tools"] = []
            for t in tools:
                # Use the input_schema directly since it's already in the correct format
                tool_schema = {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["input_schema"]
                }
                
                payload["tools"].append(tool_schema)
        
        with httpx.Client(timeout=60) as client:
            r = client.post(ANTHROPIC_API, headers=headers, json=payload)
            try:
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                raise Exception(f"HTTP error occurred: {e}. Response: {r.text if hasattr(r, 'text') else 'No response text'}")