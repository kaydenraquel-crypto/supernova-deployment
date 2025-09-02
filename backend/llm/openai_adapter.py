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

OPENAI_API = "https://api.openai.com/v1/chat/completions"

class OpenAILLM(LLM):
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-0125-preview")
        self.key = os.getenv("OPENAI_API_KEY")
        if not self.key:
            raise ValueError("OPENAI_API_KEY environment variable is required but not set")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    def chat(self, *, system: str, messages: List[Dict[str, str]], tools: Optional[List[ToolSpec]]=None) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }
        
        # Convert messages to OpenAI format
        openai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            if msg.get("role") and msg.get("content"):
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        if tools:
            # Convert tools to OpenAI function format
            payload["functions"] = []
            for t in tools:
                function_schema = {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"]
                }
                payload["functions"].append(function_schema)
            payload["function_call"] = "auto"
        
        with httpx.Client(timeout=60) as client:
            response = client.post(OPENAI_API, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()