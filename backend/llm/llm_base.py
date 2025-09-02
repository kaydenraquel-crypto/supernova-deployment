from typing import Any, Dict, List, Optional, TypedDict

class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]

class LLM:
    def chat(self, *, system: str, messages: List[Dict[str, str]], tools: Optional[List[ToolSpec]]=None) -> Dict[str, Any]:
        raise NotImplementedError