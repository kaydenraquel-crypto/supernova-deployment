import json
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
try:
    from ..core.schema_overlay import AnalyzeResponse
    from ..llm.llm_factory import get_llm
    from ..core.prompts import SUPERVISOR_SYSTEM, TOOLS
    from ..tools.market_data import get_market_data
    from ..tools.indicators import compute_indicators
    from ..tools.backtester import run_backtest
    from ..tools.orders import propose_orders
    from ..tools.overlays import draw_overlays
    from ..tools.news_fundamentals import fetch_news, analyze_fundamentals, forecast_stock
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.schema_overlay import AnalyzeResponse
    from llm.llm_factory import get_llm
    from core.prompts import SUPERVISOR_SYSTEM, TOOLS
    from tools.market_data import get_market_data
    from tools.indicators import compute_indicators
    from tools.backtester import run_backtest
    from tools.orders import propose_orders
    from tools.overlays import draw_overlays
    from tools.news_fundamentals import fetch_news, analyze_fundamentals, forecast_stock

router = APIRouter()

# Initialize LLM with fallback support
try:
    _llm = get_llm()  # Will automatically try Claude first, then OpenAI
except Exception as e:
    print(f"[ERROR] Failed to initialize LLM: {e}")
    _llm = None

# Tool function registry
TOOL_FUNCTIONS = {
    "get_market_data": get_market_data,
    "compute_indicators": compute_indicators,
    "run_backtest": run_backtest,
    "propose_orders": propose_orders,
    "draw_overlays": draw_overlays,
    "fetch_news": fetch_news,
    "analyze_fundamentals": analyze_fundamentals,
    "forecast_stock": forecast_stock
}

class AnalyzeRequest(BaseModel):
    query: str
    model: Optional[str] = "claude-3-7-sonnet-20250219"
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = "auto"

def _extract_json(response_content: str) -> dict:
    """Extract JSON from Claude's response."""
    # Try to find JSON block in the response
    json_match = re.search(r'\{[\s\S]*\}', response_content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: return empty structure
    return {
        "explanation": "Analysis completed but no structured output generated.",
        "overlays": [],
        "signals": [],
        "orders": []
    }

def _execute_tool_call(tool_name: str, arguments: dict):
    """Execute a tool function call."""
    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool_func = TOOL_FUNCTIONS[tool_name]
    return tool_func(**arguments)

def _process_messages_with_tools(messages: List[dict], tools: List[dict]) -> dict:
    """Process messages with tool calls iteratively until final response."""
    current_messages = messages.copy()
    max_iterations = 10  # Prevent infinite loops
    
    for _ in range(max_iterations):
        # Call Claude with current messages
        result = _llm.chat(
            system=SUPERVISOR_SYSTEM,
            messages=current_messages,
            tools=tools
        )
        
        # Check if Claude wants to use tools
        if result.get("stop_reason") == "tool_use":
            # First, add the assistant's message with tool_use blocks
            current_messages.append({
                "role": "assistant",
                "content": result.get("content", [])
            })
            
            # Then execute tool calls and collect results
            tool_results = []
            for content in result.get("content", []):
                if content.get("type") == "tool_use":
                    tool_name = content["name"]
                    tool_args = content["input"]
                    
                    try:
                        tool_result = _execute_tool_call(tool_name, tool_args)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content["id"],
                            "content": json.dumps(tool_result)
                        })
                    except Exception as e:
                        # Add error result
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content["id"],
                            "content": f"Error: {str(e)}"
                        })
            
            # Add all tool results in a single user message
            if tool_results:
                current_messages.append({
                    "role": "user",
                    "content": tool_results
                })
        else:
            # No more tool calls, return final result
            return result
    
    # If we hit max iterations, return the last result
    return result

@router.post("", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Analyze a trading query using the LLM and available tools."""
    try:
        # Use tools from request or default to all available tools
        tools_to_use = req.tools if req.tools is not None else TOOLS
        
        # Create initial message
        messages = [{
            "role": "user",
            "content": req.query
        }]
        
        # Process with tool calls
        result = _process_messages_with_tools(messages, tools_to_use)
        
        # Extract content from result
        content_text = ""
        if isinstance(result.get("content"), list):
            for content_item in result["content"]:
                if content_item.get("type") == "text":
                    content_text += content_item.get("text", "")
        else:
            content_text = result.get("content", "")
        
        # Extract and parse JSON
        content_dict = _extract_json(content_text)
        
        # Validate and return
        return AnalyzeResponse(**content_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))