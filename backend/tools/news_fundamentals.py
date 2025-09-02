from typing import Dict, Any
try:
    from ..llm.fingpt_adapter import FinGPTClient
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llm.fingpt_adapter import FinGPTClient

_fingpt_client = FinGPTClient()

def fetch_news(symbol: str, days_back: int = 7) -> Dict[str, Any]:
    """
    Fetch news and sentiment data for the given symbol via FinGPT.
    """
    return _fingpt_client.fetch_news(symbol, days_back)

def analyze_fundamentals(ticker: str, query: str) -> Dict[str, Any]:
    """
    Analyze fundamentals for the given ticker via FinGPT.
    """
    return _fingpt_client.analyze_fundamentals(ticker, query)

def forecast_stock(symbol: str, date: str = "2024-01-01", weeks_back: int = 2, include_financials: bool = True) -> Dict[str, Any]:
    """
    Generate stock price forecast and analysis via FinGPT.
    """
    return _fingpt_client.forecast_stock(symbol, date, weeks_back, include_financials)