import os
import httpx
from datetime import datetime
from typing import Dict, Any

class FinGPTClient:
    def __init__(self):
        self.mode = os.getenv("FINGPT_MODE", "auto")
        
        # Auto-detect mode if not explicitly set
        if self.mode == "auto":
            self.mode = self._detect_best_mode()
        
        if self.mode == "local":
            # Local FinGPT server configuration
            self.local_port = os.getenv("FINGPT_LOCAL_PORT", "8080")
            self.base = f"http://localhost:{self.local_port}"
            self.key = ""  # No API key needed for local server
        else:
            # Remote FinGPT service configuration
            self.base = os.getenv("FINGPT_BASE_URL", "http://localhost:8080")
            self.key = os.getenv("FINGPT_API_KEY", "")
    
    def _detect_best_mode(self) -> str:
        """Auto-detect the best FinGPT mode based on availability."""
        local_port = os.getenv("FINGPT_LOCAL_PORT", "8080")
        local_url = f"http://localhost:{local_port}"
        
        try:
            # Try to connect to local server
            with httpx.Client(timeout=5) as c:
                response = c.get(f"{local_url}/health")
                if response.status_code == 200:
                    return "local"
        except Exception:
            pass
        
        # Fallback to remote mode
        return "remote"

    def fetch_news(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Fetch news for a symbol."""
        headers = {}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        
        try:
            with httpx.Client(timeout=30) as c:
                if self.mode == "local":
                    # Use local FinGPT server endpoint
                    payload = {"symbol": symbol, "days_back": days_back}
                    r = c.post(f"{self.base}/news", json=payload, headers=headers)
                else:
                    # Use remote FinGPT service endpoint
                    r = c.get(
                        f"{self.base}/news",
                        params={"symbol": symbol, "days_back": days_back},
                        headers=headers
                    )
                r.raise_for_status()
                return r.json()
        except Exception as e:
            # Fallback response if FinGPT is unavailable
            return {
                "news": [
                    {
                        "headline": f"Market analysis for {symbol}",
                        "summary": f"Recent market activity and sentiment analysis for {symbol}",
                        "source": "Internal Analysis",
                        "datetime": datetime.now().isoformat()
                    }
                ],
                "summary": f"News analysis for {symbol} - FinGPT service unavailable, using fallback data",
                "error": str(e)
            }

    def analyze_fundamentals(self, ticker: str, query: str) -> Dict[str, Any]:
        """Analyze company fundamentals."""
        headers = {}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        
        try:
            with httpx.Client(timeout=60) as c:
                payload = {"ticker": ticker, "query": query}
                if self.mode == "local":
                    # Use local FinGPT server endpoint
                    r = c.post(f"{self.base}/fundamentals", json=payload, headers=headers)
                else:
                    # Use remote FinGPT service endpoint
                    r = c.post(f"{self.base}/fundamentals", json=payload, headers=headers)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            # Fallback response if FinGPT is unavailable
            return {
                "analysis": f"Fundamental analysis for {ticker}: {query}. Analysis service unavailable - please check FinGPT connection.",
                "key_metrics": {
                    "status": "unavailable",
                    "ticker": ticker,
                    "query": query
                },
                "error": str(e)
            }

    def forecast_stock(self, symbol: str, date: str = None, weeks_back: int = 2, include_financials: bool = True) -> Dict[str, Any]:
        """Generate stock forecast using FinGPT."""
        headers = {}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        
        # Use current date if not provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            with httpx.Client(timeout=120) as c:  # Longer timeout for forecasting
                payload = {
                    "symbol": symbol,
                    "date": date,
                    "weeks_back": weeks_back,
                    "include_financials": include_financials
                }
                
                if self.mode == "local":
                    # Use local FinGPT server endpoint
                    r = c.post(f"{self.base}/forecast", json=payload, headers=headers)
                else:
                    # Use remote FinGPT service endpoint (if available)
                    r = c.post(f"{self.base}/forecast", json=payload, headers=headers)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            # Fallback response if FinGPT is unavailable
            return {
                "prediction": f"Technical outlook for {symbol}: Unable to generate detailed forecast. FinGPT service unavailable.",
                "positive_developments": [
                    "Market presence maintained",
                    "Operational continuity"
                ],
                "potential_concerns": [
                    "Service connectivity issues",
                    "Limited analysis availability"
                ],
                "analysis_summary": f"Forecast for {symbol} - FinGPT service unavailable, using fallback analysis",
                "error": str(e)
            }