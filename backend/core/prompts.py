SUPERVISOR_SYSTEM = """You are SuperNova, a trading supervisor.
Goals: detect opportunities across options/FX/futures/crypto/equities; generate overlays; propose risk-aware orders.
Always: (1) fetch fresh data, (2) compute indicators, (3) optionally run a quick backtest, (4) produce concise explanation,
(5) output overlays/signals/orders JSON that conforms to schema.
Constraints: obey risk rules, never live-trade without confirmation, prefer 15m–1D timeframes unless asked otherwise."""

TOOLS = [
    {
        "name": "get_market_data",
        "description": "OHLCV, options, FX, crypto",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol (e.g., AAPL, BTC-USD)"},
                "tf": {"type": "string", "description": "Timeframe (1m, 5m, 15m, 1h, 4h, 1d)", "default": "1d"},
                "lookback": {"type": "integer", "description": "Number of periods to fetch", "default": 100}
            },
            "required": ["symbol"],
            "additionalProperties": False
        }
    },
    {
        "name": "compute_indicators",
        "description": "TA suite",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol"},
                "tf": {"type": "string", "description": "Timeframe", "default": "1d"},
                "indicators": {"type": "array", "description": "List of technical indicators to compute", "items": {"type": "string"}}
            },
            "required": ["symbol", "indicators"],
            "additionalProperties": False
        }
    },
    {
        "name": "run_backtest",
        "description": "Fast sim",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "description": "Strategy name"},
                "params": {"type": "object", "description": "Strategy parameters"},
                "symbol": {"type": "string", "description": "Trading symbol"},
                "tf": {"type": "string", "description": "Timeframe", "default": "1d"},
                "period": {"type": "string", "description": "Backtest period", "default": "1y"}
            },
            "required": ["strategy", "symbol"],
            "additionalProperties": False
        }
    },
    {
        "name": "propose_orders",
        "description": "Paper/live ticket gen",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol"},
                "side": {"type": "string", "description": "Order side (buy/sell)"},
                "qty": {"type": "number", "description": "Quantity"},
                "type": {"type": "string", "description": "Order type", "default": "market"},
                "price": {"type": "number", "description": "Price for limit orders"},
                "risk": {"type": "object", "description": "Risk management parameters"}
            },
            "required": ["symbol", "side", "qty"],
            "additionalProperties": False
        }
    },
    {
        "name": "draw_overlays",
        "description": "Chart shapes/labels",
        "input_schema": {
            "type": "object",
            "properties": {
                "overlays": {"type": "array", "description": "Array of overlay objects", "items": {"type": "object"}}
            },
            "required": ["overlays"],
            "additionalProperties": False
        }
    },
    {
        "name": "fetch_news",
        "description": "News & sentiment via FinGPT",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol"},
                "days_back": {"type": "integer", "description": "Days of news to fetch", "default": 7}
            },
            "required": ["symbol"],
            "additionalProperties": False
        }
    },
    {
        "name": "analyze_fundamentals",
        "description": "FinGPT fundamentals",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "query": {"type": "string", "description": "Specific analysis query"}
            },
            "required": ["ticker", "query"],
            "additionalProperties": False
        }
    },
    {
        "name": "forecast_stock",
        "description": "FinGPT price forecast & analysis",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol"},
                "date": {"type": "string", "description": "Target date for forecast"},
                "weeks_back": {"type": "integer", "description": "Historical data period", "default": 2},
                "include_financials": {"type": "boolean", "description": "Include fundamental data", "default": True}
            },
            "required": ["symbol", "date"],
            "additionalProperties": False
        }
    }
]