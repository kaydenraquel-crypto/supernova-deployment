from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
try:
    from ..tools.backtester import run_backtest
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.backtester import run_backtest

router = APIRouter()

class BacktestRequest(BaseModel):
    strategy: str
    params: Dict[str, Any]
    symbol: str
    timeframe: str
    period: str

@router.post("")
def backtest(req: BacktestRequest):
    """Run a backtest for the given strategy and parameters."""
    try:
        return run_backtest(
            req.strategy,
            req.params,
            req.symbol,
            req.timeframe,
            req.period
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))