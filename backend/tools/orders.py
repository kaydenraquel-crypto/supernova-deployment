from typing import Dict, Any, List, Optional

def propose_orders(symbol: str, side: str, qty: float, type: str, price: Optional[float], risk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Propose orders based on analysis results.
    """
    order = {
        "symbol": symbol,
        "side": side,
        "type": type,
        "qty": qty,
        "price": price,
        "sl": risk.get("sl"),
        "tp": risk.get("tp")
    }
    
    return [order]