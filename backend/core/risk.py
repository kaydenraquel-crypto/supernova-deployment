from typing import List, Dict, Any

MAX_RISK_PCT = 0.02
MAX_DAILY_DD = 0.05

def enforce_orders(orders: List[Dict[str, Any]], equity: float) -> List[Dict[str, Any]]:
    """
    Enforce risk management rules on proposed orders.
    Clamp sizes, reject risky tickets, add SL/TP defaults if missing.
    """
    safe_orders = []
    
    for order in orders:
        # Calculate position size limit based on risk percentage
        max_position_value = equity * MAX_RISK_PCT
        
        # Adjust quantity if position value exceeds risk limit
        if order.get("price") and order.get("qty"):
            position_value = order["price"] * order["qty"]
            if position_value > max_position_value:
                order["qty"] = max_position_value / order["price"]
        
        # Add default stop loss if missing (2% for long, 2% for short)
        if not order.get("sl"):
            if order.get("side") == "buy" and order.get("price"):
                order["sl"] = order["price"] * 0.98  # 2% below entry
            elif order.get("side") == "sell" and order.get("price"):
                order["sl"] = order["price"] * 1.02  # 2% above entry
        
        # Add default take profit if missing (3:1 risk/reward)
        if not order.get("tp") and order.get("price") and order.get("sl"):
            if order.get("side") == "buy":
                risk = order["price"] - order["sl"]
                order["tp"] = order["price"] + (risk * 3)
            elif order.get("side") == "sell":
                risk = order["sl"] - order["price"]
                order["tp"] = order["price"] - (risk * 3)
        
        safe_orders.append(order)
    
    return safe_orders