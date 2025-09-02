from pydantic import BaseModel, Field, conlist
from typing import List, Literal, Tuple, Optional

Point = Tuple[float, float]  # [timestamp_ms, price]

class OverlayLine(BaseModel):
    type: Literal["line"] = "line"
    points: conlist(Point, min_length=2)
    style: Literal["solid","dashed","dotted"] = "dashed"

class OverlayLabel(BaseModel):
    type: Literal["label"] = "label"
    at: Point
    text: str

Overlay = OverlayLine | OverlayLabel

class Signal(BaseModel):
    symbol: str
    tf: str
    bias: Literal["long","short","neutral"]
    confidence: float = Field(ge=0, le=1)

class Order(BaseModel):
    symbol: str
    side: Literal["buy","sell"]
    type: Literal["limit","market","stop"]
    qty: float
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None

class AnalyzeResponse(BaseModel):
    explanation: str
    overlays: List[Overlay] = []
    signals: List[Signal] = []
    orders: List[Order] = []