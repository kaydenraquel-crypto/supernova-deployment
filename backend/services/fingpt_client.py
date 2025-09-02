"""
Legacy import for backward compatibility.
The FinGPTClient is now located in llm/fingpt_adapter.py
"""

from ..llm.fingpt_adapter import FinGPTClient

__all__ = ['FinGPTClient']