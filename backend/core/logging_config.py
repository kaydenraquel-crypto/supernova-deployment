import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True
) -> None:
    """
    Setup comprehensive logging for SuperNova.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_console: Enable console logging
        enable_file: Enable file logging
    """
    # Use environment variable if not provided
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    if log_file is None:
        log_file = os.getenv("LOG_FILE_PATH", "logs/supernova.log")
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        try:
            # Parse log size (e.g., "100MB" -> 100 * 1024 * 1024)
            max_bytes = _parse_size(os.getenv("LOG_MAX_SIZE", "100MB"))
            backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            # Fallback to basic file handler if rotation fails
            logging.warning(f"Failed to setup rotating file handler: {e}. Using basic file handler.")
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
            except Exception as e2:
                logging.error(f"Failed to setup file logging: {e2}")
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"SuperNova logging initialized - Level: {log_level}, File: {log_file}")

def _parse_size(size_str: str) -> int:
    """Parse size string like '100MB' to bytes."""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Assume bytes
        return int(size_str)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

# Structured logging for trading operations
class TradingLogger:
    """Specialized logger for trading operations with structured logging."""
    
    def __init__(self, name: str = "trading"):
        self.logger = get_logger(name)
    
    def log_analysis_request(self, symbols: list, timeframe: str, tasks: list, user_id: Optional[str] = None):
        """Log analysis request."""
        self.logger.info(
            "Analysis request received",
            extra={
                "event_type": "analysis_request",
                "symbols": symbols,
                "timeframe": timeframe,
                "tasks": tasks,
                "user_id": user_id,
                "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            }
        )
    
    def log_analysis_complete(self, symbols: list, timeframe: str, duration_ms: float, signals_count: int):
        """Log analysis completion."""
        self.logger.info(
            "Analysis completed",
            extra={
                "event_type": "analysis_complete",
                "symbols": symbols,
                "timeframe": timeframe,
                "duration_ms": duration_ms,
                "signals_count": signals_count,
                "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            }
        )
    
    def log_tool_call(self, tool_name: str, symbol: str, duration_ms: float, success: bool):
        """Log tool call execution."""
        level = logging.INFO if success else logging.WARNING
        self.logger.log(
            level,
            "Tool call executed",
            extra={
                "event_type": "tool_call",
                "tool_name": tool_name,
                "symbol": symbol,
                "duration_ms": duration_ms,
                "success": success,
                "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            }
        )
    
    def log_error(self, error_type: str, message: str, context: dict = None):
        """Log error with context."""
        self.logger.error(
            f"Error: {message}",
            extra={
                "event_type": "error",
                "error_type": error_type,
                "message": message,
                "context": context or {},
                "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            }
        )

# Performance logging
class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)
    
    def log_api_call(self, endpoint: str, method: str, duration_ms: float, status_code: int):
        """Log API call performance."""
        self.logger.info(
            "API call completed",
            extra={
                "event_type": "api_call",
                "endpoint": endpoint,
                "method": method,
                "duration_ms": duration_ms,
                "status_code": status_code,
                "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            }
        )
    
    def log_llm_call(self, model: str, duration_ms: float, tokens_used: int, success: bool):
        """Log LLM call performance."""
        level = logging.INFO if success else logging.WARNING
        self.logger.log(
            level,
            "LLM call completed",
            extra={
                "event_type": "llm_call",
                "model": model,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
                "success": success,
                "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            }
        )

# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()
