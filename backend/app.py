# Load environment variables first
import os
from pathlib import Path

# Load .env file from the supernova directory
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"[OK] Loaded environment from: {env_file}")
    except ImportError:
        print("[WARN] python-dotenv not installed, using system environment variables")
else:
    print(f"[WARN] .env file not found at: {env_file}")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from contextlib import asynccontextmanager

# Import routes using absolute imports
from routes import analyze, backtest

# Import configuration and logging
from core.config import settings
from core.logging_config import get_logger, PerformanceLogger

# Initialize logging
logger = get_logger(__name__)
perf_logger = PerformanceLogger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("[STARTUP] Starting SuperNova Trading Analysis Service...")
    logger.info(f"Environment: {settings.supernova_env}")
    logger.info(f"Port: {settings.port}")
    logger.info(f"Debug Mode: {settings.debug_mode}")
    
    # Log available data sources
    if settings.available_data_sources:
        logger.info(f"[DATA] Available data sources: {', '.join(settings.available_data_sources)}")
    else:
        logger.warning("[WARN] No market data providers configured")
    
    if settings.has_crypto_provider:
        logger.info("[CRYPTO] Cryptocurrency data available")
    
    if settings.has_news_provider:
        logger.info("[NEWS] News and sentiment data available")
    
    # Log FinGPT configuration
    if settings.fingpt_mode == "local":
        logger.info(f"[FINGPT] Local FinGPT server configured on port {settings.fingpt_local_port}")
    elif settings.fingpt_mode == "remote":
        logger.info(f"[FINGPT] Remote FinGPT server: {settings.fingpt_base_url}")
    
    yield
    
    # Shutdown
    logger.info("[SHUTDOWN] Shutting down SuperNova...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="SuperNova Trading Analysis Service",
    description="AI-powered trading analysis, signal generation, and backtesting service",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add performance logging middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    perf_logger.log_api_call(
        endpoint=str(request.url.path),
        method=request.method,
        duration_ms=duration_ms,
        status_code=response.status_code
    )
    
    return response

# Include routers
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(backtest.router, prefix="/backtest", tags=["backtest"])

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check if required services are available
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "environment": settings.supernova_env,
            "services": {
                "anthropic": bool(settings.anthropic_api_key),
                "market_data": settings.has_market_data_provider,
                "crypto": settings.has_crypto_provider,
                "news": settings.has_news_provider,
                "fingpt": settings.fingpt_mode in ["local", "remote"]
            },
            "data_sources": settings.available_data_sources,
            "config": {
                "port": settings.port,
                "log_level": settings.log_level,
                "rate_limit": settings.rate_limit,
                "cors_enabled": settings.cors_enabled,
                "debug_mode": settings.debug_mode
            }
        }
        
        # Determine overall health
        if not settings.anthropic_api_key:
            health_status["status"] = "degraded"
            health_status["warnings"] = ["Anthropic API key not configured"]
        
        if not settings.has_market_data_provider:
            health_status["status"] = "degraded"
            if "warnings" not in health_status:
                health_status["warnings"] = []
            health_status["warnings"].append("No market data providers configured")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "SuperNova Trading Analysis Service",
        "version": "2.0.0",
        "description": "AI-powered trading analysis and signal generation",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "backtest": "/backtest"
        },
        "status": "running"
    }

# Configuration endpoint
@app.get("/config", tags=["config"])
async def get_config():
    """Get current configuration (without sensitive data)."""
    return {
        "environment": settings.supernova_env,
        "port": settings.port,
        "log_level": settings.log_level,
        "rate_limit": settings.rate_limit,
        "cors_enabled": settings.cors_enabled,
        "debug_mode": settings.debug_mode,
        "llm_provider": settings.llm_provider,
        "anthropic_model": settings.anthropic_model,
        "fingpt_mode": settings.fingpt_mode,
        "data_sources": {
            "market_data": settings.has_market_data_provider,
            "crypto": settings.has_crypto_provider,
            "news": settings.has_news_provider,
            "available_sources": settings.available_data_sources
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug_mode else "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("SUPERNOVA_HOST", "0.0.0.0")
    port = int(os.getenv("SUPERNOVA_PORT", "8081"))
    reload = os.getenv("SUPERNOVA_RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting SuperNova on {host}:{port}")
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower()
    )