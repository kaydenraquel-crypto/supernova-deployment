#!/usr/bin/env python3
"""
SuperNova FinGPT Local Server
A FastAPI server that wraps FinGPT functionality for SuperNova integration.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import torch
import pandas as pd
import yfinance as yf
import finnhub
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class FinGPTConfig:
    def __init__(self):
        self.port = int(os.getenv("FINGPT_LOCAL_PORT", 8080))
        self.model_path = os.getenv("FINGPT_MODEL_PATH", "./models/fingpt")
        self.gpu_enabled = os.getenv("FINGPT_GPU_ENABLED", "true").lower() == "true"
        self.max_memory = int(os.getenv("FINGPT_MAX_MEMORY", 8))
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.huggingface_token = os.getenv("HF_TOKEN")

config = FinGPTConfig()

# Global model variables
model = None
tokenizer = None
finnhub_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup and shutdown."""
    # Startup
    global model, tokenizer, finnhub_client
    
    logger.info("Starting SuperNova FinGPT Server...")
    logger.info(f"GPU Enabled: {config.gpu_enabled}")
    logger.info(f"Max Memory: {config.max_memory}GB")
    
    # Initialize Finnhub client
    if config.finnhub_api_key:
        finnhub_client = finnhub.Client(api_key=config.finnhub_api_key)
        logger.info("Finnhub client initialized")
    
    # Initialize FinGPT model (this will take time)
    try:
        await load_fingpt_model()
        logger.info("FinGPT model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load FinGPT model: {e}")
        logger.warning("Running in fallback mode without FinGPT model")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SuperNova FinGPT Server...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SuperNova FinGPT Server",
    description="Local FinGPT server for SuperNova financial analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class NewsRequest(BaseModel):
    symbol: str
    days_back: int = 7

class FundamentalsRequest(BaseModel):
    ticker: str
    query: str

class ForecastRequest(BaseModel):
    symbol: str
    date: str
    weeks_back: int = 2
    include_financials: bool = True

class AnalysisResponse(BaseModel):
    analysis: str
    sentiment: str = "neutral"
    confidence: float = 0.5

class NewsResponse(BaseModel):
    news: List[Dict[str, Any]]
    summary: str = ""

class FundamentalsResponse(BaseModel):
    analysis: str
    key_metrics: Dict[str, Any] = {}

class ForecastResponse(BaseModel):
    prediction: str
    positive_developments: List[str] = []
    potential_concerns: List[str] = []
    analysis_summary: str = ""


async def load_fingpt_model():
    """Load the FinGPT model asynchronously with enhanced error handling."""
    global model, tokenizer
    
    logger.info("Loading FinGPT model... This may take several minutes.")
    
    # Validate HuggingFace token
    if not config.huggingface_token:
        logger.error("HF_TOKEN environment variable not set. Cannot load model without authentication.")
        raise ValueError("HuggingFace token is required but not provided")
    
    # Determine device and settings
    device = "cuda" if config.gpu_enabled and torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    logger.info(f"Using device: {device}")
    logger.info(f"Using dtype: {torch_dtype}")
    
    try:
        # Load base model with enhanced error handling
        logger.info("Loading base Llama-2 model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            token=config.huggingface_token,
            trust_remote_code=True,
            device_map="auto" if config.gpu_enabled else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        logger.info("Base model loaded successfully")
        
        # Load FinGPT LoRA adapter
        logger.info("Loading FinGPT LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora',
            torch_dtype=torch_dtype
        )
        model = model.eval()
        logger.info("FinGPT LoRA adapter loaded successfully")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            token=config.huggingface_token
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("FinGPT model and tokenizer loaded successfully")
        
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.error("Please install missing packages: pip install torch transformers peft accelerate")
        raise
    except OSError as e:
        logger.error(f"Model file access error: {e}")
        logger.error("Check your internet connection and HuggingFace access permissions")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading FinGPT model: {e}")
        logger.error("The server will continue in fallback mode without model-based analysis")
        model = None
        tokenizer = None
        # Don't re-raise - allow server to start in fallback mode

def get_stock_data(symbol: str, days_back: int = 30) -> pd.DataFrame:
    """Get stock price data using yfinance."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return pd.DataFrame()

def get_company_news(symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
    """Get company news using Finnhub or fallback methods."""
    news = []
    
    if finnhub_client:
        try:
            # Get company news from Finnhub
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            news_data = finnhub_client.company_news(
                symbol, 
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
            for item in news_data[:10]:  # Limit to 10 most recent
                news.append({
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'datetime': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                    'source': item.get('source', 'Unknown')
                })
                
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
    
    # Fallback: Generate mock news if API fails
    if not news:
        news = [{
            'headline': f'Market analysis for {symbol}',
            'summary': f'Recent trading activity and market sentiment for {symbol}',
            'url': '',
            'datetime': datetime.now().isoformat(),
            'source': 'Internal Analysis'
        }]
    
    return news

def get_company_fundamentals(symbol: str) -> Dict[str, Any]:
    """Get company fundamental data."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        fundamentals = {
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'roe': info.get('returnOnEquity'),
            'revenue_growth': info.get('revenueGrowth'),
            'profit_margins': info.get('profitMargins'),
            'current_ratio': info.get('currentRatio'),
            'beta': info.get('beta')
        }
        
        # Filter out None values
        fundamentals = {k: v for k, v in fundamentals.items() if v is not None}
        return fundamentals
        
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        return {}

def generate_fingpt_response(prompt: str) -> str:
    """Generate response using FinGPT model."""
    if model is None or tokenizer is None:
        return "FinGPT model not available. Using fallback analysis."
    
    try:
        # Format prompt for Llama-2 chat
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        system_prompt = ("You are a seasoned stock market analyst. Your task is to provide "
                        "financial analysis based on the given information.")
        
        formatted_prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{prompt} {E_INST}"
        
        # Tokenize and generate
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        if torch.cuda.is_available() and config.gpu_enabled:
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the model's response part
        response = response.split(E_INST)[-1].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating FinGPT response: {e}")
        return "Analysis generation failed. Please try again."

@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "service": "SuperNova FinGPT Server",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "gpu_enabled": config.gpu_enabled,
        "endpoints": ["/news", "/fundamentals", "/forecast", "/health"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not_loaded",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/news", response_model=NewsResponse)
async def get_news(request: NewsRequest):
    """Get company news and basic sentiment analysis."""
    try:
        news = get_company_news(request.symbol, request.days_back)
        
        # Generate summary using FinGPT if available
        if news and model is not None:
            news_text = "\n".join([f"- {item['headline']}: {item['summary']}" for item in news[:5]])
            prompt = f"Summarize the key themes in this recent news for {request.symbol}:\n{news_text}"
            summary = generate_fingpt_response(prompt)
        else:
            summary = f"Recent news summary for {request.symbol} - {len(news)} articles found"
        
        return NewsResponse(news=news, summary=summary)
        
    except Exception as e:
        logger.error(f"Error in news endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fundamentals", response_model=FundamentalsResponse)
async def analyze_fundamentals(request: FundamentalsRequest):
    """Analyze company fundamentals."""
    try:
        fundamentals = get_company_fundamentals(request.ticker)
        
        if model is not None:
            # Create analysis prompt
            fund_text = json.dumps(fundamentals, indent=2)
            prompt = f"Analyze the following financial metrics for {request.ticker} and answer: {request.query}\n\nMetrics:\n{fund_text}"
            analysis = generate_fingpt_response(prompt)
        else:
            analysis = f"Fundamental analysis for {request.ticker}: Key metrics retrieved. Model analysis not available."
        
        return FundamentalsResponse(analysis=analysis, key_metrics=fundamentals)
        
    except Exception as e:
        logger.error(f"Error in fundamentals endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_stock(request: ForecastRequest):
    """Generate stock forecast using FinGPT."""
    try:
        # Gather data
        news = get_company_news(request.symbol, request.weeks_back * 7)
        stock_data = get_stock_data(request.symbol, request.weeks_back * 7)
        fundamentals = get_company_fundamentals(request.symbol) if request.include_financials else {}
        
        if model is not None:
            # Create comprehensive prompt
            news_text = "\n".join([f"- {item['headline']}" for item in news[:10]])
            
            stock_summary = ""
            if not stock_data.empty:
                latest_price = stock_data['Close'].iloc[-1]
                price_change = ((latest_price - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100
                stock_summary = f"Recent price: ${latest_price:.2f}, Change: {price_change:.1f}%"
            
            fund_summary = ""
            if fundamentals:
                key_metrics = [f"{k}: {v}" for k, v in list(fundamentals.items())[:5]]
                fund_summary = f"Key metrics: {', '.join(key_metrics)}"
            
            prompt = f"""Analyze {request.symbol} for the week starting {request.date}:

Recent News:
{news_text}

Stock Performance:
{stock_summary}

Fundamentals:
{fund_summary}

Provide analysis in this format:
[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]
Prediction: ...
Analysis: ..."""
            
            analysis = generate_fingpt_response(prompt)
            
            # Parse the response
            positive_developments = []
            potential_concerns = []
            prediction = "Neutral outlook based on available data"
            analysis_summary = analysis
            
            # Simple parsing (could be improved with regex)
            if "[Positive Developments]" in analysis:
                pos_section = analysis.split("[Positive Developments]")[1].split("[Potential Concerns]")[0]
                positive_developments = [line.strip() for line in pos_section.split('\n') if line.strip() and line.strip().startswith(('1.', '2.', '3.', '-'))]
            
            if "[Potential Concerns]" in analysis:
                concern_section = analysis.split("[Potential Concerns]")[1].split("[Prediction & Analysis]")[0]
                potential_concerns = [line.strip() for line in concern_section.split('\n') if line.strip() and line.strip().startswith(('1.', '2.', '3.', '-'))]
            
            if "Prediction:" in analysis:
                pred_section = analysis.split("Prediction:")[1].split("Analysis:")[0]
                prediction = pred_section.strip()
            
        else:
            # Fallback analysis
            analysis_summary = f"Technical analysis for {request.symbol}: {len(news)} news items analyzed"
            positive_developments = ["Strong market presence", "Continued operations"]
            potential_concerns = ["Market volatility", "Sector challenges"]
            prediction = "Cautious outlook - monitor key developments"
        
        return ForecastResponse(
            prediction=prediction,
            positive_developments=positive_developments,
            potential_concerns=potential_concerns,
            analysis_summary=analysis_summary
        )
        
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting SuperNova FinGPT Server on port {config.port}")
    logger.info("Note: Model loading may take several minutes on first startup")
    
    uvicorn.run(
        "supernova_fingpt_server:app",
        host="0.0.0.0",
        port=config.port,
        log_level="info",
        reload=False  # Disable reload for model persistence
    )