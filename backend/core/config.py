import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """SuperNova configuration settings compatible with GUI configuration."""
    
    # Core Configuration
    supernova_env: str = Field(default="development", env="SUPERNOVA_ENV")
    supernova_debug: bool = Field(default=True, env="SUPERNOVA_DEBUG")
    supernova_log_level: str = Field(default="INFO", env="SUPERNOVA_LOG_LEVEL")
    
    # Server Configuration
    supernova_host: str = Field(default="0.0.0.0", env="SUPERNOVA_HOST")
    supernova_port: int = Field(default=8081, env="SUPERNOVA_PORT")
    supernova_workers: int = Field(default=4, env="SUPERNOVA_WORKERS")
    supernova_reload: bool = Field(default=True, env="SUPERNOVA_RELOAD")
    
    # Model Configuration (GUI format)
    llm_provider: str = Field(default="anthropic", env="LLM_PROVIDER")
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", env="ANTHROPIC_MODEL")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.2, env="LLM_TEMPERATURE")
    
    # FinGPT Integration (GUI format)
    fingpt_mode: str = Field(default="remote", env="FINGPT_MODE")
    fingpt_base_url: str = Field(default="http://fingpt:8080", env="FINGPT_BASE_URL")
    fingpt_api_key: Optional[str] = Field(default=None, env="FINGPT_API_KEY")
    fingpt_local_port: int = Field(default=8080, env="FINGPT_LOCAL_PORT")
    fingpt_model_path: str = Field(default="./models/fingpt", env="FINGPT_MODEL_PATH")
    fingpt_gpu_enabled: bool = Field(default=True, env="FINGPT_GPU_ENABLED")
    fingpt_max_memory: str = Field(default="8", env="FINGPT_MAX_MEMORY")
    
    # Market Data APIs (GUI format)
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    polygon_api_key: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")
    yahoo_finance_enabled: bool = Field(default=True, env="YAHOO_FINANCE_ENABLED")
    alpaca_api_key: Optional[str] = Field(default=None, env="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, env="ALPACA_SECRET_KEY")
    fred_api_key: Optional[str] = Field(default=None, env="FRED_API_KEY")
    iex_api_key: Optional[str] = Field(default=None, env="IEX_API_KEY")
    quandl_api_key: Optional[str] = Field(default=None, env="QUANDL_API_KEY")
    tiingo_api_key: Optional[str] = Field(default=None, env="TIINGO_API_KEY")
    
    # Cryptocurrency APIs (GUI format)
    coinbase_api_key: Optional[str] = Field(default=None, env="COINBASE_API_KEY")
    coinbase_secret: Optional[str] = Field(default=None, env="COINBASE_SECRET")
    coinbase_passphrase: Optional[str] = Field(default=None, env="COINBASE_PASSPHRASE")
    coingecko_api_key: Optional[str] = Field(default=None, env="COINGECKO_API_KEY")
    coinmarketcap_api_key: Optional[str] = Field(default=None, env="COINMARKETCAP_API_KEY")
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    kraken_api_key: Optional[str] = Field(default=None, env="KRAKEN_API_KEY")
    kraken_secret_key: Optional[str] = Field(default=None, env="KRAKEN_SECRET_KEY")
    cryptocompare_api_key: Optional[str] = Field(default=None, env="CRYPTOCOMPARE_API_KEY")
    messari_api_key: Optional[str] = Field(default=None, env="MESSARI_API_KEY")
    
    # News and Alternative Data APIs (GUI format)
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    coindesk_enabled: bool = Field(default=True, env="COINDESK_ENABLED")
    bloomberg_api_key: Optional[str] = Field(default=None, env="BLOOMBERG_API_KEY")
    reuters_api_key: Optional[str] = Field(default=None, env="REUTERS_API_KEY")
    ft_api_key: Optional[str] = Field(default=None, env="FT_API_KEY")
    marketwatch_enabled: bool = Field(default=True, env="MARKETWATCH_ENABLED")
    seekingalpha_enabled: bool = Field(default=True, env="SEEKINGALPHA_ENABLED")
    reddit_api_key: Optional[str] = Field(default=None, env="REDDIT_API_KEY")
    reddit_secret: Optional[str] = Field(default=None, env="REDDIT_SECRET")
    twitter_bearer_token: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    stocktwits_api_key: Optional[str] = Field(default=None, env="STOCKTWITS_API_KEY")
    sentiment_api_key: Optional[str] = Field(default=None, env="SENTIMENT_API_KEY")
    
    # Backend Settings (GUI format)
    port: int = Field(default=8081, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    rate_limit: int = Field(default=60, env="RATE_LIMIT")
    cors_enabled: bool = Field(default=True, env="CORS_ENABLED")
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    
    # Legacy support (for backward compatibility)
    news_api_key_legacy: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    
    @validator('fingpt_mode')
    def validate_fingpt_mode(cls, v):
        valid_modes = ['local', 'remote', 'auto']
        if v not in valid_modes:
            raise ValueError(f'fingpt_mode must be one of {valid_modes}')
        return v
    
    @validator('llm_provider')
    def validate_llm_provider(cls, v):
        valid_providers = ['anthropic', 'openai', 'local']
        if v not in valid_providers:
            raise ValueError(f'llm_provider must be one of {valid_providers}')
        return v
    
    @property
    def has_market_data_provider(self) -> bool:
        """Check if at least one market data provider is configured."""
        return bool(
            self.alpha_vantage_api_key or 
            self.polygon_api_key or 
            self.finnhub_api_key or 
            self.yahoo_finance_enabled or
            self.alpaca_api_key or
            self.fred_api_key or
            self.iex_api_key or
            self.quandl_api_key or
            self.tiingo_api_key
        )
    
    @property
    def has_crypto_provider(self) -> bool:
        """Check if at least one cryptocurrency provider is configured."""
        return bool(
            self.coinbase_api_key or
            self.coingecko_api_key or
            self.coinmarketcap_api_key or
            self.binance_api_key or
            self.kraken_api_key or
            self.cryptocompare_api_key or
            self.messari_api_key
        )
    
    @property
    def has_news_provider(self) -> bool:
        """Check if at least one news provider is configured."""
        return bool(
            self.news_api_key or
            self.coindesk_enabled or
            self.bloomberg_api_key or
            self.reuters_api_key or
            self.ft_api_key or
            self.marketwatch_enabled or
            self.seekingalpha_enabled or
            self.reddit_api_key or
            self.twitter_bearer_token or
            self.stocktwits_api_key or
            self.sentiment_api_key
        )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.supernova_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.supernova_env.lower() == "development"
    
    @property
    def available_data_sources(self) -> List[str]:
        """Get list of available data sources."""
        sources = []
        if self.alpha_vantage_api_key: sources.append("Alpha Vantage")
        if self.polygon_api_key: sources.append("Polygon")
        if self.finnhub_api_key: sources.append("Finnhub")
        if self.yahoo_finance_enabled: sources.append("Yahoo Finance")
        if self.alpaca_api_key: sources.append("Alpaca")
        if self.fred_api_key: sources.append("FRED")
        if self.iex_api_key: sources.append("IEX Cloud")
        if self.quandl_api_key: sources.append("Quandl")
        if self.tiingo_api_key: sources.append("Tiingo")
        if self.coinbase_api_key: sources.append("Coinbase")
        if self.coingecko_api_key: sources.append("CoinGecko")
        if self.binance_api_key: sources.append("Binance")
        if self.kraken_api_key: sources.append("Kraken")
        if self.news_api_key: sources.append("News API")
        if self.twitter_bearer_token: sources.append("Twitter")
        return sources
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Global settings instance
settings = get_settings()