import functools
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_KEY: str

    # Supabase JWT secret (for token verification)
    SUPABASE_JWT_SECRET: str = ""

    # Google / Gemini
    GOOGLE_API_KEY: str

    # LLM configuration
    LLM_PROVIDER: str = "gemini"          # "gemini" | "ollama_gemma"
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"
    GEMMA_MODEL: str = "gemma3:4b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # LangChain / LangSmith
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "invoice-analyst-dev"

    # Sentry
    SENTRY_DSN: str = ""

    # Developer alerting
    ALERT_DISCORD_WEBHOOK_URL: str = ""   # empty -> alerts logged to DB only
    ALERT_COOLDOWN_SECONDS: int = 600     # Discord suppression window per fingerprint

    # App
    ENV: str = "development"
    ALLOWED_ORIGINS: str = "http://localhost:8501"
    API_BASE_URL: str = "http://localhost:8000"

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 80

    # Agent loop limits
    MAX_AGENT_ITERATIONS: int = 3
    MAX_CRITIQUE_ITERATIONS: int = 2

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
