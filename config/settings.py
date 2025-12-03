# config/settings.py

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env locally only ##
if not os.getenv("AZURE_DEPLOYMENT"):
    load_dotenv()

class Settings:
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Azure deployment flag
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "false").lower() == "true"

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # Spotify (ingestion)
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

    # Azure SQL (optional)
    AZURE_SQL_CONN = os.getenv("AZURE_SQL_CONN", "")

    BASE_DIR = Path(__file__).resolve().parent.parent
    STATIC_DIR = BASE_DIR / "static"
    LOGS_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "llm_recommender" / "data"

settings = Settings()
