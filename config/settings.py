import os
from dotenv import load_dotenv

# Load .env when running locally
load_dotenv()

class Settings:
    # ─────────────────────────────────────────────
    # Server Settings
    # ─────────────────────────────────────────────
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # ─────────────────────────────────────────────
    # Azure Deployment Flag
    # ─────────────────────────────────────────────
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "false").lower() == "true"

    # ─────────────────────────────────────────────
    # Azure OpenAI — Embeddings
    # ─────────────────────────────────────────────
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION = os.getenv(
        "AZURE_OPENAI_API_VERSION",
        "2024-12-01-preview"
    )

    # Embedding deployment (text → vector)
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "text-embedding-3-small"
    )

    # ─────────────────────────────────────────────
    # Azure OpenAI — Intent LLM (chat completions)
    # ─────────────────────────────────────────────
    # Your fine-tuned model should be set here
    AZURE_OPENAI_CHAT_MODEL = os.getenv(
        "AZURE_OPENAI_CHAT_MODEL",
        "gpt-4.1-mini"
    )

    # ─────────────────────────────────────────────
    # Spotify OAuth (used by ingestion)
    # ─────────────────────────────────────────────
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    SPOTIFY_REDIRECT_URI = os.getenv(
        "SPOTIFY_REDIRECT_URI",
        "http://127.0.0.1:8888/callback"
    )

# IMPORTANT — this is what recommender.py imports
settings = Settings()
