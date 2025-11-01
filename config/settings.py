import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env if present

class Settings:
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 8080))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "False").lower() == "true"
    AZURE_SQL_CONN = os.getenv("AZURE_SQL_CONN", "")

settings = Settings()
