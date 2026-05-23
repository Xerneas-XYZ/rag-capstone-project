from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    faiss_index_path: str = "data/faiss_index"

    csv_data_path: str = "data/csv"

    crag_relevance_threshold: float = 0.65
    crag_max_reflections: int = 2

    mcp_server_url: str = "http://localhost:8001"
    
    langchain_tracing_v2: str = "true"
    langchain_api_key: str = ""
    langchain_project: str = "capstone - project"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()