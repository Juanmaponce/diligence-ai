from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    embed_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    collection_name: str = "financial_docs"
    chunk_size: int = 512      # tokens
    chunk_overlap: int = 64
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    api_key: str = ""  # If empty, auth is disabled (dev mode)
    max_content_length: int = 2_000_000  # ~2MB of text

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def chroma_base_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"

    class Config:
        env_file = ".env"


settings = Settings()
