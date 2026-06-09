from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "rag-engine"
    log_level: str = "INFO"
    log_json: bool = True

    corpus_path: str = "data/corpus"
    index_path: str = "data/index"

    eval_results_path: str = "eval/results"


settings = Settings()
