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

    # Serving — tuned on M4 Pro (8 performance + 4 efficiency cores)
    # Phase 4 baseline: c=8 is the Pareto knee — 18.5K QPS, p99=0.736ms
    # c=12 adds only +12% QPS but doubles p99; c=16 hits 4.7ms p99
    hnsw_path: str = "data/hnsw.index"
    hnsw_ef_search: int = 64  # Phase 3 Pareto knee: 98.6% recall, 0.387ms p50
    search_workers: int = 8  # one per performance core; efficiency cores add noise
    faiss_omp_threads: int = 1  # 1 OMP thread per search; higher oversubscribes at c=8


settings = Settings()
