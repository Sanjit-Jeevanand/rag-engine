from prometheus_client import Counter, Gauge, Histogram

QUERY_TOTAL = Counter(
    "rag_queries_total",
    "Total queries received",
    ["tenant", "cache_hit"],
)

QUERY_ERRORS = Counter(
    "rag_query_errors_total",
    "Query errors by stage",
    ["tenant", "stage"],
)

STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Latency per pipeline stage",
    ["stage"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

HOP_COUNT = Histogram(
    "rag_hop_count",
    "Number of retrieval hops per query",
    buckets=[1, 2, 3],
)

CACHE_HIT_RATE = Gauge(
    "rag_cache_hit_rate",
    "Rolling semantic cache hit rate",
)

FAITHFULNESS_SCORE = Gauge(
    "rag_eval_faithfulness",
    "Latest faithfulness score from eval suite (0-1)",
)
