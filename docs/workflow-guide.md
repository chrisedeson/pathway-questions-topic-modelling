# Workflow Guide

## Output Files (S3)

| File | Purpose |
|------|---------|
| `similar_questions_*.parquet` | Questions matched to existing topics |
| `new_topics_*.parquet` | Newly discovered topic clusters |
| `pathway_questions_review_*.parquet` | All questions with classifications |
| `error_log_*.txt` | Processing errors and warnings |

**S3 Bucket:** `byupathway-public/topic-modeling-data/`

## Configuration Options

### In Notebook

```python
EVAL_MODE = "sample"         # or "all"
SAMPLE_SIZE = 1000           # if mode is "sample"
SIMILARITY_THRESHOLD = 0.70  # 0.0 to 1.0
HDBSCAN_MIN_CLUSTER_SIZE = 3
```

### In Dashboard

Edit `v2.0.0/config.py`:
```python
CACHE_TTL = 3600             # Cache duration (seconds)
S3_BUCKET_NAME = "byupathway-public"
```