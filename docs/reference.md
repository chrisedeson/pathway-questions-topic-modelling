# Reference

## Quick Commands

### Colab Notebook
```bash
# Install dependencies (run first cell)
!pip install openai pandas hdbscan boto3

# Set credentials
import os
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["AWS_ACCESS_KEY_ID"] = "your-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret"
```

### Streamlit Dashboard
```bash
cd v2.0.0
pip install -r requirements.txt
streamlit run main.py  # Runs on http://localhost:8501
```

## File Structure

```
pathway-questions-topic-modelling/
├── notebook/
│   └── *.ipynb           # Colab notebooks
├── v2.0.0/               # Streamlit dashboard
│   ├── main.py           # Entry point
│   ├── config.py         # Settings
│   └── utils/            # Data loading, charts
├── docs/                 # Documentation
└── langfuse_*.csv        # Sample data files
```

## Common Tasks

### Change Classification Threshold
Edit in notebook: `SIMILARITY_THRESHOLD = 0.75`  
Higher = stricter matching, fewer matches

### Add New Topic
1. Open linked Google Sheets
2. Add topic name and sample questions
3. Run notebook again

### View Specific Date Range
Use dashboard date picker (top left)

### Export Data
Dashboard "Export CSV" button, or download parquet from S3

## Key URLs

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/PioneerAIAcademy/pathway-questions-topic-modelling |
| Streamlit Cloud | Check README for current deployment |
| S3 Data | s3://byupathway-public/topic-modeling-data/ |

## Environment Variables

| Variable | Purpose | Where |
|----------|---------|-------|
| `OPENAI_API_KEY` | AI embeddings | Colab, Streamlit |
| `AWS_ACCESS_KEY_ID` | S3 access | Colab, Streamlit |
| `AWS_SECRET_ACCESS_KEY` | S3 access | Colab, Streamlit |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No data" in dashboard | Check S3 connection, verify parquet files exist |
| Slow notebook | Use `EVAL_MODE = "sample"` |
| High unclassified % | Lower `SIMILARITY_THRESHOLD` |
| API errors | Check OpenAI/AWS credentials |
| Memory errors | Reduce `SAMPLE_SIZE` |

## Version History

| Version | Change |
|---------|--------|
| v2.0.0 | Streamlit dashboard, S3 integration |
| v1.0.0 | Google Colab notebook, local processing |
