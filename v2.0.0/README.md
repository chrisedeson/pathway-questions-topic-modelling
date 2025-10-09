# BYU Pathway Dashboard v2.0.0

Interactive Streamlit dashboard for analyzing student questions and discovering topics.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env file
echo "AWS_ACCESS_KEY_ID=your_key" > ../.env
echo "AWS_SECRET_ACCESS_KEY=your_secret" >> ../.env

# Run dashboard
streamlit run app.py
```

## Features

- **4,595 questions** from Langfuse data
- **Advanced filtering**: classification, date, country, similarity, text search
- **Visualizations**: distributions, timelines, heatmaps
- **Topic discovery**: explore 302 new topics
- **Dark/Light theme** toggle

## Pages

1. **Home** - KPIs and overview
2. **Questions Table** - Filter and search questions
3. **Trends & Analytics** - Charts and insights
4. **New Topics** - Review discovered topics

## Configuration

Edit `config.py` for:
- Cache TTL (default: 3600s)
- Display columns
- BYU brand colors
- S3 bucket settings

## Data Files (from S3)

- `pathway_questions_review_*.parquet` - All questions
- `similar_questions_*.parquet` - Matched topics
- `new_topics_*.parquet` - Discovered topics
- `topic_distribution_*.parquet` - Statistics

Dashboard automatically loads the most recent files.

## Troubleshooting

- **No data**: Check AWS credentials and S3 bucket name
- **Slow**: Increase `CACHE_TTL` in config.py
- **Missing columns**: Verify parquet files from notebook

See main README for full project documentation.


