# Documentation

## Quick Navigation

- [Getting Started](getting-started.md) - Access and setup
- [Workflow Guide](workflow-guide.md) - How data flows from questions to analytics
- [Notebook Guide](notebook-guide.md) - Running analysis in Google Colab
- [Dashboard Guide](dashboard-guide.md) - Using the Streamlit analytics dashboard

## What This Project Does

Analyzes student questions from the [Missionary Chatbot](https://github.com/PioneerAIAcademy/pathway-chatbot) to:

1. **Match questions** to existing topics using AI similarity
2. **Discover new topics** through clustering analysis
3. **Provide analytics** via Streamlit dashboard
4. **Track trends** in student questions over time

## How It Works

```
Langfuse traces → Google Colab notebook → AWS S3 → Streamlit dashboard
```

1. Get user questions from Langfuse (via [pathway-indexer](https://github.com/PioneerAIAcademy/pathway-indexer))
2. Run analysis notebook in Google Colab
3. Results automatically upload to S3
4. View insights in Streamlit dashboard

## Versions

- **v2.0.0** (Current) - Notebook → S3 → Streamlit workflow
- **v1.0.0** (Deprecated) - Direct upload to Streamlit, slow and error-prone

**Always use v2.0.0**

## Related Projects

- [pathway-indexer](https://github.com/PioneerAIAcademy/pathway-indexer) - Provides `extract_questions.py` for Langfuse data
- [pathway-chatbot](https://github.com/DallanQ/pathway-chatbot) - The chatbot this project analyzes
