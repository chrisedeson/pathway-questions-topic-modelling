# BYU Pathway Topic Analysis

Analyze student questions, match to existing topics, and discover new topics using AI.

## Quick Start

1. **Run the notebook** to process data and upload to S3:
   ```bash
   jupyter notebook notebook/Hybrid_Topic_Discovery_and_Classification_with_AWS_Integration.ipynb
   ```

2. **Launch the dashboard**:
   ```bash
   cd v2.0.0
   streamlit run app.py
   ```

3. **Configure** `.env` file with:
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   OPENAI_API_KEY=your_key
   ```

## Structure

```
├── v2.0.0/          # Streamlit dashboard (current)
├── notebook/        # Data processing with Jupyter
└── v1.0.0/          # Legacy version (archived)
```

## Features

- Match questions to existing topics using semantic similarity
- Discover new topics via clustering
- Interactive dashboard with filtering and visualizations
- Export results to CSV

## Dashboard Pages

- **Questions Table**: Filter and search 4,595 questions
- **Trends & Analytics**: Charts and insights
- **New Topics**: Review discovered topics

See `v2.0.0/README.md` for detailed documentation.

