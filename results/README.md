# Results Directory

This directory contains exported analysis results and visualizations.

## Generated Files

- `pathway_questions_analysis_*.csv` - Complete analysis results with topics
- `topic_model_*.pkl` - Saved BERTopic models for reuse
- Charts and visualizations (when exported)

## File Structure

Each CSV contains:

- **Question**: Original question text
- **Topic_ID**: Numerical topic identifier (-1 for uncategorized)
- **Topic_Name**: AI-generated topic label
- **Probability**: Confidence score for topic assignment
- **Cluster**: HDBSCAN cluster assignment
- **Representative_Doc**: Whether question is representative of its topic
