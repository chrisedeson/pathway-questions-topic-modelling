# Hybrid Topic Analysis

AI-powered hybrid topic modeling system that combines similarity-based classification with clustering to analyze student questions and manage topic taxonomies.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/chrisedeson/pathway-questions-topic-modelling
cd pathway-questions-topic-modelling
make install

# 2. Configure API key
cp .env.template .env
# Edit .env with your OpenAI API key

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Run the application
make run
```

## Features

### Hybrid Processing

- **Similarity Classification**: Matches questions to existing topics using semantic similarity
- **Topic Discovery**: Uses clustering to identify new topics from unmatched questions
- **AI Topic Naming**: Generates meaningful topic names using GPT-5-nano

### Data Sources

- **File Upload**: CSV/TXT files with question data
- **Google Sheets**: Direct integration with existing topic databases
- **Batch Processing**: Handles large datasets efficiently

### Interactive Analysis

- **Real-time Progress**: Live tracking of embedding generation and clustering
- **Dynamic Visualizations**: Responsive charts with dataset-appropriate sizing
- **Complete Transparency**: View all data in UI and export full results

## Data Format

**Questions File** (CSV/TXT):

```
question
How do I register for classes?
What financial aid is available?
When does the semester start?
```

**Topics File** (Google Sheets or CSV):

```
Topic,Subtopic,Question
Academic,Registration,How do I register for fall classes?
Financial,Aid,What scholarships are available?
```

## Commands

```bash
make install      # Setup virtual environment and dependencies
make activate     # Shows command to activate venv
make run          # Launch Streamlit application
make clean        # Clean up cache and temporary files
make info         # Show project information
```

## Configuration

Key settings in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `SIMILARITY_THRESHOLD`: 0.70 (matching threshold)
- `MIN_CLUSTER_SIZE`: 8 (minimum questions per new topic)
- `SAMPLE_SIZE`: 2000 (questions to process in sample mode)

## Analysis Workflow

1. **Load Data**: Upload questions and existing topics
2. **Similarity Matching**: Find questions similar to existing topics
3. **Clustering**: Discover new topics from unmatched questions
4. **Topic Generation**: AI creates meaningful topic names
5. **Export Results**: Download complete analysis as CSV files

## Outputs

- **Similar Questions**: Matched to existing topics with confidence scores
- **New Topics**: Discovered topics with representative questions
- **Complete Review**: All questions with topic assignments
- **Interactive Charts**: Hierarchical clustering and similarity visualizations

## Troubleshooting

**High clustering ratio (>50%)**:

- Increase `MIN_CLUSTER_SIZE` in advanced settings
- Check similarity threshold (may be too low)

**No matches found**:

- Lower similarity threshold to 0.6-0.65
- Verify topic database format and content

**Performance issues**:

- Use sample mode for testing
- Enable caching for faster repeated analysis
