# BYU Pathway Questions Topic Modeling

Automatically cluster and analyze student questions using AI-powered topic modeling.

## Overview

This tool helps analyze thousands of student questions by automatically:

- Clustering similar questions together
- Identifying common topics and themes
- Generating interactive visualizations
- Creating meaningful topic labels using AI

## Quick Start

### Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/chrisedeson/pathway-questions-topic-modelling
cd pathway-questions-topic-modelling
make setup

# 2. Add your OpenAI API key
make create-env
# Edit .env file with your OpenAI API key

# 3. Run the analysis
make run-notebook
```

### Usage

1. Place your questions file (`.txt`, one question per line) in the `data/` folder
2. Open the Jupyter notebook
3. Select the "Python (Pathway Questions)" kernel
4. Run all cells to perform the analysis

## Features

- **File Upload Interface** in Streamlit dashboard for easy analysis
- **OpenAI Embeddings** for semantic understanding
- **HDBSCAN Clustering** for robust topic detection
- **Interactive Visualizations** with hover details
- **AI-Generated Topic Labels** using GPT-4o-mini
- **Google Colab Support** for interactive analysis sessions
- **CSV Export** functionality for results sharing

## Requirements

- Python 3.8+
- OpenAI API key
- Questions dataset (text file, one question per line)

## Commands

```bash
make setup        # Complete project setup
make run-notebook # Start Jupyter notebook
make streamlit    # Launch Streamlit dashboard
make clean        # Clean up environment
```

## 📁 Project Structure

```
pathway-questions-topic-modelling/
├── README.md                          # This file
├── Makefile                           # Automation commands
├── requirements.txt                   # Python dependencies
├── .env.template                      # Environment variables template
├── data/                             # Data directory
│   ├── README.md                     # Data format documentation
│   └── extracted_user_inputs_*.txt   # Your question datasets
├── Text_Clustering_&_Topic_Modeling_*.ipynb  # Main analysis notebook
├── results/                          # Generated outputs (created automatically)
└── .venv/                           # Virtual environment (created by make)
```

## 📊 Data Format

Place your question files in the `data/` directory. Expected format:

- **File naming**: `extracted_user_inputs_<date>.txt`
- **Content**: One question per line
- **Encoding**: UTF-8
- **Format**: Plain text, no headers

Example:

```
How do I register for English Connect?
What are the requirements for BYU Pathway?
How can I get financial aid?
When does the next semester start?
```

### Key Parameters (adjustable in notebook)

- **min_cluster_size**: Minimum questions per cluster (default: 5)
- **embedding_model**: OpenAI embedding model (default: text-embedding-3-large)
- **chat_model**: Model for topic labeling (default: gpt-4o-mini)
- **batch_size**: Questions per API batch (default: 1000)

## 🛠 Available Commands

```bash
make help              # Show all available commands
make setup             # Complete project setup
make install           # Create venv and install dependencies
make activate          # Show activation command
make run-notebook      # Start Jupyter Notebook
make run-lab           # Start JupyterLab
make check-env         # Verify environment setup
make create-env        # Create .env template
make clean             # Remove venv and cache files
make update            # Update all dependencies
make info              # Show project information
```

## 🔬 Analysis Workflow

The notebook follows this workflow:

### 1. **Data Loading**

- Interactive file selection widget
- Automatic data validation and preview
- Support for multiple file formats

### 2. **Embedding Generation**

- OpenAI text-embedding-3-large model
- Batch processing for API efficiency
- Progress tracking and error handling

### 3. **Dimensionality Reduction**

- UMAP reduces 3072D → 5D for clustering
- UMAP reduces 3072D → 2D for visualization
- Cosine distance metric for semantic similarity

### 4. **Clustering**

- HDBSCAN identifies natural question groups
- Configurable cluster size thresholds
- Noise detection for outlier questions

### 5. **Topic Modeling**

- BERTopic combines clustering with topic extraction
- Keyword-based initial topic identification
- OpenAI GPT-4o-mini enhances topic labels

### 6. **Visualization**

- Interactive scatter plots with question hover
- Cluster color coding and size indication
- Hierarchical topic structure visualization

## 📈 Expected Outcomes

From ~3000 student questions, you can expect:

- **15-25 main topic clusters** (adjustable)
- **60-80% questions clustered** (20-40% noise is normal)
- **Clear topic separation** for major themes like:
  - Course Registration
  - Financial Aid
  - Technical Support
  - Academic Requirements
  - Program Information

## 🎨 Visualizations Generated

- **Question Clusters**: 2D scatter plot with hover details
- **Topic Bar Charts**: Most frequent keywords per topic
- **Topic Heatmap**: Similarity between topics
- **Hierarchical Clustering**: Topic relationship tree
- **Document-Topic Matrix**: Question-to-topic assignments

## 🚀 Next Steps: Streamlit Deployment

The project includes two analysis modes:

### 📊 Web Dashboard (Streamlit)

```bash
# Start the interactive dashboard:
make streamlit
```

**Features:**

- **📤 File Upload**: Drag & drop your questions file for instant analysis
- **🎯 Interactive Results**: View existing analysis results
- **📈 Real-time Processing**: Complete analysis pipeline in the browser
- **📥 CSV Export**: Download results for stakeholder sharing

### 🌐 Google Colab (Interactive Notebooks)

For maximum flexibility and step-by-step analysis:

1. Download `BYU_Pathway_Topic_Modeling_Colab.ipynb`
2. Upload to [Google Colab](https://colab.research.google.com)
3. Follow the guided analysis steps
4. Full control over parameters and intermediate results

**Perfect for:** Research, experimentation, and detailed analysis sessions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `make setup && make run-notebook`
5. Submit a pull request

### Technical Decisions

- **OpenAI over Sentence Transformers**: Higher quality embeddings
- **Batch processing**: Handles rate limits and large datasets
- **Local environment**: Better version control and reproducibility
- **Jupyter + VS Code**: Familiar development environment

## 🆘 Troubleshooting

### Common Issues

**"No embeddings generated"**

- Check your OpenAI API key in `.env`
- Verify you have questions loaded
- Check API quotas and rate limits

**"Few clusters found"**

- Reduce `min_cluster_size` parameter
- Check if you have enough questions (>100 recommended)
- Review data quality (remove duplicates)

**"High noise ratio"**

- Reduce `min_cluster_size` further
- Consider data preprocessing (cleaning, filtering)
- Check if questions are too diverse

**Import errors**

- Run `make install` to ensure all dependencies
- Check that you're using the right kernel
- Restart Jupyter after installing packages

### Getting Help

- Review the Jupyter notebook output for detailed error messages
- Ensure all requirements are met (Python version, API key, data format)
