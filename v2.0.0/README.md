# BYU Pathway Missionary Question Analysis Dashboard v2.0.0

A professional, scalable Streamlit dashboard for analyzing student questions and discovering topics.

## 🌟 Features

### 📊 Interactive Dashboard
- **Real-time KPIs**: Total questions, matched topics, new discoveries, geographic reach
- **Instant filtering**: No page refresh needed - all operations happen in memory
- **Smart caching**: Data loaded once and cached for performance

### 📋 Questions Table
- **Advanced filtering**: By classification, date range, country, similarity score
- **Search functionality**: Find specific keywords in questions
- **Column customization**: Show/hide columns based on your needs
- **Multiple sort options**: Sort by timestamp, similarity, country, etc.
- **Export capabilities**: Download filtered or complete data as CSV

### 📈 Trends & Analytics
- **Classification distribution**: Pie charts showing existing vs new topics
- **Geographic insights**: Bar charts and statistics by country
- **Temporal patterns**: Timeline graphs and activity heatmaps
- **Topic analysis**: Most common topics and coverage statistics
- **Language distribution**: User language breakdown

### 🆕 New Topics Discovery
- **Topic exploration**: Browse newly discovered topics one by one
- **Representative questions**: See all questions in each topic
- **Topic summaries**: Keywords, names, and statistics
- **Export options**: Download individual topics or all new topics

## 🏗️ Architecture

```
v2.0.0/
├── app.py                      # Main entry point
├── config.py                   # Configuration and constants
├── requirements.txt            # Python dependencies
├── pages/                      # Streamlit pages
│   ├── 1_📋_Questions_Table.py
│   ├── 2_📈_Trends_&_Analytics.py
│   └── 3_🆕_New_Topics.py
└── utils/                      # Utility modules
    ├── data_loader.py          # S3 data loading and processing
    └── visualizations.py       # Chart and visualization functions
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- AWS credentials configured
- Data files uploaded to S3 by the Jupyter notebook

### Installation

1. **Install dependencies:**
```bash
cd v2.0.0
pip install -r requirements.txt
```

2. **Set up environment variables:**

Create a `.env` file in the project root (not in v2.0.0 folder) with:
```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_S3_BUCKET=byupathway-public
AWS_S3_PREFIX=topic-modeling-data
```

3. **Run the dashboard:**
```bash
streamlit run app.py
```

4. **Access the dashboard:**
Open your browser to `http://localhost:8501`

## 📦 Data Requirements

The dashboard expects these files in S3 (uploaded by the notebook):

1. **similar_questions_*.parquet** - Questions matched to existing topics
2. **new_topics_*.parquet** - Newly discovered topics from clustering
3. **pathway_questions_review_*.parquet** - Complete review file
4. **topic_distribution_*.parquet** - Topic distribution statistics

The dashboard automatically loads the most recent version of each file.

## 🎨 Features in Detail

### Instant Filtering
All filters work on cached data in memory. No server calls or database queries needed:
- Classification (Existing/New Topic)
- Date range selection
- Country multiselect
- Similarity score threshold
- Text search in questions

### Column Customization
Default visible columns:
- Question (input)
- Timestamp
- Country
- State/Province

Additional columns available:
- City
- Matched Topic & Subtopic
- Similarity Score
- Classification
- Response (output)
- User Feedback
- Language
- IP Address
- Suspicious Flag
- Cluster ID
- Topic Keywords

### Performance Optimization
- **Caching**: Data loaded once per hour (configurable)
- **In-memory operations**: All filtering and sorting in pandas
- **Efficient data types**: Parquet format for fast loading
- **Lazy loading**: Only load data when needed

## 🎯 Use Cases

### For Data Scientists
- Analyze question patterns and trends
- Identify emerging topics
- Validate clustering results
- Export data for modeling

### For Content Managers
- Review new topics for taxonomy updates
- Monitor question distribution
- Track geographic reach
- Understand student needs

### For Administrators
- Monitor system usage
- Track processing statistics
- Review geographic coverage
- Export reports for stakeholders

## 🔧 Configuration

### Cache Settings
Adjust cache TTL in `config.py`:
```python
CACHE_TTL = 3600  # seconds (default: 1 hour)
```

### Display Defaults
Customize default columns in `config.py`:
```python
DEFAULT_VISIBLE_COLUMNS = [
    "input",
    "timestamp",
    "country",
    "state"
]
```

### Chart Colors
Modify BYU brand colors in `config.py`:
```python
BYU_COLORS = {
    "primary": "#002E5D",    # BYU Navy
    "secondary": "#FFB933",  # BYU Gold
    ...
}
```

## 📊 Data Flow

```
Jupyter Notebook → Process Questions → Upload to S3
                                          ↓
                                    Streamlit Dashboard
                                          ↓
                        Load & Cache Data (1 hour TTL)
                                          ↓
                                    User Interactions
                                    (instant filters)
```

## 🐛 Troubleshooting

### No data appears
- Verify AWS credentials are correct
- Check S3 bucket name and prefix
- Ensure notebook has uploaded files
- Check browser console for errors

### Slow performance
- Increase cache TTL
- Reduce data volume in notebook
- Check network connection to S3

### Missing columns
- Verify column names in S3 files
- Check config.py for column mappings
- Ensure notebook outputs required columns

## 🤝 Contributing

This is v2.0.0 - a complete rewrite focused on:
- ✅ Professional UI/UX
- ✅ Scalable architecture
- ✅ Performance optimization
- ✅ Clean separation of concerns
- ✅ No settings/configuration UI (all in notebook)

## 📝 Version History

### v2.0.0 (Current)
- Complete rewrite from v1.0.0
- Moved all processing to notebook
- Dashboard is pure visualization
- Added advanced filtering and search
- Improved performance with caching
- Professional UI with BYU branding
- Modular architecture for scalability

### v1.0.0 (Archived in v1.0.0/ folder)
- Original version with embedded processing
- Settings and configuration in UI
- Monolithic structure

## 📄 License

See LICENSE file in repository root.

## 🙏 Acknowledgments

- **BYU Pathway Worldwide** for the project requirements
- **OpenAI** for embeddings and topic generation
- **AWS S3** for reliable data storage
- **Streamlit** for the dashboard framework
