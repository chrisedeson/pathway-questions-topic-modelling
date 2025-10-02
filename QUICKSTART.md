# 🚀 Quick Start Guide - Database Edition

This guide will get you up and running with the new database-backed BYU Pathway Questions Analytics system in just a few minutes.

## ⚡ Prerequisites

- **PostgreSQL 12+** with pgvector extension
- **Python 3.8+**
- **OpenAI API Key**
- **Google Cloud Service Account** with Sheets API access

## 🛠️ 5-Minute Setup

### 1. Environment Setup
```bash
# Automated setup (recommended)
make install
make setup-env

# OR Manual setup
make install
make create-env
# Edit .env with your actual values
```

### 2. Database Initialization
```bash
# Initialize database tables
make init-db

# Validate everything is working
make validate-config
```

### 3. Launch Application
```bash
# Start the database-backed dashboard
make run-db

# Access at: http://localhost:8501
```

## 🔧 Configuration Checklist

Edit your `.env` file with these required values:

```bash
# Database (REQUIRED)
DATABASE_URL=postgresql://user:password@localhost:5432/pathway_questions

# OpenAI API (REQUIRED)
OPENAI_API_KEY=sk-your-actual-api-key

# Google Sheets (REQUIRED)
GOOGLE_CREDENTIALS_PATH=credentials/service-account.json
QUESTIONS_SHEET_ID=your-actual-sheet-id
TOPICS_SHEET_ID=your-actual-sheet-id

# Developer Mode (OPTIONAL)
DEV_MODE_PASSWORD=your-secure-password
```

## 📊 First Time Usage

### 1. Access Developer Mode
- Open the app: `make run-db`
- In the sidebar, click "Developer Mode"
- Enter your password (default: `pathway2024`)

### 2. Sync Data
- Click "Sync from Google Sheets"
- Monitor progress in the sync status section

### 3. Run Analysis
- Click "Start Analysis"
- Wait for background processing to complete
- View results in the main dashboard

## 🔄 Migration from File-Based System

If you have existing data from the old file-based system:

```bash
# Migrate existing CSV and cache files
make migrate-db

# This will preserve your analysis history
```

## 🚨 Troubleshooting

### Database Issues
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test database connection
make validate-config

# Recreate database tables
dropdb pathway_questions && createdb pathway_questions
make init-db
```

### Google Sheets Issues
```bash
# Verify credentials file exists
ls -la credentials/

# Test connection
make validate-config

# Check sheet permissions (service account must have Editor access)
```

### OpenAI Issues
```bash
# Validate API key
make validate-config

# Check usage/credits at: https://platform.openai.com/usage
```

## 📋 Common Commands

```bash
# Development
make run-db              # Start database app
make run                 # Start file-based app (legacy)
make validate-config     # Check configuration

# Database Management
make init-db             # Initialize database
make migrate-db          # Migrate old data
make backup-db           # Create backup

# Environment
make setup-env           # Automated setup
make check-env           # Check configuration
make info                # Show help information
```

## 🎯 What's Different?

### ⚡ Performance
- **Instant Loading**: Dashboard loads immediately from pre-computed results
- **Background Processing**: Analysis runs without blocking the UI
- **Smart Caching**: Results cached in database for fast access

### 🔄 Data Management
- **Automated Sync**: Google Sheets sync on configurable schedule
- **Duplicate Prevention**: Smart detection prevents data duplication
- **Version Control**: Analysis history preserved in database

### 🔐 Developer Features
- **Sidebar Authentication**: Secure access to admin features
- **Real-time Monitoring**: Track sync and analysis progress
- **Manual Controls**: Override automatic processes when needed

## 📈 Next Steps

1. **Explore the Dashboard**: View insights, topics, and trends
2. **Configure Sync Schedule**: Adjust sync frequency in Developer Mode
3. **Export Data**: Use export features for reports and presentations
4. **Monitor Performance**: Check system status and logs regularly

## 🤝 Need Help?

- **Configuration Issues**: Run `make validate-config`
- **Database Problems**: Check `logs/app.log`
- **Performance Issues**: Monitor system resources and database performance
- **Feature Questions**: Refer to `README_DATABASE.md` for detailed documentation

---

**🎉 You're ready to analyze BYU Pathway student questions with instant insights and automated workflows!**