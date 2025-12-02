# Getting Started

## What You Need

- **Google Colab access** - Where analysis runs
- **AWS S3 access** - Where results are stored
- **Langfuse traces file** - Input data from pathway-indexer

Contact **Christopher or Ernest** for access. They will share the Google Colab notebook link, which automatically includes all API keys and secrets.

## Quick Start

### Running Analysis (Weekly Task)

1. Get latest Langfuse traces from pathway-indexer:
   ```bash
   cd pathway-indexer
   python extract_questions.py
   # Creates: traces_YYYY-MM-DD.csv
   ```

2. Open Google Colab notebook (link from Christopher/Ernest)

3. Click "Runtime" → "Run all"

4. When prompted, upload the traces CSV

5. Wait 5-15 minutes for processing

6. Results automatically upload to S3

### Viewing Analytics

If someone already ran the notebook, just open the dashboard:

```bash
cd v2.0.0
pip install -r requirements.txt
streamlit run app.py
```

Or use the deployed version (link from Christopher/Ernest).

## File Requirements

### Input: Langfuse Traces CSV

Get this from running `extract_questions.py` in pathway-indexer. Required columns:
- `timestamp` or `createdAt`
- `input` (user questions)
- `output` (chatbot responses)
- `metadata` (JSON with location, language)
- `user_feedback`

**Use the traces file, not extracted_user_inputs** - traces has all fields.

## Project Structure

```
pathway-questions-topic-modelling/
├── v2.0.0/                  # Current dashboard
│   ├── app.py              # Main entry
│   ├── config.py           # Configuration
│   └── pages/              # Dashboard pages
├── notebook/                # Reference notebook (actual work in Colab)
├── v1.0.0/                 # Deprecated
└── .env                    # AWS credentials (not in git)
```

## Environment Variables

Create `.env` file in project root:
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

**Never commit `.env` to git!**

## Common Questions

**Do I need to install anything locally?**
- For running analysis: No, use Google Colab
- For viewing dashboard: Only if running locally

**How often should I run the notebook?**
- Weekly for regular updates
- After major chatbot changes

**Where is the actual notebook?**
- In Google Colab, not this repository
- Repository notebook is for reference only
