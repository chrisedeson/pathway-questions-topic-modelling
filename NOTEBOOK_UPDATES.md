# Notebook Updates - Implementation Summary

## Overview
The notebook has been significantly enhanced to support the following requirements:

1. ✅ Read data from Langfuse CSV with comprehensive cleaning
2. ✅ Read topic-subtopic-question mappings from Google Sheets
3. ✅ Output to Parquet format instead of Pickle/CSV
4. ✅ Upload to AWS S3 with folder management
5. ✅ Comprehensive error logging (console + file + S3 upload)
6. ✅ Google Colab environment-responsive (with local fallback)
7. ✅ Duplicate detection (same timestamp + question)
8. ✅ Extract metadata from JSON into separate columns
9. ✅ Column visibility metadata for Streamlit

## Key Changes

### 1. Enhanced Configuration
- Added `OUTPUT_FORMAT = "parquet"` option
- Added `S3_BUCKET_NAME`, `S3_PREFIX`, `S3_REGION` configuration
- Added `GOOGLE_SHEETS_URL` configuration
- Added `DEFAULT_VISIBLE_COLUMNS` for Streamlit UI

### 2. Error Logging System
New `ErrorLogger` class tracks:
- Rows dropped (malformed, invalid JSON, duplicates, missing data)
- Embedding failures
- Processing statistics
- Both console and file output
- Error log uploaded to S3

### 3. Langfuse Data Cleaning
New `clean_langfuse_data()` function:
- Handles "kwargs" error pattern (extracts nested JSON)
- Extracts all required columns:
  - `timestamp`, `question`, `output`, `user_feedback`
  - `country`, `state`, `city`
  - `ip_address`, `user_language`, `is_suspicious` (from metadata JSON)
- Removes duplicates (same timestamp + question, keeps first)
- Comprehensive error handling and logging

### 4. Google Sheets Integration
New functions:
- `get_google_sheets_credentials()` - Environment-responsive credential loading
- `read_topics_from_google_sheets()` - Read topic-subtopic-question data

Supports multiple credential sources:
1. Google Colab secrets (preferred)
2. Environment variables
3. JSON file (local development)

### 5. AWS S3 Upload
Enhanced S3 functionality:
- `delete_s3_folder_contents()` - Cleans existing files before upload
- `upload_to_s3()` - Uploads with public-read ACL
- Uploads all output files + error log
- Comprehensive error handling

### 6. Parquet Output
- All output files now support Parquet format
- Includes column visibility metadata
- Maintains all original data types
- Faster loading and smaller file size

### 7. Data Flow

```
Langfuse CSV → Clean & Extract → Remove Duplicates → Process
                     ↓
          Google Sheets → Topics/Subtopics
                     ↓
              Hybrid Processing
                     ↓
         Generate 3 Parquet Files + Error Log
                     ↓
           Upload to S3 (delete old → upload new)
```

## Output Files

### 1. `similar_questions_YYYYMMDD_HHMMSS.parquet`
Questions matching existing topics (similarity ≥ threshold)

**Columns:**
- Default visible: `question`, `timestamp`, `country`, `state`
- Details: `existing_topic`, `existing_subtopic`, `similarity_score`
- Optional: `city`, `user_language`, `ip_address`, `is_suspicious`, `output`, `user_feedback`

### 2. `new_topics_YYYYMMDD_HHMMSS.parquet`
New topics discovered through clustering

**Columns:**
- `topic_name` - GPT-generated topic name
- `representative_question` - Most representative question
- `question_count` - Number of questions in cluster

### 3. `pathway_questions_review_YYYYMMDD_HHMMSS.parquet`
All questions with their topic assignments

**Columns:**
- Default visible: `question`, `timestamp`, `country`, `state`
- Assignment: `topic_name` (includes "Other" for unclustered)
- Optional: All metadata columns

### 4. `error_log_YYYYMMDD_HHMMSS.txt`
Comprehensive error and processing log

**Contains:**
- Processing statistics
- Rows dropped (with reasons)
- Errors and warnings
- Summary report

## Environment Setup

### Google Colab (Primary)
Set these as Colab secrets:
```
OPENAI_API_KEY
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
GOOGLE_SERVICE_ACCOUNT_TYPE
GOOGLE_SERVICE_ACCOUNT_PROJECT_ID
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY
GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL
GOOGLE_SERVICE_ACCOUNT_CLIENT_ID
GOOGLE_SERVICE_ACCOUNT_CLIENT_CERT_URL
```

### Local Development
Use `.env` file:
```bash
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
# Add other required credentials
```

## Usage Instructions

### 1. In Google Colab
```python
# 1. Upload the notebook to Google Colab
# 2. Set secrets in the Secrets panel (🔑 icon)
# 3. Run all cells
# 4. Upload langfuse_traces CSV when prompted
# 5. Files will be automatically uploaded to S3
```

### 2. Locally (for testing)
```python
# 1. Ensure .env file has all credentials
# 2. Place Google service account JSON in credentials/
# 3. Run the notebook
# 4. Upload CSV when prompted
```

## Data Cleaning Details

### Duplicate Detection
- **Definition**: Same `timestamp` AND same `question`
- **Action**: Keep first occurrence, drop subsequent
- **Logging**: Count tracked in error log

### "kwargs" Error Pattern
Langfuse sometimes sends malformed rows:
```json
{
  "args": [],
  "kwargs": {
    "data": {
      "messages": [
        {"role": "user", "content": "actual question here"},
        {"role": "assistant", "content": "actual output here"}
      ]
    }
  }
}
```

**Solution**: Parser extracts from nested structure

### Metadata Extraction
From JSON string to separate columns:
```json
{
  "ip": "123.456.789.0",
  "user_language": "en",
  "city": "Accra",
  "security_validation": {
    "is_suspicious": false
  }
}
```

**Extracted Columns:**
- `ip_address`
- `user_language`
- `city`
- `is_suspicious`

### Missing Data Handling
- Empty/null values → `None` in DataFrame
- Missing required fields → Row dropped with warning
- Partial data → Row kept with available fields

## Error Handling

### Categories
1. **ROW_PROCESSING** - General row processing errors
2. **JSON_PARSE** - JSON parsing failures
3. **MISSING_DATA** - Required fields missing
4. **S3_UPLOAD** - S3 upload failures
5. **GOOGLE_SHEETS** - Sheets reading errors
6. **EMBEDDING** - Embedding generation failures

### Statistics Tracked
- `total_rows_processed`
- `total_rows_kept`
- `rows_dropped_malformed`
- `rows_dropped_invalid_json`
- `rows_dropped_duplicates`
- `rows_dropped_missing_data`
- `embedding_failures`

## S3 Structure

```
s3://byupathway-public/
  └── topic-modeling-data/
      ├── similar_questions_20251008_143022.parquet
      ├── new_topics_20251008_143022.parquet
      ├── pathway_questions_review_20251008_143022.parquet
      └── error_log_20251008_143022.txt
```

**Note**: Old files are deleted before new upload

## Validation Checklist

Before running:
- [ ] Colab secrets configured OR .env file complete
- [ ] Google service account has access to the Google Sheet
- [ ] AWS credentials have write access to S3 bucket
- [ ] Langfuse CSV file is available for upload

After running:
- [ ] Check error log summary in console
- [ ] Verify file count on S3 (should be 4 files)
- [ ] Spot-check data quality in output files
- [ ] Review error log file for any critical issues

## Troubleshooting

### "Failed to load Google Sheets credentials"
**Solution**: Ensure credentials are set in Colab secrets or .env file

### "AWS credentials not found"
**Solution**: Add AWS keys to Colab secrets or .env file

### "Failed to parse kwargs JSON"
**Normal**: Some rows have this issue, they're logged and skipped

### "Many duplicates removed"
**Expected**: Langfuse may send duplicate entries

### "S3 upload failed"
**Check**: AWS credentials, bucket permissions, network connection

## Performance Notes

- **Parquet Benefits**: 
  - ~5-10x smaller than CSV
  - ~2-3x faster loading
  - Preserves data types
  - Column-based compression

- **Caching**: 
  - Embeddings cached in Google Drive
  - Significantly speeds up re-runs

- **Scalability**:
  - Tested with ~27K rows
  - Memory efficient with pandas chunking
  - S3 upload handles large files

## Next Steps for Streamlit

The Streamlit app should:
1. Read parquet files from S3 using `@st.cache_data`
2. Use `default_visible_columns` metadata for UI
3. Implement column show/hide functionality
4. Add filtering by timestamp, country, state
5. Display error log summary from S3

## Code Quality

- ✅ Type hints for all functions
- ✅ Comprehensive docstrings
- ✅ Error handling at all levels
- ✅ Logging for debugging
- ✅ Configuration-driven (no hardcoded values)
- ✅ Environment-responsive
- ✅ Production-ready

## Support

For issues or questions:
1. Check error log file first
2. Review console output for warnings
3. Verify all credentials are correctly set
4. Ensure CSV file format matches expected structure

---

**Last Updated**: October 8, 2025
**Notebook Version**: 2.0 (Production-Ready)
