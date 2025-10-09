# Quick Start Guide - Hybrid Topic Discovery Notebook

## 🚀 Running in Google Colab (Recommended)

### Step 1: Setup Secrets
Click the 🔑 Secrets icon in Colab sidebar and add:

#### Required Secrets
```
OPENAI_API_KEY = your-openai-api-key

AWS_ACCESS_KEY_ID = your-aws-access-key
AWS_SECRET_ACCESS_KEY = your-aws-secret-key

GOOGLE_SERVICE_ACCOUNT_TYPE = service_account
GOOGLE_SERVICE_ACCOUNT_PROJECT_ID = your-project-id
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID = your-private-key-id
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY = -----BEGIN PRIVATE KEY-----
(copy from your credentials file)
-----END PRIVATE KEY-----
GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL = streamlit-sheets-reader@byu-pathway-chatbot.iam.gserviceaccount.com
GOOGLE_SERVICE_ACCOUNT_CLIENT_ID = 112187323533628090799
GOOGLE_SERVICE_ACCOUNT_CLIENT_CERT_URL = https://www.googleapis.com/robot/v1/metadata/x509/streamlit-sheets-reader%40byu-pathway-chatbot.iam.gserviceaccount.com
```

### Step 2: Enable Notebook Access
- Toggle "Notebook access" ON for all secrets

### Step 3: Run the Notebook
1. Click "Runtime" → "Run all"
2. When prompted, upload `langfuse_traces_10_08_25.csv` file
3. Wait for processing (~5-15 minutes depending on data size)

### Step 4: Verify Results
Check console output for:
- ✅ Data loading success
- ✅ Processing summary
- ✅ S3 upload confirmation
- ✅ Error log summary

### Step 5: Access Files on S3
Files will be at:
```
https://byupathway-public.s3.amazonaws.com/topic-modeling-data/similar_questions_TIMESTAMP.parquet
https://byupathway-public.s3.amazonaws.com/topic-modeling-data/new_topics_TIMESTAMP.parquet
https://byupathway-public.s3.amazonaws.com/topic-modeling-data/pathway_questions_review_TIMESTAMP.parquet
https://byupathway-public.s3.amazonaws.com/topic-modeling-data/error_log_TIMESTAMP.txt
```

---

## 💻 Running Locally (Testing Only)

### Prerequisites
```bash
pip install openai pandas numpy scipy scikit-learn matplotlib seaborn tqdm \
            umap-learn hdbscan bertopic backoff boto3 pyarrow fastparquet \
            gspread google-auth google-auth-oauthlib python-dotenv
```

### Step 1: Verify .env File
Ensure `.env` has all credentials (already configured)

### Step 2: Place Service Account JSON
```bash
mkdir -p credentials
# Copy your service account JSON to:
# credentials/byu-pathway-chatbot-service-account.json
```

### Step 3: Run Notebook
```bash
jupyter notebook
# Open notebook and run all cells
```

---

## 📊 Configuration Options

Edit these in the Configuration cell:

### Processing Mode
```python
EVAL_MODE = "sample"  # or "all"
SAMPLE_SIZE = 1000    # if mode is "sample"
```

### Similarity Threshold
```python
SIMILARITY_THRESHOLD = 0.70  # 0.0 to 1.0
```

### Output Format
```python
OUTPUT_FORMAT = "parquet"  # or "csv"
```

### Clustering Parameters
```python
HDBSCAN_MIN_CLUSTER_SIZE = 3  # minimum questions per cluster
```

---

## 📁 Expected Input File Format

### Langfuse CSV Columns
Required columns:
- `timestamp` or `createdAt`
- `input` (may contain kwargs JSON)
- `output`
- `metadata` (JSON string with ip, user_language, city, etc.)
- `user_feedback`

Optional direct columns:
- `country`
- `state`
- `city`

---

## 🔍 What the Notebook Does

1. **Loads Data**
   - Topics from Google Sheets
   - Questions from Langfuse CSV

2. **Cleans Data**
   - Removes duplicates
   - Parses nested JSON
   - Extracts metadata

3. **Processes Questions**
   - Generates embeddings
   - Matches to existing topics (similarity)
   - Discovers new topics (clustering)

4. **Generates Outputs**
   - Similar questions file
   - New topics file
   - Complete review file
   - Error log

5. **Uploads to S3**
   - Deletes old files
   - Uploads new files
   - Makes files public

---

## ✅ Success Indicators

Look for these messages:

```
✅ Error logging initialized
✅ Langfuse data cleaning utilities loaded!
✅ Google Sheets integration loaded!
✅ Topics loaded successfully
✅ Langfuse data cleaned successfully
✅ Data loading and preprocessing complete!
✅ AWS credentials loaded
✅ Uploaded and made public: [filename]
✅ OUTPUT FILES GENERATION COMPLETE
✅ S3 UPLOAD COMPLETE!
```

---

## ⚠️ Common Issues

### Issue: "AWS credentials not found"
**Solution**: Add to Colab secrets or .env file

### Issue: "Failed to read Google Sheet"
**Solution**: Verify service account has access to the sheet

### Issue: "File not found for upload"
**Solution**: Check that output files were generated successfully

### Issue: Many rows dropped
**Review**: Error log to understand why (likely duplicates or malformed data)

---

## 📈 Performance Tips

1. **Use Google Colab Pro** for faster processing
2. **Enable GPU** (Runtime → Change runtime type)
3. **Mount Google Drive** for embedding cache
4. **Sample mode** for quick testing
5. **Full mode** for production runs

---

## 🆘 Getting Help

If you encounter issues:

1. **Check error log** (`error_log_TIMESTAMP.txt`)
2. **Review console output** for specific error messages
3. **Verify credentials** are correctly set
4. **Test with sample mode** first
5. **Check S3 bucket** permissions

---

## 📝 Notes

- **First run**: Takes longer due to embedding generation
- **Subsequent runs**: Faster with cached embeddings
- **Sample mode**: Good for testing (~2-3 minutes)
- **Full mode**: For production (~10-20 minutes)
- **Data size**: Tested with 27K rows successfully

---

**Happy Processing! 🎉**
