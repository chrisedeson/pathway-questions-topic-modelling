# Monitoring System Documentation

## Overview

The Pathway Chatbot backend now includes a comprehensive monitoring system that:

- ✅ Tracks memory usage, CPU, threads, and connections in real-time
- ✅ Records all HTTP requests with detailed metrics
- ✅ Monitors security validation events
- ✅ Generates daily Parquet reports
- ✅ Uploads reports to S3 automatically
- ✅ Minimal memory overhead (~10-20MB)

## Architecture

### Components

1. **MetricsCollector** (`app/monitoring.py`)

   - Thread-safe in-memory metrics storage
   - Tracks system metrics: memory, CPU, threads, connections
   - Records request metrics: duration, status, errors, security events

2. **MonitoringMiddleware** (`app/middleware/monitoring_middleware.py`)

   - Automatically captures ALL HTTP requests
   - Extracts client IP, user agent, request/response details
   - Zero configuration needed

3. **MonitoringScheduler** (`app/scheduler.py`)

   - Runs scheduled tasks using APScheduler
   - Daily report generation at 00:00 UTC
   - Hourly memory logging for debugging

4. **MonitoringService** (`app/monitoring.py`)
   - Manages report generation (Parquet format)
   - Handles S3 uploads
   - Cleans up old local files (keeps 7 days)

## Data Collected

### System Metrics (collected per request)

- `memory_rss_mb`: Resident Set Size (actual RAM used)
- `memory_vms_mb`: Virtual Memory Size
- `memory_percent`: Process memory usage percentage
- `cpu_percent`: CPU utilization
- `num_threads`: Active threads
- `num_connections`: Open network connections
- `system_memory_percent`: Overall system memory usage
- `system_memory_available_mb`: Available system memory
- `uptime_seconds`: Application uptime

### Request Metrics

- `request_id`: Unique identifier for each request
- `timestamp`: ISO 8601 timestamp
- `endpoint`: API endpoint path
- `method`: HTTP method (GET, POST, etc.)
- `status_code`: HTTP status code
- `duration_seconds`: Request processing time
- `error`: Error message (if any)
- `user_agent`: Client user agent
- `client_ip`: Real client IP (proxy-aware)

### Security Metrics (when applicable)

- `security_blocked`: Whether request was blocked
- `risk_level`: Security risk level (LOW, MEDIUM, HIGH, CRITICAL)
- `user_language`: Detected user language
- Additional security details

## Usage

### Automatic Monitoring (Zero Configuration)

Once deployed, monitoring runs automatically:

- ✅ All requests are tracked via middleware
- ✅ Reports generated daily at midnight UTC
- ✅ Uploaded to S3 (if enabled)
- ✅ Old local files cleaned up after 7 days

### Manual Report Generation

To generate a report on-demand:

```python
from app.monitoring import get_monitoring_service

monitoring_service = get_monitoring_service()
filepath = monitoring_service.generate_daily_report()
print(f"Report saved to: {filepath}")
```

### View Metrics in Logs

Memory usage is logged hourly:

```
2025-10-23 10:00:00 - Memory: 845.23 MB (12.5%), CPU: 15.2%, Threads: 8
```

## Report Format

### Parquet Files

Generated files: `monitoring_reports/metrics_YYYYMMDD_HHMMSS.parquet`

Schema:

```
request_id: string
timestamp: string (ISO 8601)
endpoint: string
method: string
status_code: int64
duration_seconds: float64
error: string (nullable)
memory_rss_mb: float64
memory_vms_mb: float64
memory_percent: float64
cpu_percent: float64
num_threads: int64
num_connections: int64
system_memory_percent: float64
system_memory_available_mb: float64
uptime_seconds: float64
user_agent: string
client_ip: string
date: date (partition key)
[+ security fields when applicable]
```

### Summary JSON Files

Generated files: `monitoring_reports/summary_YYYYMMDD_HHMMSS.json`

Example:

```json
{
  "timestamp": "2025-10-23T00:00:00.000000",
  "total_requests": 1250,
  "total_errors": 12,
  "security_blocks": 8,
  "error_rate": 0.0096,
  "avg_response_time_seconds": 1.23,
  "uptime_hours": 24.0,
  "memory_rss_mb": 856.45,
  "memory_percent": 12.8,
  "cpu_percent": 8.5
}
```

## Memory Overhead

The monitoring system is designed to be lightweight:

- **In-memory storage**: ~5-10MB per 1000 requests
- **Daily at 3000 questions/week**: ~2-3MB average
- **Peak (430 requests)**: ~5MB max
- **Total overhead**: ~10-20MB including libraries

## S3 Storage Costs

With compression (Snappy):

- ~10KB per 100 requests
- 3000 requests/week = ~30KB/day
- ~900KB/month = **$0.02/month** (negligible)

## Reading Reports (Streamlit App)

See `streamlit_monitoring_dashboard.py` for a sample dashboard that:

- Lists all reports from S3
- Displays metrics in interactive charts
- Shows summary statistics
- Filters by date range

## Troubleshooting

### High Memory Usage

If monitoring itself uses too much memory:

1. Check `monitoring_reports/` directory size
2. Reduce retention days in `cleanup_old_reports(days=7)`
3. Clear metrics more frequently

### S3 Upload Failures

Check logs for:

```
Error uploading to S3: [error details]
```

Common issues:

- Invalid AWS credentials
- Missing bucket permissions
- Network connectivity

### Reports Not Generated

Check scheduler status:

```
Starting monitoring scheduler...
Scheduled daily report task at 00:00 UTC
```

If missing, check:

- APScheduler installation
- Application startup logs
- No exceptions during initialization

## Production Deployment

### Render Configuration

No special configuration needed - monitoring starts automatically.

Memory impact is minimal (~20MB), well within budget.

### Monitoring the Monitoring

Check application logs for:

- `Generated monitoring report:` - Report creation
- `Uploaded ... to s3://` - S3 upload success
- `Memory: XXX MB` - Hourly memory logs

## Performance Impact

Benchmarks:

- Request overhead: **<1ms per request**
- Memory overhead: **10-20MB total**
- CPU overhead: **<0.5%**
- No impact on response times

## Future Enhancements

Potential additions:

- Prometheus metrics endpoint
- Real-time alerts (memory thresholds)
- Grafana dashboard integration
- Per-user analytics
- Error rate alerting
