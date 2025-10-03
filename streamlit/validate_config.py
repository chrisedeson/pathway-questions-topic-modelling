"""
Configuration Validation Script for BYU Pathway Questions Analytics
Validates all environment variables and external service connections
"""

import os
import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Any

# Add src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str = "info"):
    """Print colored status messages"""
    if status == "success":
        print(f"{Colors.GREEN}✅ {message}{Colors.END}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠️ {message}{Colors.END}")
    elif status == "error":
        print(f"{Colors.RED}❌ {message}{Colors.END}")
    else:
        print(f"{Colors.BLUE}ℹ️ {message}{Colors.END}")

def load_environment():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if not env_file.exists():
        print_status(".env file not found", "error")
        return False
    
    try:
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value
        
        print_status(".env file loaded successfully", "success")
        return True
        
    except Exception as e:
        print_status(f"Failed to load .env file: {e}", "error")
        return False

def validate_database_config() -> Tuple[bool, List[str]]:
    """Validate database configuration"""
    issues = []
    
    # Check DATABASE_URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        issues.append("DATABASE_URL is not set")
    elif not database_url.startswith('postgresql://'):
        issues.append("DATABASE_URL must be a PostgreSQL connection string")
    elif 'your_' in database_url or 'password' in database_url:
        issues.append("DATABASE_URL appears to contain placeholder values")
    
    # Check database connection settings
    db_settings = {
        'DB_POOL_SIZE': (5, 50),
        'DB_MAX_OVERFLOW': (10, 100),
        'DB_POOL_TIMEOUT': (10, 120),
        'DB_POOL_RECYCLE': (1800, 7200)
    }
    
    for setting, (min_val, max_val) in db_settings.items():
        value = os.getenv(setting)
        if value:
            try:
                int_value = int(value)
                if not (min_val <= int_value <= max_val):
                    issues.append(f"{setting} should be between {min_val} and {max_val}")
            except ValueError:
                issues.append(f"{setting} must be a valid integer")
    
    return len(issues) == 0, issues

def test_database_connection() -> bool:
    """Test actual database connection"""
    try:
        from database import get_db_manager
        db_manager = get_db_manager()
        return db_manager.test_connection()
    except Exception as e:
        print_status(f"Database connection test failed: {e}", "error")
        return False

def validate_openai_config() -> Tuple[bool, List[str]]:
    """Validate OpenAI configuration"""
    issues = []
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        issues.append("OPENAI_API_KEY is not set")
    elif api_key.startswith('your_') or api_key == 'your_openai_api_key_here':
        issues.append("OPENAI_API_KEY appears to contain placeholder value")
    elif not api_key.startswith('sk-'):
        issues.append("OPENAI_API_KEY should start with 'sk-'")
    
    # Check model configuration
    embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    valid_models = ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
    if embedding_model not in valid_models:
        issues.append(f"OPENAI_EMBEDDING_MODEL should be one of: {', '.join(valid_models)}")
    
    return len(issues) == 0, issues

def test_openai_connection() -> bool:
    """Test OpenAI API connection"""
    try:
        from hybrid_topic_processor import HybridTopicProcessor
        processor = HybridTopicProcessor()
        test_embedding = processor.embedder._get_embedding("test connection")
        return test_embedding is not None and len(test_embedding) > 0
    except Exception as e:
        print_status(f"OpenAI connection test failed: {e}", "error")
        return False

def validate_google_sheets_config() -> Tuple[bool, List[str]]:
    """Validate Google Sheets configuration"""
    issues = []
    
    # Check credentials path
    creds_path = os.getenv('GOOGLE_CREDENTIALS_PATH') or os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')
    if not creds_path:
        issues.append("GOOGLE_CREDENTIALS_PATH is not set")
    elif not Path(creds_path).exists():
        issues.append(f"Google credentials file not found: {creds_path}")
    else:
        try:
            with open(creds_path) as f:
                creds = json.load(f)
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                for field in required_fields:
                    if field not in creds:
                        issues.append(f"Google credentials missing field: {field}")
        except json.JSONDecodeError:
            issues.append("Google credentials file is not valid JSON")
        except Exception as e:
            issues.append(f"Error reading Google credentials: {e}")
    
    # Check Sheet IDs
    questions_sheet_id = os.getenv('QUESTIONS_SHEET_ID')
    topics_sheet_id = os.getenv('TOPICS_SHEET_ID')
    
    if not questions_sheet_id or questions_sheet_id.startswith('your_'):
        issues.append("QUESTIONS_SHEET_ID is not properly configured")
    
    if not topics_sheet_id or topics_sheet_id.startswith('your_'):
        issues.append("TOPICS_SHEET_ID is not properly configured")
    
    return len(issues) == 0, issues

def test_google_sheets_connection() -> bool:
    """Test Google Sheets API connection"""
    try:
        from sheets_sync import get_sheets_manager
        sheets_manager = get_sheets_manager()
        return sheets_manager.test_connection()
    except Exception as e:
        print_status(f"Google Sheets connection test failed: {e}", "error")
        return False

def validate_developer_mode_config() -> Tuple[bool, List[str]]:
    """Validate developer mode configuration"""
    issues = []
    
    # Check password
    dev_password = os.getenv('DEV_MODE_PASSWORD')
    if not dev_password:
        issues.append("DEV_MODE_PASSWORD is not set")
    elif dev_password == 'pathway2024':
        issues.append("DEV_MODE_PASSWORD is using default value (consider changing for production)")
    elif len(dev_password) < 8:
        issues.append("DEV_MODE_PASSWORD should be at least 8 characters")
    
    return len(issues) == 0, issues

def validate_caching_config() -> Tuple[bool, List[str]]:
    """Validate caching configuration"""
    issues = []
    
    # Check cache settings
    cache_ttl = os.getenv('CACHE_TTL_HOURS')
    if cache_ttl:
        try:
            ttl_value = int(cache_ttl)
            if ttl_value < 1 or ttl_value > 168:  # 1 hour to 1 week
                issues.append("CACHE_TTL_HOURS should be between 1 and 168 (1 week)")
        except ValueError:
            issues.append("CACHE_TTL_HOURS must be a valid integer")
    
    # Check cache directory
    cache_dir = os.getenv('CACHE_DIR', 'embeddings_cache/')
    if not Path(cache_dir).exists():
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            print_status(f"Created cache directory: {cache_dir}", "info")
        except Exception as e:
            issues.append(f"Cannot create cache directory {cache_dir}: {e}")
    
    return len(issues) == 0, issues

def validate_sync_config() -> Tuple[bool, List[str]]:
    """Validate sync configuration"""
    issues = []
    
    # Check sync interval
    sync_interval = os.getenv('DEFAULT_SYNC_INTERVAL_MINUTES')
    if sync_interval:
        try:
            interval_value = int(sync_interval)
            if interval_value < 5 or interval_value > 1440:  # 5 minutes to 24 hours
                issues.append("DEFAULT_SYNC_INTERVAL_MINUTES should be between 5 and 1440")
        except ValueError:
            issues.append("DEFAULT_SYNC_INTERVAL_MINUTES must be a valid integer")
    
    return len(issues) == 0, issues

def generate_config_report() -> Dict[str, Any]:
    """Generate comprehensive configuration report"""
    report = {
        'overall_status': 'unknown',
        'sections': {},
        'recommendations': []
    }
    
    # Database configuration
    db_valid, db_issues = validate_database_config()
    db_connected = test_database_connection() if db_valid else False
    
    report['sections']['database'] = {
        'config_valid': db_valid,
        'connection_working': db_connected,
        'issues': db_issues
    }
    
    # OpenAI configuration
    openai_valid, openai_issues = validate_openai_config()
    openai_connected = test_openai_connection() if openai_valid else False
    
    report['sections']['openai'] = {
        'config_valid': openai_valid,
        'connection_working': openai_connected,
        'issues': openai_issues
    }
    
    # Google Sheets configuration
    sheets_valid, sheets_issues = validate_google_sheets_config()
    sheets_connected = test_google_sheets_connection() if sheets_valid else False
    
    report['sections']['google_sheets'] = {
        'config_valid': sheets_valid,
        'connection_working': sheets_connected,
        'issues': sheets_issues
    }
    
    # Developer mode configuration
    dev_valid, dev_issues = validate_developer_mode_config()
    report['sections']['developer_mode'] = {
        'config_valid': dev_valid,
        'issues': dev_issues
    }
    
    # Caching configuration
    cache_valid, cache_issues = validate_caching_config()
    report['sections']['caching'] = {
        'config_valid': cache_valid,
        'issues': cache_issues
    }
    
    # Sync configuration
    sync_valid, sync_issues = validate_sync_config()
    report['sections']['sync'] = {
        'config_valid': sync_valid,
        'issues': sync_issues
    }
    
    # Overall status
    all_configs_valid = all(section['config_valid'] for section in report['sections'].values())
    critical_connections = ['database', 'openai', 'google_sheets']
    critical_connected = all(
        report['sections'][service].get('connection_working', False) 
        for service in critical_connections
    )
    
    if all_configs_valid and critical_connected:
        report['overall_status'] = 'ready'
    elif all_configs_valid:
        report['overall_status'] = 'configured'
    else:
        report['overall_status'] = 'needs_configuration'
    
    return report

def print_configuration_report(report: Dict[str, Any]):
    """Print formatted configuration report"""
    print(f"\n{Colors.BOLD}BYU Pathway Questions Analytics - Configuration Report{Colors.END}")
    print("=" * 70)
    
    # Overall status
    if report['overall_status'] == 'ready':
        print_status("System is fully configured and ready to use!", "success")
    elif report['overall_status'] == 'configured':
        print_status("System is configured but some connections failed", "warning")
    else:
        print_status("System needs additional configuration", "error")
    
    print()
    
    # Section details
    for section_name, section_data in report['sections'].items():
        print(f"{Colors.BOLD}{section_name.replace('_', ' ').title()}:{Colors.END}")
        
        # Configuration status
        if section_data['config_valid']:
            print_status("Configuration is valid", "success")
        else:
            print_status("Configuration has issues", "error")
            for issue in section_data['issues']:
                print(f"  • {issue}")
        
        # Connection status (if applicable)
        if 'connection_working' in section_data:
            if section_data['connection_working']:
                print_status("Connection test passed", "success")
            else:
                print_status("Connection test failed", "error")
        
        print()
    
    # Recommendations
    if report['overall_status'] != 'ready':
        print(f"{Colors.BOLD}Recommendations:{Colors.END}")
        
        if report['overall_status'] == 'needs_configuration':
            print("1. Review and update .env file with correct values")
            print("2. Ensure all required services are accessible")
            print("3. Run this validation script again after making changes")
        
        if not report['sections']['database']['connection_working']:
            print("• Set up PostgreSQL database with pgvector extension")
            print("• Run: python setup_database.py")
        
        if not report['sections']['openai']['connection_working']:
            print("• Verify OpenAI API key is valid and has sufficient credits")
        
        if not report['sections']['google_sheets']['connection_working']:
            print("• Check Google service account credentials and permissions")
            print("• Ensure sheets are shared with service account email")
        
        print()

def main():
    """Main validation process"""
    print(f"{Colors.BOLD}BYU Pathway Questions Analytics - Configuration Validator{Colors.END}")
    print("This script validates your environment configuration and service connections")
    print()
    
    # Load environment
    if not load_environment():
        sys.exit(1)
    
    # Generate and print report
    try:
        report = generate_config_report()
        print_configuration_report(report)
        
        # Exit with appropriate code
        if report['overall_status'] == 'ready':
            print_status("Validation completed successfully! You can now run the application.", "success")
            sys.exit(0)
        else:
            print_status("Validation completed with issues. Please address them before running the application.", "warning")
            sys.exit(1)
            
    except Exception as e:
        print_status(f"Validation failed with error: {e}", "error")
        sys.exit(1)

if __name__ == "__main__":
    main()