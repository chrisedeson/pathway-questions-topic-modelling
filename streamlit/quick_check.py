"""
Quick Configuration Check for BYU Pathway Questions Analytics
Lightweight validation without loading heavy dependencies
"""

import os
import json
from pathlib import Path

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
        return False
    
    try:
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    value = value.strip('"\'')
                    os.environ[key] = value
        return True
    except Exception:
        return False

def check_basic_config():
    """Check basic configuration requirements"""
    issues = []
    
    # Check .env file exists
    if not Path('.env').exists():
        issues.append("❌ .env file not found")
        return issues
    
    print_status(".env file found", "success")
    
    # Load environment
    if not load_environment():
        issues.append("❌ Failed to load .env file")
        return issues
    
    print_status(".env file loaded", "success")
    
    # Check required variables
    required_vars = {
        'DATABASE_URL': 'PostgreSQL connection string',
        'OPENAI_API_KEY': 'OpenAI API key for embeddings',
        'GOOGLE_CREDENTIALS_PATH': 'Path to Google service account JSON',
        'QUESTIONS_SHEET_ID': 'Google Sheets ID for questions',
        'TOPICS_SHEET_ID': 'Google Sheets ID for topics'
    }
    
    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if not value:
            issues.append(f"❌ {var_name} not set ({description})")
        elif value.startswith('your_') or 'your-' in value or value.endswith('_here'):
            issues.append(f"⚠️ {var_name} has placeholder value")
        else:
            print_status(f"{var_name} configured", "success")
    
    # Check Google credentials file
    creds_path = os.getenv('GOOGLE_CREDENTIALS_PATH')
    if creds_path:
        if Path(creds_path).exists():
            try:
                with open(creds_path) as f:
                    creds = json.load(f)
                    if 'client_email' in creds and 'private_key' in creds:
                        print_status("Google credentials file valid", "success")
                    else:
                        issues.append("⚠️ Google credentials file missing required fields")
            except json.JSONDecodeError:
                issues.append("❌ Google credentials file is not valid JSON")
        else:
            issues.append(f"❌ Google credentials file not found: {creds_path}")
    
    # Check directories
    required_dirs = ['logs', 'credentials']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            Path(dir_name).mkdir(exist_ok=True)
            print_status(f"Created {dir_name}/ directory", "info")
    
    return issues

def check_python_environment():
    """Check Python environment and dependencies"""
    issues = []
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
    else:
        print_status(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}", "success")
    
    # Check virtual environment
    if sys.prefix == sys.base_prefix:
        issues.append("⚠️ Not running in virtual environment (recommended: source .venv/bin/activate)")
    else:
        print_status("Virtual environment active", "success")
    
    # Check key dependencies (without importing)
    key_deps = ['sqlalchemy', 'streamlit', 'pandas', 'psycopg2']
    try:
        import pkg_resources
        installed_packages = {pkg.project_name.lower() for pkg in pkg_resources.working_set}
        
        for dep in key_deps:
            if dep.lower() in installed_packages:
                print_status(f"{dep} installed", "success")
            else:
                issues.append(f"❌ {dep} not installed")
                
    except ImportError:
        issues.append("⚠️ Cannot check installed packages (pkg_resources not available)")
    
    return issues

def check_system_requirements():
    """Check system-level requirements"""
    issues = []
    
    # Check PostgreSQL client
    import subprocess
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status(f"PostgreSQL client: {version}", "success")
        else:
            issues.append("⚠️ PostgreSQL client (psql) not found")
    except FileNotFoundError:
        issues.append("⚠️ PostgreSQL client (psql) not found")
    
    return issues

def main():
    """Main configuration check"""
    print(f"{Colors.BOLD}🚀 BYU Pathway Questions Analytics - Quick Config Check{Colors.END}")
    print("=" * 60)
    
    all_issues = []
    
    # Basic configuration check
    print("\n📋 Checking basic configuration...")
    config_issues = check_basic_config()
    all_issues.extend(config_issues)
    
    # Python environment check
    print("\n🐍 Checking Python environment...")
    python_issues = check_python_environment()
    all_issues.extend(python_issues)
    
    # System requirements check
    print("\n🖥️ Checking system requirements...")
    system_issues = check_system_requirements()
    all_issues.extend(system_issues)
    
    # Summary
    print("\n" + "=" * 60)
    
    if not all_issues:
        print_status("✨ All basic checks passed!", "success")
        print("\n📋 Next steps:")
        print("1. Run: make init-db          # Initialize database")
        print("2. Run: make run-db           # Start the application")
        print("3. Use Developer Mode to sync data and run analysis")
        return True
    else:
        print_status(f"Found {len(all_issues)} issues to address:", "warning")
        for issue in all_issues:
            print(f"  {issue}")
        
        print("\n🛠️ How to fix:")
        if any("not set" in issue for issue in all_issues):
            print("• Edit .env file with your actual configuration values")
        if any("not installed" in issue for issue in all_issues):
            print("• Run: make install (or pip install -r requirements.txt)")
        if any("not found" in issue for issue in all_issues):
            print("• Install missing system dependencies")
        
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)