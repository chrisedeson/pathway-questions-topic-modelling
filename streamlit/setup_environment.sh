#!/bin/bash

# ================================================================
# BYU Pathway Questions Analytics - Environment Setup Script
# ================================================================
# This script helps set up the environment for the database-backed system

set -e

echo "🚀 BYU Pathway Questions Analytics - Environment Setup"
echo "=" * 60

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_info "Creating .env file from template..."
    cp .env.template .env
    print_status ".env file created"
else
    print_warning ".env file already exists"
fi

# Check if credentials directory exists
if [ ! -d "credentials" ]; then
    print_info "Creating credentials directory..."
    mkdir -p credentials
    print_status "credentials/ directory created"
fi

# Check if logs directory exists
if [ ! -d "logs" ]; then
    print_info "Creating logs directory..."
    mkdir -p logs
    print_status "logs/ directory created"
fi

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_status "Python $python_version detected (meets minimum requirement 3.8+)"
else
    print_error "Python 3.8+ required, found $python_version"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv .venv
    print_status "Virtual environment created in .venv/"
    
    print_info "Activating virtual environment and installing dependencies..."
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    print_status "Dependencies installed"
else
    print_warning "Virtual environment already exists in .venv/"
fi

# Check PostgreSQL installation
print_info "Checking PostgreSQL installation..."
if command -v psql >/dev/null 2>&1; then
    pg_version=$(psql --version | cut -d' ' -f3)
    print_status "PostgreSQL $pg_version detected"
    
    # Check for pgvector extension
    print_info "Checking for pgvector extension..."
    if psql -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw template1; then
        print_status "PostgreSQL is accessible"
    else
        print_warning "PostgreSQL may not be running or accessible"
    fi
else
    print_error "PostgreSQL not found. Please install PostgreSQL 12+ with pgvector extension"
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib postgresql-14-pgvector"
    echo "  macOS: brew install postgresql pgvector"
    echo "  Docker: docker run -d --name pathway-postgres -e POSTGRES_PASSWORD=password -p 5432:5432 pgvector/pgvector:pg14"
fi

# Environment variables checklist
echo ""
print_info "Environment Configuration Checklist:"
echo ""

# Required variables
required_vars=(
    "DATABASE_URL:PostgreSQL connection string"
    "OPENAI_API_KEY:OpenAI API key for embeddings"
    "GOOGLE_CREDENTIALS_PATH:Path to Google service account JSON"
    "QUESTIONS_SHEET_ID:Google Sheets ID for questions data"
    "TOPICS_SHEET_ID:Google Sheets ID for topics data"
)

all_configured=true

for var_info in "${required_vars[@]}"; do
    var_name=$(echo $var_info | cut -d':' -f1)
    var_desc=$(echo $var_info | cut -d':' -f2)
    
    if grep -q "^${var_name}=.*[^=]" .env 2>/dev/null; then
        value=$(grep "^${var_name}=" .env | cut -d'=' -f2- | sed 's/^["'\'']*\(.*\)["'\'']*$/\1/')
        if [[ $value =~ ^(your_|.*_here|.*_id).*$ ]] || [ -z "$value" ]; then
            print_warning "$var_name needs to be configured ($var_desc)"
            all_configured=false
        else
            print_status "$var_name is configured"
        fi
    else
        print_error "$var_name is missing ($var_desc)"
        all_configured=false
    fi
done

echo ""

if [ "$all_configured" = true ]; then
    print_status "All required environment variables are configured!"
    
    echo ""
    print_info "Next steps:"
    echo "1. Activate virtual environment: source .venv/bin/activate"
    echo "2. Initialize database: python setup_database.py"
    echo "3. Run the application: streamlit run streamlit_app_db.py"
    echo "4. Access developer mode in sidebar to sync data and run analysis"
    
else
    print_warning "Some environment variables need configuration"
    echo ""
    print_info "Please edit .env file with your actual values:"
    echo "1. Set DATABASE_URL to your PostgreSQL connection string"
    echo "2. Add your OpenAI API key"
    echo "3. Set path to Google service account credentials file"
    echo "4. Configure Google Sheets IDs"
    echo ""
    print_info "Then run this script again to verify configuration"
fi

# Optional: Test database connection if configured
if grep -q "^DATABASE_URL=postgresql" .env 2>/dev/null; then
    database_url=$(grep "^DATABASE_URL=" .env | cut -d'=' -f2-)
    if [[ ! $database_url =~ your_ ]]; then
        print_info "Testing database connection..."
        if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from database import init_database
    init_database()
    print('Database connection successful!')
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
            print_status "Database connection test passed!"
        else
            print_warning "Database connection test failed (this is normal if DB isn't set up yet)"
        fi
    fi
fi

echo ""
print_info "Setup script completed!"
echo ""