"""
Database initialization script for BYU Pathway Questions Analysis
Creates all necessary tables and sets up initial configuration
"""

import os
import sys
from pathlib import Path
import logging

# Load environment variables from .env file
def load_environment():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env file")
        return True
    else:
        print("❌ .env file not found")
        return False

# Load environment first
if not load_environment():
    print("❌ Failed to load environment variables")
    sys.exit(1)

# Add src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

def init_database():
    """Initialize the database with all required tables"""
    print("🔧 Initializing BYU Pathway Questions Database...")
    
    try:
        from database import init_database as setup_db
        from config import DATABASE_URL
        
        print(f"📊 Connecting to database...")
        setup_db()
        print("✅ Database tables created successfully!")
        
        # Test the connection
        from database import get_db_manager
        db_manager = get_db_manager()
        
        if db_manager.test_connection():
            print("✅ Database connection test passed!")
        else:
            print("❌ Database connection test failed!")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def setup_environment_check():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    required_vars = [
        'DATABASE_URL',
        'OPENAI_API_KEY',
        'GOOGLE_CREDENTIALS_PATH'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n📝 Please set the following environment variables:")
        for var in missing_vars:
            if var == 'DATABASE_URL':
                print(f"   {var}: PostgreSQL connection string (e.g., postgresql://user:pass@localhost/dbname)")
            elif var == 'OPENAI_API_KEY':
                print(f"   {var}: Your OpenAI API key for embeddings")
            elif var == 'GOOGLE_CREDENTIALS_PATH':
                print(f"   {var}: Path to Google service account JSON file")
        return False
    
    print("✅ Environment variables configured!")
    return True

def test_integrations():
    """Test external integrations"""
    print("🧪 Testing external integrations...")
    
    try:
        # Test Google Sheets connection
        from sheets_sync import get_sheets_manager
        sheets_manager = get_sheets_manager()
        
        print("📊 Testing Google Sheets connection...")
        if sheets_manager.test_connection():
            print("✅ Google Sheets connection successful!")
        else:
            print("⚠️ Google Sheets connection failed (check credentials)")
            
    except Exception as e:
        print(f"⚠️ Google Sheets test failed: {e}")
    
    try:
        # Test OpenAI connection
        print("🤖 Testing OpenAI connection...")
        from hybrid_topic_processor import HybridTopicProcessor
        processor = HybridTopicProcessor()
        
        # Test with a simple embedding
        test_embedding = processor.embedder._get_embedding("test")
        if test_embedding and len(test_embedding) > 0:
            print("✅ OpenAI embeddings working!")
        else:
            print("⚠️ OpenAI embeddings test failed")
            
    except Exception as e:
        print(f"⚠️ OpenAI test failed: {e}")

def main():
    """Main initialization process"""
    print("🚀 BYU Pathway Questions Analytics - Database Setup")
    print("=" * 60)
    
    # Step 1: Environment check
    if not setup_environment_check():
        print("\n❌ Setup failed: Environment not properly configured")
        return False
    
    # Step 2: Database initialization
    if not init_database():
        print("\n❌ Setup failed: Database initialization failed")
        return False
    
    # Step 3: Test integrations
    test_integrations()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the Streamlit app: streamlit run streamlit_app_db.py")
    print("2. Use Developer Mode to sync data from Google Sheets")
    print("3. Run initial analysis to generate topics and insights")
    print("4. View the dashboard for instant analytics!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)