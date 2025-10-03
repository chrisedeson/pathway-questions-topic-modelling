"""
Migration script to transfer data from file-based system to database-backed system
Handles existing CSV files, cache files, and preserves analysis history
"""

import os
import sys
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
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

def migrate_csv_data():
    """Migrate existing CSV files to database"""
    print("📊 Migrating CSV data to database...")
    
    try:
        from data_service import get_data_service
        from data_cleaning import QuestionCleaner
        
        data_service = get_data_service()
        cleaner = QuestionCleaner()
        
        # Look for existing CSV files
        csv_files = [
            'Questions.csv',
            'data/processed_questions.csv'
        ]
        
        questions_migrated = 0
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                print(f"📁 Processing {csv_file}...")
                
                try:
                    df = pd.read_csv(csv_file)
                    print(f"   Found {len(df)} rows")
                    
                    # Clean and prepare data
                    cleaned_questions = []
                    for _, row in df.iterrows():
                        try:
                            # Extract question text
                            if 'Question' in row:
                                question_text = row['Question']
                            elif 'question' in row:
                                question_text = row['question']
                            else:
                                # Try to extract from kwargs if present
                                kwargs_text = row.get('kwargs', '{}')
                                question_text = cleaner.extract_question_from_kwargs(kwargs_text)
                            
                            if question_text and len(question_text.strip()) > 10:
                                question_data = {
                                    'original_text': question_text,
                                    'cleaned_text': question_text.strip(),
                                    'language': cleaner.detect_language(question_text),
                                    'source': 'csv_migration',
                                    'metadata': {
                                        'migrated_from': csv_file,
                                        'original_row': dict(row)
                                    }
                                }
                                
                                # Add timestamp if available
                                if 'timestamp' in row:
                                    question_data['timestamp'] = pd.to_datetime(row['timestamp'])
                                elif 'created_at' in row:
                                    question_data['timestamp'] = pd.to_datetime(row['created_at'])
                                else:
                                    question_data['timestamp'] = datetime.now()
                                
                                cleaned_questions.append(question_data)
                                
                        except Exception as e:
                            print(f"   ⚠️ Skipped row due to error: {e}")
                            continue
                    
                    # Store in database
                    if cleaned_questions:
                        stored_count = data_service.store_questions(cleaned_questions)
                        print(f"   ✅ Migrated {stored_count} questions to database")
                        questions_migrated += stored_count
                    
                except Exception as e:
                    print(f"   ❌ Failed to process {csv_file}: {e}")
                    continue
        
        print(f"✅ Total questions migrated: {questions_migrated}")
        return questions_migrated > 0
        
    except Exception as e:
        print(f"❌ CSV migration failed: {e}")
        return False

def migrate_cache_files():
    """Migrate existing analysis cache files"""
    print("💾 Migrating cache files...")
    
    try:
        from data_service import get_data_service
        
        data_service = get_data_service()
        cache_dirs = ['cache/', 'embeddings_cache/']
        
        migrated_files = 0
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                print(f"📁 Processing {cache_dir}...")
                
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        if file.endswith('.pkl'):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'rb') as f:
                                    cache_data = pickle.load(f)
                                
                                # Create cache entry in database
                                cache_key = f"migration_{file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                
                                cache_entry = {
                                    'cache_key': cache_key,
                                    'cache_type': 'migrated_analysis',
                                    'data': cache_data,
                                    'metadata': {
                                        'source_file': file_path,
                                        'migration_timestamp': datetime.now().isoformat()
                                    }
                                }
                                
                                data_service.store_cache_result(cache_entry)
                                migrated_files += 1
                                print(f"   ✅ Migrated {file}")
                                
                            except Exception as e:
                                print(f"   ⚠️ Failed to migrate {file}: {e}")
                                continue
        
        print(f"✅ Cache files migrated: {migrated_files}")
        return migrated_files > 0
        
    except Exception as e:
        print(f"❌ Cache migration failed: {e}")
        return False

def migrate_results_files():
    """Migrate existing results files"""
    print("📈 Migrating results files...")
    
    try:
        results_dir = 'results/'
        if not os.path.exists(results_dir):
            print("   No results directory found")
            return True
        
        from data_service import get_data_service
        data_service = get_data_service()
        
        migrated_results = 0
        
        for file in os.listdir(results_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(results_dir, file)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Determine result type from filename
                    if 'topics' in file.lower():
                        result_type = 'topic_analysis'
                    elif 'similar' in file.lower():
                        result_type = 'similar_questions'
                    elif 'review' in file.lower():
                        result_type = 'question_review'
                    else:
                        result_type = 'unknown'
                    
                    # Extract timestamp from filename
                    timestamp_str = file.split('_')[-1].replace('.csv', '')
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    except:
                        timestamp = datetime.now()
                    
                    # Store as cache entry
                    cache_key = f"migrated_{result_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                    
                    cache_entry = {
                        'cache_key': cache_key,
                        'cache_type': result_type,
                        'data': df.to_dict('records'),
                        'metadata': {
                            'source_file': file_path,
                            'result_type': result_type,
                            'migration_timestamp': datetime.now().isoformat(),
                            'original_timestamp': timestamp.isoformat()
                        }
                    }
                    
                    data_service.store_cache_result(cache_entry)
                    migrated_results += 1
                    print(f"   ✅ Migrated {file}")
                    
                except Exception as e:
                    print(f"   ⚠️ Failed to migrate {file}: {e}")
                    continue
        
        print(f"✅ Results files migrated: {migrated_results}")
        return True
        
    except Exception as e:
        print(f"❌ Results migration failed: {e}")
        return False

def create_backup():
    """Create backup of existing files before migration"""
    print("💾 Creating backup of existing files...")
    
    try:
        import shutil
        from datetime import datetime
        
        backup_dir = f"backup_pre_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Files and directories to backup
        backup_items = [
            'Questions.csv',
            'data/',
            'cache/',
            'embeddings_cache/',
            'results/',
            'streamlit_app.py',
            'streamlit_app_old.py'
        ]
        
        backed_up = 0
        for item in backup_items:
            if os.path.exists(item):
                try:
                    if os.path.isfile(item):
                        shutil.copy2(item, backup_dir)
                    else:
                        shutil.copytree(item, os.path.join(backup_dir, item))
                    backed_up += 1
                    print(f"   ✅ Backed up {item}")
                except Exception as e:
                    print(f"   ⚠️ Failed to backup {item}: {e}")
        
        print(f"✅ Backup created in {backup_dir} ({backed_up} items)")
        return backup_dir
        
    except Exception as e:
        print(f"❌ Backup creation failed: {e}")
        return None

def verify_migration():
    """Verify that migration was successful"""
    print("🔍 Verifying migration...")
    
    try:
        from data_service import get_data_service
        data_service = get_data_service()
        
        # Check question count
        questions_count = data_service.get_questions_count()
        print(f"   📊 Questions in database: {questions_count}")
        
        # Check if analysis can run
        if questions_count > 0:
            print("   ✅ Database has questions ready for analysis")
        else:
            print("   ⚠️ No questions found in database")
            return False
        
        # Test dashboard data
        try:
            dashboard_data = data_service.get_dashboard_data()
            if dashboard_data:
                print("   ✅ Dashboard data accessible")
            else:
                print("   ℹ️ No dashboard data yet (run analysis to generate)")
        except Exception as e:
            print(f"   ℹ️ Dashboard data not available: {e}")
        
        print("✅ Migration verification completed")
        return True
        
    except Exception as e:
        print(f"❌ Migration verification failed: {e}")
        return False

def main():
    """Main migration process"""
    print("🚀 BYU Pathway Questions Analytics - Data Migration")
    print("=" * 60)
    print("This script will migrate your existing file-based data to the new database system.")
    print()
    
    # Check if database is initialized
    try:
        from database import get_db_manager
        db_manager = get_db_manager()
        if not db_manager.test_connection():
            print("❌ Database not accessible. Please run setup_database.py first.")
            return False
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        print("Please run setup_database.py first.")
        return False
    
    # Create backup
    backup_dir = create_backup()
    if not backup_dir:
        print("❌ Failed to create backup. Migration aborted.")
        return False
    
    print()
    
    # Run migrations
    success_csv = migrate_csv_data()
    print()
    
    success_cache = migrate_cache_files()
    print()
    
    success_results = migrate_results_files()
    print()
    
    # Verify migration
    success_verify = verify_migration()
    
    print("\n" + "=" * 60)
    
    if success_csv or success_cache or success_results:
        print("🎉 Migration completed successfully!")
        print(f"📁 Backup created in: {backup_dir}")
        print("\n📋 Next steps:")
        print("1. Run the new database app: streamlit run streamlit_app_db.py")
        print("2. Use Developer Mode to sync fresh data from Google Sheets")
        print("3. Run analysis to generate new topics and insights")
        print("4. Verify dashboard displays correctly")
        print("\n⚠️ Keep the backup until you're satisfied with the migration")
    else:
        print("⚠️ Migration completed with no data transferred")
        print("This might be normal if no compatible files were found")
        print("You can proceed with the new database system")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)