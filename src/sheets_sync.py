"""
Google Sheets synchronization service with scheduled updates
"""
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import gspread
from google.oauth2.service_account import Credentials

from data_service import get_data_service
from data_cleaning import QuestionCleaner
from config import GOOGLE_SHEETS_CREDENTIALS_PATH

logger = logging.getLogger(__name__)

class GoogleSheetsSync:
    """Handle Google Sheets synchronization"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize Google Sheets synchronization"""
        self.credentials_path = credentials_path or GOOGLE_SHEETS_CREDENTIALS_PATH
        self.client = None
        self.data_service = get_data_service()
        self.cleaner = QuestionCleaner()
        self.scheduler = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Sheets client"""
        try:
            if not os.path.exists(self.credentials_path):
                logger.error(f"Google Sheets credentials not found: {self.credentials_path}")
                return
            
            # Set up credentials
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_file(
                self.credentials_path, scopes=scope
            )
            
            self.client = gspread.authorize(credentials)
            logger.info("Google Sheets client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets client: {e}")
            self.client = None
    
    def test_connection(self, sheet_id: str) -> bool:
        """Test connection to a specific Google Sheet"""
        try:
            if not self.client:
                return False
            
            sheet = self.client.open_by_key(sheet_id)
            worksheet = sheet.get_worksheet(0)  # Get first worksheet
            
            # Try to read first cell
            worksheet.cell(1, 1)
            return True
            
        except Exception as e:
            logger.error(f"Google Sheets connection test failed: {e}")
            return False
    
    def fetch_questions_data(self, sheet_id: str) -> pd.DataFrame:
        """Fetch questions data from Google Sheets"""
        try:
            if not self.client:
                raise Exception("Google Sheets client not initialized")
            
            sheet = self.client.open_by_key(sheet_id)
            worksheet = sheet.get_worksheet(0)
            
            # Get all records
            records = worksheet.get_all_records()
            
            if not records:
                logger.warning("No data found in Google Sheets")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df)} records from Google Sheets")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch questions data: {e}")
            raise
    
    def fetch_topics_data(self, sheet_id: str) -> pd.DataFrame:
        """Fetch topics/subtopics data from Google Sheets"""
        try:
            if not self.client:
                raise Exception("Google Sheets client not initialized")
            
            sheet = self.client.open_by_key(sheet_id)
            worksheet = sheet.get_worksheet(0)
            
            records = worksheet.get_all_records()
            
            if not records:
                logger.warning("No topics data found in Google Sheets")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df)} topic records from Google Sheets")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch topics data: {e}")
            raise
    
    def upload_questions_to_sheets(self, df: pd.DataFrame, sheet_id: str, 
                                 append: bool = True) -> bool:
        """
        Upload questions DataFrame to Google Sheets
        
        Args:
            df: DataFrame with questions
            sheet_id: Google Sheets ID
            append: If True, append to existing data; if False, replace
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                raise Exception("Google Sheets client not initialized")
            
            sheet = self.client.open_by_key(sheet_id)
            worksheet = sheet.get_worksheet(0)
            
            if append:
                # Get existing data to check for duplicates and find last timestamp
                existing_data = worksheet.get_all_records()
                
                if existing_data:
                    existing_df = pd.DataFrame(existing_data)
                    
                    # Find the last timestamp in existing data
                    if 'timestamp' in existing_df.columns:
                        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], errors='coerce')
                        last_timestamp = existing_df['timestamp'].max()
                        
                        # Filter new data to only include records after last timestamp
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        new_data = df[df['timestamp'] > last_timestamp]
                        
                        if len(new_data) == 0:
                            logger.info("No new data to upload")
                            return True
                        
                        logger.info(f"Uploading {len(new_data)} new records after {last_timestamp}")
                        df = new_data
                
                # Append the new data
                values = [df.columns.values.tolist()] + df.values.tolist()
                worksheet.append_rows(values)
                
            else:
                # Replace all data
                worksheet.clear()
                values = [df.columns.values.tolist()] + df.values.tolist()
                worksheet.update('A1', values)
            
            logger.info(f"Successfully uploaded {len(df)} records to Google Sheets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload to Google Sheets: {e}")
            return False
    
    def sync_questions_from_sheets(self, sheet_id: str) -> Tuple[bool, str, Dict[str, int]]:
        """
        Sync questions from Google Sheets to database
        
        Returns:
            Tuple of (success, message, stats)
        """
        try:
            # Fetch data from sheets
            raw_df = self.fetch_questions_data(sheet_id)
            
            if raw_df.empty:
                return False, "No data found in Google Sheets", {}
            
            # Clean the data
            cleaned_df = self.cleaner.clean_questions_dataframe(raw_df)
            
            # Store in database
            added, updated, skipped = self.data_service.store_questions(
                cleaned_df, 
                source=f"google_sheets_{sheet_id}"
            )
            
            stats = {
                'added': added,
                'updated': updated,
                'skipped': skipped,
                'total_processed': len(cleaned_df)
            }
            
            message = f"Sync completed: {added} added, {updated} updated, {skipped} skipped"
            logger.info(message)
            
            return True, message, stats
            
        except Exception as e:
            error_msg = f"Sync failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {}
    
    def upload_file_to_sheets(self, file_path: str, sheet_id: str) -> Tuple[bool, str, Dict[str, int]]:
        """
        Upload a local CSV file to Google Sheets after cleaning
        
        Args:
            file_path: Path to CSV file
            sheet_id: Google Sheets ID to upload to
            
        Returns:
            Tuple of (success, message, stats)
        """
        try:
            # Read and clean the file
            raw_df = pd.read_csv(file_path)
            cleaned_df = self.cleaner.clean_questions_dataframe(raw_df)
            
            # Upload to Google Sheets
            success = self.upload_questions_to_sheets(cleaned_df, sheet_id, append=True)
            
            if success:
                # Also store in database
                added, updated, skipped = self.data_service.store_questions(
                    cleaned_df,
                    source=f"file_upload_{file_path}"
                )
                
                stats = {
                    'added': added,
                    'updated': updated,
                    'skipped': skipped,
                    'total_processed': len(cleaned_df),
                    'uploaded_to_sheets': len(cleaned_df)
                }
                
                message = f"File uploaded: {len(cleaned_df)} records to sheets, {added} added to DB"
                return True, message, stats
            else:
                return False, "Failed to upload to Google Sheets", {}
                
        except Exception as e:
            error_msg = f"File upload failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {}
    
    def start_scheduled_sync(self, questions_sheet_id: str, 
                           interval_minutes: int = 10):
        """
        Start scheduled synchronization from Google Sheets
        
        Args:
            questions_sheet_id: ID of the questions Google Sheet
            interval_minutes: Sync interval in minutes
        """
        if self.scheduler and self.scheduler.running:
            logger.warning("Scheduler is already running")
            return
        
        self.scheduler = BackgroundScheduler()
        
        # Add sync job
        self.scheduler.add_job(
            func=self._scheduled_sync_job,
            trigger=IntervalTrigger(minutes=interval_minutes),
            args=[questions_sheet_id],
            id='questions_sync',
            name='Questions Google Sheets Sync',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info(f"Started scheduled sync every {interval_minutes} minutes")
    
    def stop_scheduled_sync(self):
        """Stop scheduled synchronization"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Stopped scheduled sync")
    
    def _scheduled_sync_job(self, sheet_id: str):
        """Background job for scheduled sync"""
        try:
            success, message, stats = self.sync_questions_from_sheets(sheet_id)
            
            if success and stats.get('added', 0) > 0:
                logger.info(f"Scheduled sync completed: {message}")
            elif success:
                logger.debug(f"Scheduled sync - no new data: {message}")
            else:
                logger.error(f"Scheduled sync failed: {message}")
                
        except Exception as e:
            logger.error(f"Scheduled sync job failed: {e}")
    
    def get_sync_schedule_info(self) -> Dict[str, Any]:
        """Get information about the sync schedule"""
        if not self.scheduler:
            return {'status': 'not_configured'}
        
        if not self.scheduler.running:
            return {'status': 'stopped'}
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'status': 'running',
            'jobs': jobs
        }


class GoogleSheetsManager:
    """Main manager for Google Sheets operations"""
    
    def __init__(self):
        self.sync = GoogleSheetsSync()
        self._questions_sheet_id = None
        self._topics_sheet_id = None
        self._sync_interval = 10  # minutes
    
    def set_sheet_ids(self, questions_sheet_id: str, topics_sheet_id: str):
        """Set the Google Sheets IDs"""
        self._questions_sheet_id = questions_sheet_id
        self._topics_sheet_id = topics_sheet_id
        logger.info(f"Set sheet IDs: questions={questions_sheet_id}, topics={topics_sheet_id}")
    
    def set_sync_interval(self, minutes: int):
        """Set the sync interval in minutes"""
        self._sync_interval = minutes
        
        # Restart scheduler with new interval if it's running
        if self.sync.scheduler and self.sync.scheduler.running:
            self.sync.stop_scheduled_sync()
            if self._questions_sheet_id:
                self.sync.start_scheduled_sync(self._questions_sheet_id, minutes)
    
    def start_sync(self) -> bool:
        """Start scheduled synchronization"""
        if not self._questions_sheet_id:
            logger.error("Questions sheet ID not set")
            return False
        
        try:
            self.sync.start_scheduled_sync(self._questions_sheet_id, self._sync_interval)
            return True
        except Exception as e:
            logger.error(f"Failed to start sync: {e}")
            return False
    
    def stop_sync(self):
        """Stop scheduled synchronization"""
        self.sync.stop_scheduled_sync()
    
    def manual_sync(self) -> Tuple[bool, str, Dict[str, int]]:
        """Manually trigger synchronization"""
        if not self._questions_sheet_id:
            return False, "Questions sheet ID not set", {}
        
        return self.sync.sync_questions_from_sheets(self._questions_sheet_id)
    
    def upload_file(self, file_path: str) -> Tuple[bool, str, Dict[str, int]]:
        """Upload a file to Google Sheets"""
        if not self._questions_sheet_id:
            return False, "Questions sheet ID not set", {}
        
        return self.sync.upload_file_to_sheets(file_path, self._questions_sheet_id)
    
    def test_connections(self) -> Dict[str, bool]:
        """Test connections to configured sheets"""
        results = {}
        
        if self._questions_sheet_id:
            results['questions'] = self.sync.test_connection(self._questions_sheet_id)
        
        if self._topics_sheet_id:
            results['topics'] = self.sync.test_connection(self._topics_sheet_id)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of Google Sheets integration"""
        return {
            'configured_sheets': {
                'questions': self._questions_sheet_id,
                'topics': self._topics_sheet_id
            },
            'sync_interval_minutes': self._sync_interval,
            'schedule_info': self.sync.get_sync_schedule_info(),
            'connection_tests': self.test_connections(),
            'client_initialized': self.sync.client is not None
        }


# Global manager instance
_sheets_manager = None

def get_sheets_manager() -> GoogleSheetsManager:
    """Get global Google Sheets manager instance"""
    global _sheets_manager
    if _sheets_manager is None:
        _sheets_manager = GoogleSheetsManager()
    return _sheets_manager