"""
Google Sheets integration utilities for BYU Pathway Hybrid Topic Analysis
"""
import streamlit as st
import gspread
import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any, List
from google.auth.exceptions import RefreshError
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
import time
import backoff
from pathlib import Path
import os
import json
from google.oauth2.service_account import Credentials

from config import GOOGLE_SHEETS_CREDENTIALS_PATH

logger = logging.getLogger(__name__)

class SheetsPermission:
    """Enum for Google Sheets permission levels"""
    NO_ACCESS = "no_access"
    READ_ONLY = "read_only"


class GoogleSheetsManager:
    """Manages Google Sheets integration with error handling and permission checking"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize Google Sheets manager"""
        self.credentials_path = credentials_path or GOOGLE_SHEETS_CREDENTIALS_PATH
        self.client = None
        self.last_error = None
    
    def _get_credentials_from_env(self) -> Optional[Dict[str, Any]]:
        """
        Construct Google Service Account credentials from environment variables.
        This is used for Streamlit Cloud deployment where JSON files can't be uploaded.
        
        Returns:
            Optional[Dict[str, Any]]: Credentials dictionary or None if env vars not set
        """
        try:
            # Check if all required environment variables are present
            required_vars = [
                'GOOGLE_SERVICE_ACCOUNT_TYPE',
                'GOOGLE_SERVICE_ACCOUNT_PROJECT_ID',
                'GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID',
                'GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY',
                'GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL',
                'GOOGLE_SERVICE_ACCOUNT_CLIENT_ID'
            ]
            
            if not all(os.getenv(var) for var in required_vars):
                return None
            
            # Construct credentials dict from environment variables
            credentials_dict = {
                "type": os.getenv('GOOGLE_SERVICE_ACCOUNT_TYPE'),
                "project_id": os.getenv('GOOGLE_SERVICE_ACCOUNT_PROJECT_ID'),
                "private_key_id": os.getenv('GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID'),
                "private_key": os.getenv('GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY', '').replace('\\n', '\n'),  # Handle escaped newlines
                "client_email": os.getenv('GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL'),
                "client_id": os.getenv('GOOGLE_SERVICE_ACCOUNT_CLIENT_ID'),
                "auth_uri": os.getenv('GOOGLE_SERVICE_ACCOUNT_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.getenv('GOOGLE_SERVICE_ACCOUNT_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                "auth_provider_x509_cert_url": os.getenv('GOOGLE_SERVICE_ACCOUNT_AUTH_PROVIDER_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
                "client_x509_cert_url": os.getenv('GOOGLE_SERVICE_ACCOUNT_CLIENT_CERT_URL', ''),
                "universe_domain": os.getenv('GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN', 'googleapis.com')
            }
            
            return credentials_dict
            
        except Exception as e:
            logger.error(f"Failed to construct credentials from environment variables: {str(e)}")
            return None
        
    def _initialize_client(self) -> bool:
        """
        Initialize Google Sheets client with error handling.
        First checks for environment variables (for Streamlit Cloud),
        then falls back to JSON file (for local development).
        """
        try:
            # Try environment variables first (Streamlit Cloud deployment)
            credentials_dict = self._get_credentials_from_env()
            
            if credentials_dict:
                logger.info("Using Google Service Account credentials from environment variables")
                scopes = [
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
                credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
                self.client = gspread.authorize(credentials)
                return True
            
            # Fall back to JSON file (local development)
            if not Path(self.credentials_path).exists():
                self.last_error = f"Credentials file not found: {self.credentials_path} and environment variables not set"
                logger.error(self.last_error)
                return False
                
            logger.info(f"Using Google Service Account credentials from file: {self.credentials_path}")
            self.client = gspread.service_account(filename=self.credentials_path)
            return True
            
        except Exception as e:
            self.last_error = f"Failed to initialize Google Sheets client: {str(e)}"
            logger.error(self.last_error)
            return False
    
    def is_sheets_url(self, url: str) -> bool:
        """Check if URL is a valid Google Sheets URL"""
        sheets_patterns = [
            "docs.google.com/spreadsheets",
            "sheets.google.com"
        ]
        return any(pattern in url for pattern in sheets_patterns)
    
    def extract_spreadsheet_id(self, url: str) -> Optional[str]:
        """Extract spreadsheet ID from Google Sheets URL"""
        try:
            if "/d/" in url:
                return url.split("/d/")[1].split("/")[0]
            return None
        except Exception:
            return None
    
    @backoff.on_exception(
        backoff.expo,
        (APIError, RefreshError),
        max_tries=3,
        base=2,
        max_value=60
    )
    def _make_api_call(self, func, *args, **kwargs):
        """Make API call with retry logic"""
        return func(*args, **kwargs)
    
    def check_sheet_permissions(self, sheet_url: str) -> Tuple[SheetsPermission, Optional[str]]:
        """
        Check what permissions the service account has for a Google Sheet
        
        Returns:
            Tuple[SheetsPermission, Optional[str]]: Permission level and error message if any
        """
        if not self.client and not self._initialize_client():
            return SheetsPermission.NO_ACCESS, self.last_error
            
        try:
            spreadsheet_id = self.extract_spreadsheet_id(sheet_url)
            if not spreadsheet_id:
                return SheetsPermission.NO_ACCESS, "Invalid Google Sheets URL"
            
            # Try to open the spreadsheet
            sheet = self._make_api_call(self.client.open_by_key, spreadsheet_id)
            worksheet = sheet.sheet1
            
            # Test read access
            try:
                test_read = self._make_api_call(worksheet.cell, 1, 1)
                return SheetsPermission.READ_ONLY, None
                    
            except APIError as e:
                if "insufficientPermissions" in str(e) or "forbidden" in str(e).lower():
                    return SheetsPermission.NO_ACCESS, "No read permissions for this sheet"
                raise e
                
        except SpreadsheetNotFound:
            return SheetsPermission.NO_ACCESS, "Spreadsheet not found or not shared with service account"
        except APIError as e:
            return SheetsPermission.NO_ACCESS, f"API Error: {str(e)}"
        except Exception as e:
            return SheetsPermission.NO_ACCESS, f"Unexpected error: {str(e)}"
    
    def read_topics_from_sheet(self, sheet_url: str, worksheet_name: str = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Read topics data from Google Sheet with Topic, Subtopic, Question columns
        
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (DataFrame, error_message)
        """
        if not self.client and not self._initialize_client():
            return None, self.last_error
            
        try:
            spreadsheet_id = self.extract_spreadsheet_id(sheet_url)
            if not spreadsheet_id:
                return None, "Invalid Google Sheets URL"
            
            sheet = self._make_api_call(self.client.open_by_key, spreadsheet_id)
            
            # Get specific worksheet if specified
            if worksheet_name:
                try:
                    worksheet = sheet.worksheet(worksheet_name)
                except WorksheetNotFound:
                    return None, f"Worksheet '{worksheet_name}' not found"
            else:
                worksheet = sheet.sheet1
            
            # Get all values
            data = self._make_api_call(worksheet.get_all_values)
            
            if not data or len(data) < 2:
                return None, "Sheet is empty or has no data rows"
            
            # Convert to DataFrame
            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
            
            # Clean up the DataFrame
            df = df.dropna(how='all')  # Remove completely empty rows
            df = df.fillna('')  # Replace NaN with empty string
            
            # Validate expected columns for topics
            required_columns = ['Topic', 'Subtopic', 'Question']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return None, f"Missing required columns: {missing_columns}. Found columns: {list(df.columns)}"
            
            return df, None
            
        except Exception as e:
            error_msg = f"Error reading from Google Sheet: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def read_questions_from_sheet(self, sheet_url: str, worksheet_name: str = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Read questions data from Google Sheet - expects either:
        1. Simple list in first column
        2. CSV with 'question' column
        
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (DataFrame, error_message)
        """
        if not self.client and not self._initialize_client():
            return None, self.last_error
            
        try:
            spreadsheet_id = self.extract_spreadsheet_id(sheet_url)
            if not spreadsheet_id:
                return None, "Invalid Google Sheets URL"
            
            sheet = self._make_api_call(self.client.open_by_key, spreadsheet_id)
            
            # Get specific worksheet if specified
            if worksheet_name:
                try:
                    worksheet = sheet.worksheet(worksheet_name)
                except WorksheetNotFound:
                    return None, f"Worksheet '{worksheet_name}' not found"
            else:
                worksheet = sheet.sheet1
            
            # Get all values
            data = self._make_api_call(worksheet.get_all_values)
            
            if not data:
                return None, "Sheet is empty"
            
            # Convert to DataFrame
            if len(data) == 1 or not data[0]:  # No headers or single row
                # Treat as simple list
                questions = [row[0] for row in data if row and row[0].strip()]
                df = pd.DataFrame({'question': questions})
            else:
                # Try to parse as CSV with headers
                headers = data[0]
                rows = data[1:] if len(data) > 1 else []
                df = pd.DataFrame(rows, columns=headers)
                
                # Clean up the DataFrame
                df = df.dropna(how='all')  # Remove completely empty rows
                df = df.fillna('')  # Replace NaN with empty string
                
                # Check if we have a 'question' column
                if 'question' in df.columns:
                    # Use existing question column
                    df = df[['question']].copy()
                elif len(df.columns) >= 1:
                    # Use first column as questions
                    df = pd.DataFrame({'question': df.iloc[:, 0]})
                else:
                    return None, "No question data found in sheet"
            
            # Remove empty questions
            df = df[df['question'].notna() & (df['question'].str.strip() != '')]
            
            if len(df) == 0:
                return None, "No valid questions found in sheet"
            
            return df, None
            
        except Exception as e:
            error_msg = f"Error reading questions from Google Sheet: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    

    


def display_sheets_permission_status(permission: SheetsPermission, error_msg: str = None):
    """Display Google Sheets permission status in Streamlit UI"""
    if permission == SheetsPermission.READ_ONLY:
        st.success("✅ **Read Access**: Can view Google Sheet")
    else:
        st.error("❌ **No Access**: Cannot access Google Sheet")
        if error_msg:
            st.error(f"**Error**: {error_msg}")
        
        with st.expander("🔧 **How to Fix Access Issues**", expanded=False):
            st.markdown("""
            **To grant access to your Google Sheet:**
            
            1. **Find your service account email** in your JSON credentials file (usually ends with `.iam.gserviceaccount.com`)
            2. **Open your Google Sheet** in a browser
            3. **Click 'Share'** button (top right)
            4. **Add the service account email** 
            5. **Set permission level to Viewer** for read access
            6. **Click 'Send'**
            
            **Alternative**: Make sheet public by clicking 'Change to anyone with the link' and selecting appropriate access level.
            """)

def create_sheets_connection_ui() -> Tuple[Optional[str], Optional[str]]:
    """
    Create UI components for Google Sheets connection
    
    Returns:
        Tuple[Optional[str], Optional[str]]: Sheet URL and worksheet name
    """
    st.subheader("🔗 Google Sheets Connection")
    
    with st.expander("📋 **Google Sheets Format Requirements**", expanded=False):
        st.markdown("""
        **Your Google Sheet should have these exact column headers:**
        
        | Topic | Subtopic | Question |
        |-------|----------|----------|
        | Technical Issues | Login Problems | How do I reset my password? |
        | Academic Support | Course Materials | Where can I find my textbooks? |
        | ... | ... | ... |
        
        **Requirements:**
        - First row must contain headers: `Topic`, `Subtopic`, `Question`
        - Each row represents one question with its topic classification
        - Empty cells are allowed but avoid completely empty rows
        """)
    
    sheet_url = st.text_input(
        "🔗 **Google Sheets URL**",
        placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/edit",
        help="Paste the full URL of your Google Sheet"
    )
    
    worksheet_name = st.text_input(
        "📋 **Worksheet Name** (optional)",
        placeholder="Sheet1",
        help="Leave empty to use the first worksheet, or specify a worksheet name"
    )
    
    return sheet_url.strip() if sheet_url else None, worksheet_name.strip() if worksheet_name else None