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

from config import GOOGLE_SHEETS_CREDENTIALS_PATH

logger = logging.getLogger(__name__)

class SheetsPermission:
    """Enum for Google Sheets permissions"""
    READ_ONLY = "read"
    EDIT = "edit" 
    NO_ACCESS = "none"

class GoogleSheetsManager:
    """Manages Google Sheets integration with error handling and permission checking"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize Google Sheets manager"""
        self.credentials_path = credentials_path or GOOGLE_SHEETS_CREDENTIALS_PATH
        self.client = None
        self.last_error = None
        
    def _initialize_client(self) -> bool:
        """Initialize Google Sheets client with error handling"""
        try:
            if not Path(self.credentials_path).exists():
                self.last_error = f"Credentials file not found: {self.credentials_path}"
                return False
                
            # Use service account credentials
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
                
                # Test write access by attempting a safe write operation
                try:
                    original_value = test_read.value if test_read.value else ""
                    self._make_api_call(worksheet.update_cell, 1, 1, "TEMP_PERMISSION_CHECK")
                    self._make_api_call(worksheet.update_cell, 1, 1, original_value)
                    return SheetsPermission.EDIT, None
                    
                except APIError as e:
                    if "insufficientPermissions" in str(e) or "forbidden" in str(e).lower():
                        return SheetsPermission.READ_ONLY, None
                    raise e
                    
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
        Read topics, subtopics, and questions from Google Sheet
        
        Expected format:
        | Topic | Subtopic | Question |
        |-------|----------|----------|
        | ...   | ...      | ...      |
        
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: DataFrame and error message if any
        """
        if not self.client and not self._initialize_client():
            return None, self.last_error
            
        try:
            spreadsheet_id = self.extract_spreadsheet_id(sheet_url)
            if not spreadsheet_id:
                return None, "Invalid Google Sheets URL"
            
            sheet = self._make_api_call(self.client.open_by_key, spreadsheet_id)
            
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
            
            # Validate expected columns
            required_columns = ['Topic', 'Subtopic', 'Question']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return None, f"Missing required columns: {missing_columns}. Found columns: {list(df.columns)}"
            
            return df, None
            
        except Exception as e:
            error_msg = f"Error reading from Google Sheet: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def write_topics_to_sheet(self, sheet_url: str, df: pd.DataFrame, worksheet_name: str = None) -> Optional[str]:
        """
        Write topics DataFrame back to Google Sheet
        
        Returns:
            Optional[str]: Error message if any, None if successful
        """
        if not self.client and not self._initialize_client():
            return self.last_error
            
        try:
            spreadsheet_id = self.extract_spreadsheet_id(sheet_url)
            if not spreadsheet_id:
                return "Invalid Google Sheets URL"
            
            sheet = self._make_api_call(self.client.open_by_key, spreadsheet_id)
            
            if worksheet_name:
                try:
                    worksheet = sheet.worksheet(worksheet_name)
                except WorksheetNotFound:
                    return f"Worksheet '{worksheet_name}' not found"
            else:
                worksheet = sheet.sheet1
            
            # Clear existing content
            self._make_api_call(worksheet.clear)
            
            # Prepare data with headers
            values = [df.columns.tolist()] + df.values.tolist()
            
            # Update sheet with new data
            self._make_api_call(worksheet.update, 'A1', values)
            
            return None  # Success
            
        except Exception as e:
            error_msg = f"Error writing to Google Sheet: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def append_to_sheet(self, sheet_url: str, data: List[List], worksheet_name: str = None) -> Optional[str]:
        """
        Append rows to Google Sheet
        
        Returns:
            Optional[str]: Error message if any, None if successful
        """
        if not self.client and not self._initialize_client():
            return self.last_error
            
        try:
            spreadsheet_id = self.extract_spreadsheet_id(sheet_url)
            if not spreadsheet_id:
                return "Invalid Google Sheets URL"
            
            sheet = self._make_api_call(self.client.open_by_key, spreadsheet_id)
            
            if worksheet_name:
                try:
                    worksheet = sheet.worksheet(worksheet_name)
                except WorksheetNotFound:
                    return f"Worksheet '{worksheet_name}' not found"
            else:
                worksheet = sheet.sheet1
            
            # Append rows
            for row in data:
                self._make_api_call(worksheet.append_row, row)
            
            return None  # Success
            
        except Exception as e:
            error_msg = f"Error appending to Google Sheet: {str(e)}"
            logger.error(error_msg)
            return error_msg

def display_sheets_permission_status(permission: SheetsPermission, error_msg: str = None):
    """Display Google Sheets permission status in Streamlit UI"""
    if permission == SheetsPermission.EDIT:
        st.success("✅ **Full Access**: Can read and edit Google Sheet")
    elif permission == SheetsPermission.READ_ONLY:
        st.warning("⚠️ **Read-Only Access**: Can view but cannot edit Google Sheet")
        st.info("💡 **Tip**: To enable editing, share the sheet with your service account email with 'Editor' permissions")
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
            5. **Set permission level**:
               - **Viewer** = Read-only access
               - **Editor** = Full read/write access
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