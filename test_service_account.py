"""
Test the Google Sheets integration with the service account
"""
import sys
import os
sys.path.append('src')

from google_sheets_utils import GoogleSheetsManager, SheetsPermission
import pandas as pd

def test_service_account():
    """Test the service account integration"""
    
    print("🔑 Testing Service Account Integration...")
    
    # Check if credentials exist
    credentials_path = "credentials/byu-pathway-chatbot-service-account.json"
    if not os.path.exists(credentials_path):
        print("❌ Credentials file not found!")
        return
        
    print("✅ Credentials file found")
    
    # Test manager initialization
    manager = GoogleSheetsManager()
    
    print("\n📋 Service Account Details:")
    print("   Email: streamlit-sheets-reader@byu-pathway-chatbot.iam.gserviceaccount.com")
    print("   Project: byu-pathway-chatbot")
    
    # Test URL validation
    test_url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit"
    is_valid = manager.is_sheets_url(test_url)
    print(f"\n🔗 URL Validation Test: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Test ID extraction
    sheet_id = manager.extract_spreadsheet_id(test_url)
    print(f"📄 Sheet ID: {sheet_id}")
    
    print("\n✅ Basic integration test completed!")
    print("\n💡 To test with your actual Google Sheet:")
    print("   1. Share your sheet with: streamlit-sheets-reader@byu-pathway-chatbot.iam.gserviceaccount.com")
    print("   2. Give 'Editor' permissions for full access")
    print("   3. Use the Streamlit app to connect")

if __name__ == "__main__":
    test_service_account()