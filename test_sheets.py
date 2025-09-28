"""
Quick test script to verify Google Sheets integration works
"""
import sys
sys.path.append('src')

from google_sheets_utils import GoogleSheetsManager

def test_sheets_manager():
    """Test basic Google Sheets manager functionality"""
    
    print("🧪 Testing Google Sheets Manager...")
    
    # Initialize manager
    manager = GoogleSheetsManager()
    
    # Test URL validation
    test_urls = [
        "https://docs.google.com/spreadsheets/d/abc123/edit",
        "https://sheets.google.com/spreadsheets/d/def456",
        "https://notgoogle.com/sheets",
        "invalid-url"
    ]
    
    print("\n📝 Testing URL validation:")
    for url in test_urls:
        is_valid = manager.is_sheets_url(url)
        print(f"  {url[:50]}... → {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Test ID extraction
    print("\n🔍 Testing ID extraction:")
    test_url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit#gid=0"
    sheet_id = manager.extract_spreadsheet_id(test_url)
    print(f"  URL: {test_url[:60]}...")
    print(f"  Extracted ID: {sheet_id}")
    
    print("\n✅ Basic tests completed!")

if __name__ == "__main__":
    test_sheets_manager()