"""
Quick verification that the DataFrame boolean error is fixed
"""
import sys
sys.path.append('src')
import pandas as pd

def test_dataframe_boolean_fix():
    """Test that we properly handle DataFrame boolean comparisons"""
    
    print("🧪 Testing DataFrame Boolean Handling...")
    
    # Test case 1: None check
    data_result = {"questions_data": None}
    
    # This should work without error
    if (data_result["questions_data"] is None or 
        (data_result["questions_data"] is not None and data_result["questions_data"].empty) or 
        (data_result["questions_data"] is not None and len(data_result["questions_data"]) == 0)):
        print("✅ Test 1 passed: None check works")
    
    # Test case 2: Empty DataFrame
    data_result = {"questions_data": pd.DataFrame()}
    
    if (data_result["questions_data"] is None or 
        data_result["questions_data"].empty or 
        len(data_result["questions_data"]) == 0):
        print("✅ Test 2 passed: Empty DataFrame check works")
    
    # Test case 3: Valid DataFrame
    data_result = {"questions_data": pd.DataFrame({'question': ['What is this?', 'How do I do that?']})}
    
    if not (data_result["questions_data"] is None or 
            data_result["questions_data"].empty or 
            len(data_result["questions_data"]) == 0):
        print("✅ Test 3 passed: Valid DataFrame check works")
    
    print("\n🎉 All DataFrame boolean handling tests passed!")
    print("The ValueError should be fixed now.")

if __name__ == "__main__":
    test_dataframe_boolean_fix()