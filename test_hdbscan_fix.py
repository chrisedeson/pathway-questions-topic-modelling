#!/usr/bin/env python3
"""
Test script to verify HDBSCAN import fix
"""

def test_hdbscan_import():
    """Test HDBSCAN import and initialization"""
    try:
        from hdbscan import HDBSCAN
        print("✅ HDBSCAN import successful")
        
        # Test initialization without prediction_data parameter
        hdbscan_model = HDBSCAN(
            min_cluster_size=5,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        print("✅ HDBSCAN initialization successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def test_sklearn_imports():
    """Test other sklearn imports"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ cosine_similarity import successful")
        
        import umap
        print("✅ UMAP import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run HDBSCAN fix validation"""
    print("🔧 HDBSCAN Import Fix Validation")
    print("=" * 40)
    print()
    
    success = True
    success &= test_hdbscan_import()
    success &= test_sklearn_imports()
    
    print()
    if success:
        print("✅ All HDBSCAN imports and initialization working correctly!")
        print("The clustering step should now work without errors.")
    else:
        print("❌ There are still import issues to resolve.")

if __name__ == "__main__":
    main()