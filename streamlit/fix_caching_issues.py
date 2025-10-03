#!/usr/bin/env python3
"""
Script to analyze and fix caching issues in the BYU Pathway Topic Modeling project.
Issues found:
1. Over 2000 individual embedding cache files for only 4 analysis runs
2. Duplicate function definitions
3. Inefficient caching strategy
"""

import os
import shutil
import json
import pickle
from pathlib import Path
from typing import Dict, List

def analyze_cache_files():
    """Analyze the current cache situation"""
    cache_dir = "embeddings_cache"
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist")
        return
    
    files = os.listdir(cache_dir)
    print(f"Total cache files: {len(files)}")
    
    # Analyze file patterns
    model_counts = {}
    total_size = 0
    
    for file in files:
        if file.endswith('.pkl'):
            file_path = os.path.join(cache_dir, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            # Extract model name from filename
            model = file.split('_')[0]
            model_counts[model] = model_counts.get(model, 0) + 1
    
    print(f"Total cache size: {total_size / (1024*1024):.2f} MB")
    print("Files per model:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} files")
    
    return len(files), total_size, model_counts

def create_batch_cache():
    """Convert individual cache files to batch cache format"""
    cache_dir = "embeddings_cache"
    batch_cache_file = os.path.join(cache_dir, "batch_embeddings_cache.json")
    
    if not os.path.exists(cache_dir):
        return
    
    batch_cache = {}
    converted_count = 0
    
    print("Converting individual cache files to batch format...")
    
    for file in os.listdir(cache_dir):
        if file.endswith('.pkl') and file != "batch_embeddings_cache.pkl":
            file_path = os.path.join(cache_dir, file)
            
            try:
                with open(file_path, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Extract hash from filename (format: model_hash.pkl)
                filename_parts = file.replace('.pkl', '').split('_')
                if len(filename_parts) >= 2:
                    model = filename_parts[0]
                    hash_key = '_'.join(filename_parts[1:])
                    
                    if model not in batch_cache:
                        batch_cache[model] = {}
                    
                    batch_cache[model][hash_key] = embedding
                    converted_count += 1
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Save batch cache
    try:
        with open(batch_cache_file.replace('.json', '.pkl'), 'wb') as f:
            pickle.dump(batch_cache, f)
        print(f"Converted {converted_count} individual cache files to batch format")
        print(f"Batch cache saved to: {batch_cache_file.replace('.json', '.pkl')}")
        return True
    except Exception as e:
        print(f"Error saving batch cache: {e}")
        return False

def cleanup_individual_cache_files():
    """Remove individual cache files after successful batch conversion"""
    cache_dir = "embeddings_cache"
    batch_cache_file = os.path.join(cache_dir, "batch_embeddings_cache.pkl")
    
    if not os.path.exists(batch_cache_file):
        print("Batch cache file not found. Skipping cleanup.")
        return
    
    removed_count = 0
    
    for file in os.listdir(cache_dir):
        if file.endswith('.pkl') and file != "batch_embeddings_cache.pkl":
            file_path = os.path.join(cache_dir, file)
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    print(f"Removed {removed_count} individual cache files")

def main():
    print("=== BYU Pathway Topic Modeling Cache Analysis & Fix ===\n")
    
    # Step 1: Analyze current situation
    print("1. Analyzing current cache situation...")
    file_count, total_size, model_counts = analyze_cache_files()
    
    if file_count == 0:
        print("No cache files found. Nothing to fix.")
        return
    
    print(f"\nFound {file_count} cache files using {total_size / (1024*1024):.2f} MB")
    
    # Step 2: Create batch cache
    print("\n2. Creating optimized batch cache...")
    if create_batch_cache():
        print("✅ Batch cache created successfully")
        
        # Step 3: Cleanup individual files
        print("\n3. Cleaning up individual cache files...")
        response = input("Remove individual cache files? (y/N): ")
        if response.lower() in ['y', 'yes']:
            cleanup_individual_cache_files()
            print("✅ Cache optimization complete")
        else:
            print("⚠️  Individual cache files preserved")
    else:
        print("❌ Failed to create batch cache")

if __name__ == "__main__":
    main()