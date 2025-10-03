"""
Data cleaning and processing utilities for Questions CSV and Google Sheets data
"""
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import random

# Set seed for consistent language detection
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)

class QuestionCleaner:
    """Clean and process raw question data"""
    
    def __init__(self):
        self.us_states = [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
            'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
            'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
            'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
            'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
            'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
            'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
            'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
            'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
            'West Virginia', 'Wisconsin', 'Wyoming'
        ]
    
    def extract_question_from_kwargs(self, raw_text: str) -> Optional[str]:
        """
        Extract clean question from kwargs JSON structure
        
        Args:
            raw_text: Raw text that may contain kwargs JSON
            
        Returns:
            Cleaned question text or None if extraction fails
        """
        # If it's already a simple question, return as-is
        if not raw_text.strip().startswith('{'):
            return raw_text.strip()
        
        try:
            # Parse the JSON structure
            data = json.loads(raw_text)
            
            # Navigate through the structure to find user content
            kwargs = data.get('kwargs', {})
            if not kwargs:
                return None
            
            data_section = kwargs.get('data', {})
            if not data_section:
                return None
            
            messages = data_section.get('messages', [])
            if not messages:
                # Check if there's a direct question field
                direct_question = data_section.get('question')
                if direct_question:
                    return direct_question.strip()
                return None
            
            # Find the first user message
            for message in messages:
                if isinstance(message, dict) and message.get('role') == 'user':
                    content = message.get('content', '').strip()
                    if content:
                        return content
            
            return None
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse kwargs JSON: {e}")
            # Try to extract content using regex as fallback
            return self._extract_with_regex(raw_text)
    
    def _extract_with_regex(self, raw_text: str) -> Optional[str]:
        """Fallback regex extraction for malformed JSON"""
        try:
            # Look for "content": "..." patterns
            content_pattern = r'"content":\s*"([^"]+)"'
            matches = re.findall(content_pattern, raw_text)
            
            if matches:
                # Return the first non-empty match
                for match in matches:
                    if match.strip():
                        return match.strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"Regex extraction failed: {e}")
            return None
    
    def detect_language(self, text: str, default: str = 'en') -> str:
        """
        Detect language of text with fallback for short texts
        
        Args:
            text: Text to analyze
            default: Default language for short texts or detection failures
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        if not text or len(text.strip()) == 0:
            return default
        
        # For very short texts (< 5 words), default to English
        word_count = len(text.split())
        if word_count < 5:
            return default
        
        try:
            detected = detect(text)
            # Validate that it's a reasonable language code
            if len(detected) == 2 and detected.isalpha():
                return detected.lower()
            else:
                return default
                
        except LangDetectException:
            return default
        except Exception as e:
            logger.warning(f"Language detection error: {e}")
            return default
    
    def generate_timestamps(self, count: int, start_date: str = "2025-07-01", 
                          end_date: str = "2025-09-30") -> List[datetime]:
        """
        Generate evenly distributed timestamps across a date range
        
        Args:
            count: Number of timestamps to generate
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of datetime objects
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate total seconds in the range
        total_seconds = (end - start).total_seconds()
        
        # Generate evenly spaced timestamps
        timestamps = []
        for i in range(count):
            # Add small random variation to avoid exact patterns
            progress = i / max(count - 1, 1)  # Avoid division by zero
            offset_seconds = progress * total_seconds
            
            # Add random variation of ±30 minutes
            random_offset = random.randint(-1800, 1800)
            
            timestamp = start + timedelta(seconds=offset_seconds + random_offset)
            timestamps.append(timestamp)
        
        # Sort to maintain chronological order
        timestamps.sort()
        return timestamps
    
    def add_test_geographic_data(self, df: pd.DataFrame, 
                               test_start_date: str = "2025-09-20") -> pd.DataFrame:
        """
        Add test geographic data for questions after a certain date
        
        Args:
            df: DataFrame with questions
            test_start_date: Date from which to start adding test data
            
        Returns:
            DataFrame with updated geographic data
        """
        test_date = datetime.strptime(test_start_date, "%Y-%m-%d")
        
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        
        # Find questions after test date
        if 'timestamp' in df_copy.columns:
            # Ensure timestamp is datetime
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            test_mask = df_copy['timestamp'] >= test_date
            test_count = test_mask.sum()
            
            if test_count > 0:
                # Randomly assign some US states (about 500 records as requested)
                num_with_geo = min(500, test_count)
                
                # Get indices of test records
                test_indices = df_copy[test_mask].index.tolist()
                
                # Randomly select which ones get geographic data
                geo_indices = random.sample(test_indices, num_with_geo)
                
                # Assign random states and set country to US
                for idx in geo_indices:
                    df_copy.at[idx, 'country'] = 'United States'
                    df_copy.at[idx, 'state'] = random.choice(self.us_states)
        
        return df_copy
    
    def clean_questions_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process a questions DataFrame
        
        Args:
            df: Raw DataFrame with questions
            
        Returns:
            Cleaned DataFrame with standardized columns
        """
        logger.info(f"Starting to clean {len(df)} questions")
        
        # Create a copy to work with
        cleaned_df = df.copy()
        
        # Identify question column (could be 'question', 'Question', or first column)
        question_col = None
        for col in ['question', 'Question', 'questions', 'text']:
            if col in cleaned_df.columns:
                question_col = col
                break
        
        if question_col is None and len(cleaned_df.columns) > 0:
            question_col = cleaned_df.columns[0]
            logger.info(f"Using first column '{question_col}' as question column")
        
        if question_col is None:
            raise ValueError("No question column found in DataFrame")
        
        # Extract and clean questions
        cleaned_questions = []
        original_texts = []
        
        for idx, row in cleaned_df.iterrows():
            raw_text = str(row[question_col]) if pd.notna(row[question_col]) else ""
            original_texts.append(raw_text)
            
            # Extract clean question
            clean_question = self.extract_question_from_kwargs(raw_text)
            if clean_question is None or clean_question.strip() == "":
                clean_question = "Empty or malformed question"
            
            cleaned_questions.append(clean_question)
        
        # Create new standardized DataFrame
        result_df = pd.DataFrame({
            'original_text': original_texts,
            'cleaned_question': cleaned_questions,
        })
        
        # Add language detection
        logger.info("Detecting languages...")
        result_df['user_language'] = result_df['cleaned_question'].apply(
            lambda x: self.detect_language(x)
        )
        
        # Generate timestamps if not present
        if 'timestamp' in cleaned_df.columns:
            result_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
        else:
            logger.info("Generating evenly distributed timestamps...")
            timestamps = self.generate_timestamps(len(result_df))
            result_df['timestamp'] = timestamps
        
        # Handle geographic data
        if 'country' in cleaned_df.columns:
            result_df['country'] = cleaned_df['country']
        else:
            result_df['country'] = None
        
        if 'state' in cleaned_df.columns:
            result_df['state'] = cleaned_df['state']
        else:
            result_df['state'] = None
        
        # Add test geographic data
        result_df = self.add_test_geographic_data(result_df)
        
        # Extract user role if available
        if 'user_role' in cleaned_df.columns:
            result_df['user_role'] = cleaned_df['user_role']
        elif 'role' in cleaned_df.columns:
            result_df['user_role'] = cleaned_df['role']
        else:
            # Try to extract from original text
            result_df['user_role'] = result_df['original_text'].apply(
                self._extract_user_role
            )
        
        # Remove duplicates based on question text and timestamp
        logger.info("Removing duplicates...")
        initial_count = len(result_df)
        result_df = result_df.drop_duplicates(
            subset=['cleaned_question', 'timestamp'], 
            keep='first'
        )
        removed_count = initial_count - len(result_df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate questions")
        
        # Sort by timestamp
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Cleaning completed: {len(result_df)} questions processed")
        
        return result_df
    
    def _extract_user_role(self, raw_text: str) -> Optional[str]:
        """Extract user role from raw text if available"""
        try:
            if raw_text.strip().startswith('{'):
                data = json.loads(raw_text)
                
                # Check direct role field
                role = data.get('role')
                if role:
                    return role
                
                # Check in nested data
                kwargs = data.get('kwargs', {})
                data_section = kwargs.get('data', {})
                role = data_section.get('role')
                if role:
                    return role
                
                # Check if role is in the question data
                question_data = data_section.get('question')
                if isinstance(question_data, dict):
                    role = question_data.get('role')
                    if role:
                        return role
            
            return None
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate cleaned data and return quality metrics
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with validation results and metrics
        """
        metrics = {
            'total_questions': len(df),
            'empty_questions': 0,
            'language_distribution': {},
            'date_range': {},
            'geographic_coverage': {},
            'role_distribution': {},
            'quality_issues': []
        }
        
        # Check for empty questions
        empty_mask = (df['cleaned_question'].str.strip() == "") | \
                    (df['cleaned_question'] == "Empty or malformed question")
        metrics['empty_questions'] = empty_mask.sum()
        
        if metrics['empty_questions'] > 0:
            metrics['quality_issues'].append(
                f"{metrics['empty_questions']} empty or malformed questions"
            )
        
        # Language distribution
        lang_counts = df['user_language'].value_counts()
        metrics['language_distribution'] = lang_counts.to_dict()
        
        # Date range
        if 'timestamp' in df.columns:
            metrics['date_range'] = {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'span_days': (df['timestamp'].max() - df['timestamp'].min()).days
            }
        
        # Geographic coverage
        if 'country' in df.columns:
            country_counts = df['country'].value_counts()
            metrics['geographic_coverage']['countries'] = country_counts.to_dict()
        
        if 'state' in df.columns:
            state_counts = df['state'].value_counts()
            metrics['geographic_coverage']['states'] = state_counts.to_dict()
        
        # Role distribution
        if 'user_role' in df.columns:
            role_counts = df['user_role'].value_counts()
            metrics['role_distribution'] = role_counts.to_dict()
        
        return metrics


def clean_questions_file(file_path: str, output_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean a questions CSV file and optionally save the result
    
    Args:
        file_path: Path to the raw CSV file
        output_path: Optional path to save cleaned data
        
    Returns:
        Tuple of (cleaned DataFrame, validation metrics)
    """
    logger.info(f"Processing questions file: {file_path}")
    
    # Read the raw file
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise
    
    # Clean the data
    cleaner = QuestionCleaner()
    cleaned_df = cleaner.clean_questions_dataframe(df)
    
    # Validate the cleaned data
    validation_metrics = cleaner.validate_cleaned_data(cleaned_df)
    
    # Save if output path provided
    if output_path:
        try:
            cleaned_df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned data to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save cleaned data: {e}")
            raise
    
    return cleaned_df, validation_metrics


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        cleaned_data, metrics = clean_questions_file(input_file, output_file)
        
        print("Cleaning completed!")
        print(f"Total questions: {metrics['total_questions']}")
        print(f"Empty questions: {metrics['empty_questions']}")
        print(f"Language distribution: {metrics['language_distribution']}")
        print(f"Date range: {metrics['date_range']}")
        
        if metrics['quality_issues']:
            print("Quality issues:")
            for issue in metrics['quality_issues']:
                print(f"  - {issue}")
    else:
        print("Usage: python data_cleaning.py <input_file> [output_file]")