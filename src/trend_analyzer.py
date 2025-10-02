"""
Trend Analysis Module for BYU Pathway Questions Analytics
Provides temporal analysis and trend detection for student questions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from collections import Counter

logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """Analyzes trends and patterns in student questions over time"""
    
    def __init__(self):
        """Initialize the trend analyzer"""
        self.time_periods = {
            'hourly': '%Y-%m-%d %H:00',
            'daily': '%Y-%m-%d',
            'weekly': '%Y-W%U',
            'monthly': '%Y-%m'
        }
    
    def analyze_temporal_trends(self, df: pd.DataFrame, time_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Analyze trends over different time periods
        
        Args:
            df: DataFrame with timestamp and question data
            time_column: Name of timestamp column
            
        Returns:
            Dictionary with trend analysis results
        """
        if time_column not in df.columns or df.empty:
            return self._default_trends()
        
        try:
            # Ensure timestamp is datetime
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
                df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            
            trends = {}
            
            # Analyze different time periods
            for period, format_str in self.time_periods.items():
                trends[period] = self._analyze_period_trends(df_copy, time_column, period, format_str)
            
            # Overall statistics
            trends['overall'] = self._calculate_overall_trends(df_copy, time_column)
            
            # Peak activity analysis
            trends['peak_activity'] = self._analyze_peak_activity(df_copy, time_column)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing temporal trends: {e}")
            return self._default_trends()
    
    def calculate_topic_frequency_over_time(self, df: pd.DataFrame, time_period: str = 'week') -> pd.DataFrame:
        """
        Calculate topic frequency over time periods
        
        Args:
            df: DataFrame with columns ['topic', 'timestamp']
            time_period: 'day', 'week', or 'month'
        
        Returns:
            DataFrame with topic frequencies over time
        """
        if 'timestamp' not in df.columns:
            logger.error("DataFrame must have 'timestamp' column")
            return pd.DataFrame()
        
        # Ensure timestamp is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time period column
        if time_period == 'day':
            df['time_period'] = df['timestamp'].dt.date
        elif time_period == 'week':
            df['time_period'] = df['timestamp'].dt.to_period('W').dt.start_time
        elif time_period == 'month':
            df['time_period'] = df['timestamp'].dt.to_period('M').dt.start_time
        else:
            logger.warning(f"Unknown time period: {time_period}, using 'week'")
            df['time_period'] = df['timestamp'].dt.to_period('W').dt.start_time
        
        # Group by time period and topic
        topic_col = 'topic_name' if 'topic_name' in df.columns else 'topic'
        
        # Check if topic column exists and has valid data
        if topic_col not in df.columns:
            logger.warning(f"Topic column '{topic_col}' not found in DataFrame. Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Remove rows with missing topic values
        df_filtered = df.dropna(subset=[topic_col])
        if len(df_filtered) == 0:
            logger.warning(f"No valid topic data found in column '{topic_col}'")
            return pd.DataFrame()
        
        frequency_df = df_filtered.groupby(['time_period', topic_col]).size().reset_index(name='count')
        
        return frequency_df
    
    def identify_emerging_topics(self, df: pd.DataFrame, threshold_increase: float = 0.5, min_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Identify topics that are increasing in frequency
        
        Args:
            df: DataFrame with columns ['topic', 'timestamp']
            threshold_increase: Minimum percentage increase to be considered emerging (0.5 = 50%)
            min_questions: Minimum questions needed to be considered
        
        Returns:
            List of emerging topics with metadata
        """
        freq_df = self.calculate_topic_frequency_over_time(df, 'week')
        
        if freq_df.empty:
            return []
        
        topic_col = 'topic_name' if 'topic_name' in freq_df.columns else 'topic'
        
        # Check if topic column exists
        if topic_col not in freq_df.columns:
            logger.warning(f"Topic column '{topic_col}' not found in frequency DataFrame")
            return []
        emerging_topics = []
        
        # Analyze each topic
        for topic in freq_df[topic_col].unique():
            topic_data = freq_df[freq_df[topic_col] == topic].sort_values('time_period')
            
            if len(topic_data) < 2:
                continue
            
            # Get first half and second half counts
            mid_point = len(topic_data) // 2
            first_half_avg = topic_data.iloc[:mid_point]['count'].mean()
            second_half_avg = topic_data.iloc[mid_point:]['count'].mean()
            
            # Calculate percentage increase
            if first_half_avg > 0:
                pct_increase = (second_half_avg - first_half_avg) / first_half_avg
            else:
                pct_increase = 1.0 if second_half_avg > 0 else 0.0
            
            # Check if emerging
            total_questions = topic_data['count'].sum()
            if pct_increase >= threshold_increase and total_questions >= min_questions:
                emerging_topics.append({
                    'topic': topic,
                    'pct_increase': pct_increase,
                    'total_questions': total_questions,
                    'recent_avg': second_half_avg,
                    'previous_avg': first_half_avg,
                    'trend': 'emerging'
                })
        
        # Sort by percentage increase
        emerging_topics.sort(key=lambda x: x['pct_increase'], reverse=True)
        
        return emerging_topics
    
    def generate_trend_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive trend summary
        
        Args:
            df: DataFrame with questions and timestamps
        
        Returns:
            Dictionary with trend insights
        """
        emerging = self.identify_emerging_topics(df)
        
        return {
            'emerging_topics': emerging[:5],  # Top 5
            'total_questions': len(df),
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            }
        }
    
    def _analyze_period_trends(self, df: pd.DataFrame, time_column: str, period: str, format_str: str) -> Dict[str, Any]:
        """Analyze trends for a specific time period"""
        # Group by time period
        df['period'] = df[time_column].dt.strftime(format_str)
        period_counts = df.groupby('period').size()
        
        trend_data = {
            'counts': period_counts.to_dict(),
            'total_questions': int(period_counts.sum()),
            'average_per_period': float(period_counts.mean()),
            'peak_period': period_counts.idxmax() if not period_counts.empty else None,
            'peak_count': int(period_counts.max()) if not period_counts.empty else 0
        }
        
        # Calculate trend direction
        if len(period_counts) > 1:
            # Simple linear trend
            x = np.arange(len(period_counts))
            y = period_counts.values
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                trend_data['direction'] = 'increasing'
            elif slope < -0.1:
                trend_data['direction'] = 'decreasing'
            else:
                trend_data['direction'] = 'stable'
            
            trend_data['slope'] = float(slope)
        else:
            trend_data['direction'] = 'insufficient_data'
            trend_data['slope'] = 0.0
        
        return trend_data
    
    def _calculate_overall_trends(self, df: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """Calculate overall trend statistics"""
        min_date = df[time_column].min()
        max_date = df[time_column].max()
        date_range = (max_date - min_date).days
        
        overall = {
            'date_range': {
                'start': min_date.strftime('%Y-%m-%d'),
                'end': max_date.strftime('%Y-%m-%d'),
                'total_days': date_range
            },
            'total_questions': len(df),
            'questions_per_day': len(df) / max(date_range, 1),
            'active_days': len(df.groupby(df[time_column].dt.date)),
            'activity_rate': len(df.groupby(df[time_column].dt.date)) / max(date_range, 1) * 100
        }
        
        return overall
    
    def _analyze_peak_activity(self, df: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """Analyze peak activity patterns"""
        # Hour of day analysis
        hourly_activity = df.groupby(df[time_column].dt.hour).size()
        
        # Day of week analysis
        daily_activity = df.groupby(df[time_column].dt.day_name()).size()
        
        peak_activity = {
            'peak_hour': int(hourly_activity.idxmax()) if not hourly_activity.empty else None,
            'peak_hour_count': int(hourly_activity.max()) if not hourly_activity.empty else 0,
            'peak_day': daily_activity.idxmax() if not daily_activity.empty else None,
            'peak_day_count': int(daily_activity.max()) if not daily_activity.empty else 0,
            'hourly_distribution': hourly_activity.to_dict(),
            'daily_distribution': daily_activity.to_dict()
        }
        
        return peak_activity
    
    def _default_trends(self) -> Dict[str, Any]:
        """Return default trend analysis result"""
        return {
            'hourly': {'counts': {}, 'total_questions': 0, 'direction': 'no_data'},
            'daily': {'counts': {}, 'total_questions': 0, 'direction': 'no_data'},
            'weekly': {'counts': {}, 'total_questions': 0, 'direction': 'no_data'},
            'monthly': {'counts': {}, 'total_questions': 0, 'direction': 'no_data'},
            'overall': {'total_questions': 0, 'date_range': {'total_days': 0}},
            'peak_activity': {'peak_hour': None, 'peak_day': None}
        }

# Convenience functions
def analyze_trends(df: pd.DataFrame, time_column: str = 'timestamp') -> Dict[str, Any]:
    """Analyze trends for a DataFrame of questions"""
    analyzer = TrendAnalyzer()
    return analyzer.analyze_temporal_trends(df, time_column)

def detect_question_anomalies(df: pd.DataFrame, time_column: str = 'timestamp') -> Dict[str, Any]:
    """Detect anomalies in question patterns"""
    analyzer = TrendAnalyzer()
    return analyzer.detect_anomalies(df, time_column) if hasattr(analyzer, 'detect_anomalies') else {}