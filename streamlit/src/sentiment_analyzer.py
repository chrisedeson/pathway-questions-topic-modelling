"""
Sentiment & Urgency Analysis Module for BYU Pathway Questions Analytics
Provides sentiment classification, urgency detection, and emotional analysis
for student questions.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)


# -----------------------------
# Keyword Dictionaries
# -----------------------------
EDUCATION_KEYWORDS = {
    'positive': [
        'love', 'enjoy', 'excited', 'happy', 'great', 'amazing', 'wonderful',
        'helpful', 'clear', 'understand', 'learned', 'progress', 'success',
        'confident', 'motivated', 'inspired', 'grateful', 'thank'
    ],
    'negative': [
        'confused', 'difficult', 'hard', 'struggle', 'frustrated', 'stuck',
        'unclear', 'worried', 'stressed', 'overwhelmed', 'lost', 'behind',
        'fail', 'error', 'problem', 'issue', 'trouble', 'help'
    ],
    'neutral': [
        'question', 'ask', 'how', 'what', 'when', 'where', 'why', 'which',
        'please', 'need', 'want', 'looking', 'trying', 'about'
    ]
}

URGENCY_KEYWORDS = {
    'urgent': 3, 'emergency': 3, 'asap': 3, 'immediately': 3, 'critical': 3,
    'now': 2, 'today': 2, 'help': 2, 'lost': 2, 'confused': 2, 'stuck': 2,
    'problem': 2, 'issue': 2, 'error': 2, 'broken': 2, 'not working': 2,
    'deadline': 2, 'soon': 1, 'quickly': 1, 'fast': 1, 'worried': 1,
    'concerned': 1
}

POSITIVE_KEYWORDS = [
    'thank', 'thanks', 'appreciate', 'great', 'good', 'excellent',
    'perfect', 'wonderful', 'amazing', 'helpful', 'love', 'easy'
]

NEGATIVE_KEYWORDS = [
    'not', 'no', 'never', 'can\'t', 'cannot', 'unable', 'impossible',
    'difficult', 'hard', 'confused', 'lost', 'stuck', 'problem', 'issue',
    'error', 'fail', 'failed', 'wrong', 'incorrect', 'broken', 'frustrated'
]


# -----------------------------
# Sentiment Analyzer
# -----------------------------
class SentimentAnalyzer:
    """Enhanced sentiment and urgency analyzer for student questions."""

    def __init__(self):
        self.education_keywords = EDUCATION_KEYWORDS
        self.urgency_keywords = URGENCY_KEYWORDS
        self.positive_keywords = POSITIVE_KEYWORDS
        self.negative_keywords = NEGATIVE_KEYWORDS

    # ---------- Text Processing ----------
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    # ---------- Sentiment ----------
    def _analyze_keywords(self, text: str) -> float:
        """Analyze education-specific keywords for sentiment."""
        words = text.split()
        positive_count = sum(1 for w in words if w in self.education_keywords['positive'])
        negative_count = sum(1 for w in words if w in self.education_keywords['negative'])
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0
        return (positive_count - negative_count) / total_keywords

    def _default_sentiment(self) -> Dict[str, float]:
        return {
            'sentiment': 'neutral',
            'polarity': 0.0,
            'subjectivity': 0.0,
            'confidence': 0.0,
            'keyword_score': 0.0
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text."""
        if not text or not isinstance(text, str):
            return self._default_sentiment()

        try:
            cleaned_text = self._clean_text(text)

            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(cleaned_text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
            else:
                polarity = self._analyze_keywords(cleaned_text)
                subjectivity = 0.5

            keyword_score = self._analyze_keywords(cleaned_text)
            final_polarity = (polarity * 0.7) + (keyword_score * 0.3)

            if final_polarity > 0.1:
                sentiment = 'positive'
            elif final_polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            confidence = min(1.0, abs(final_polarity) + (subjectivity * 0.3))

            return {
                'sentiment': sentiment,
                'polarity': final_polarity,
                'subjectivity': subjectivity,
                'confidence': confidence,
                'keyword_score': keyword_score
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return self._default_sentiment()

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze_text(t) for t in texts]

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """Analyze sentiment for a DataFrame of questions."""
        if text_column not in df.columns:
            logger.warning(f"Column '{text_column}' not found in DataFrame")
            return df

        sentiment_results = self.analyze_batch(df[text_column].fillna('').tolist())

        df_copy = df.copy()
        df_copy['sentiment'] = [r['sentiment'] for r in sentiment_results]
        df_copy['sentiment_polarity'] = [r['polarity'] for r in sentiment_results]
        df_copy['sentiment_subjectivity'] = [r['subjectivity'] for r in sentiment_results]
        df_copy['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]

        return df_copy

    # ---------- Urgency ----------
    def analyze_urgency(self, text: str) -> Tuple[str, float, List[str]]:
        """Analyze urgency level of a question."""
        if not text:
            return 'low', 0.0, []

        text_lower = text.lower()
        matched_keywords = []
        urgency_score = 0

        for kw, weight in self.urgency_keywords.items():
            if kw in text_lower:
                matched_keywords.append(kw)
                urgency_score += weight

        # Question marks, exclamations, ALL CAPS
        urgency_score += text.count('?') - 1 if text.count('?') > 1 else 0
        urgency_score += text.count('!')
        urgency_score += len(re.findall(r'\b[A-Z]{3,}\b', text)) * 0.5

        normalized = min(urgency_score / 10, 1.0)

        if normalized >= 0.6:
            level = 'high'
        elif normalized >= 0.3:
            level = 'medium'
        else:
            level = 'low'

        return level, normalized, matched_keywords

    # ---------- Summaries ----------
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for sentiment analysis."""
        if 'sentiment' not in df.columns:
            return {}

        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)

        summary = {
            'total_questions': total,
            'sentiment_distribution': {
                'positive': sentiment_counts.get('positive', 0),
                'negative': sentiment_counts.get('negative', 0),
                'neutral': sentiment_counts.get('neutral', 0),
            },
            'sentiment_percentages': {
                k: (sentiment_counts.get(k, 0) / total * 100) if total > 0 else 0
                for k in ['positive', 'negative', 'neutral']
            }
        }

        if 'sentiment_polarity' in df.columns:
            summary['polarity_stats'] = {
                'mean': df['sentiment_polarity'].mean(),
                'std': df['sentiment_polarity'].std(),
                'min': df['sentiment_polarity'].min(),
                'max': df['sentiment_polarity'].max()
            }

        return summary


# -----------------------------
# Convenience Functions
# -----------------------------
def analyze_question_sentiment(question: str) -> Dict[str, float]:
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_text(question)


def analyze_questions_sentiment(questions: List[str]) -> List[Dict[str, float]]:
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_batch(questions)
