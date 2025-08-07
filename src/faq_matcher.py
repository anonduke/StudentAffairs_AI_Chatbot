import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_data
def load_faq_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'questions' in df.columns and 'answers' in df.columns:
        df = df.rename(columns={'questions': 'question', 'answers': 'answer'})
    return df

@st.cache_resource
def get_tfidf_vectorizer_and_matrix(questions):
    vectorizer = TfidfVectorizer(
        max_features=1000, ngram_range=(1, 2), stop_words='english'
    )
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix

class EnhancedStudentAffairsChatbot:
    def __init__(self, csv_path: str = "data/conestoga_faqs.csv"):
        self.csv_path = csv_path
        self.df = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        self.analytics = {
            'total_queries': 0,
            'escalations': 0,
            'successful_matches': 0,
            'avg_confidence': 0.0,
            'query_history': [],
            'top_queries': {},
            'response_times': []
        }

        self.escalation_keywords = [
            'stressed', 'anxious', 'depressed', 'overwhelmed', 'crisis',
            'help me', 'desperate', 'suicide', 'self harm', 'mental health',
            'counseling', 'therapy', 'emergency', 'urgent', 'breakdown'
        ]

        self.load_and_process_data()
        self.sentiment_analyzer = get_sentiment_pipeline()

    def load_and_process_data(self):
        try:
            self.df = load_faq_data(self.csv_path)
            logger.info(f"Loaded {len(self.df)} Q&A pairs")
            self.df['processed_question'] = self.df['question'].apply(self.preprocess_text)
            self.tfidf_vectorizer, self.tfidf_matrix = get_tfidf_vectorizer_and_matrix(
                self.df['processed_question'].fillna('').tolist()
            )
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.create_dummy_data()

    def create_dummy_data(self):
        dummy_data = [
            {'question': 'How do I pay my fees?', 'answer': 'You can pay fees through the online portal.'},
            {'question': 'How to get DMC certificate?', 'answer': 'Submit Form-A at Academic Office.'},
            {'question': 'Course registration process?', 'answer': 'Use the Course Registration portal.'}
        ]
        self.df = pd.DataFrame(dummy_data)
        self.df['processed_question'] = self.df['question'].apply(self.preprocess_text)
        self.tfidf_vectorizer, self.tfidf_matrix = get_tfidf_vectorizer_and_matrix(
            self.df['processed_question'].fillna('').tolist()
        )

    def preprocess_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)

    def check_escalation_needed(self, text: str) -> bool:
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.escalation_keywords):
            return True
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text)[0]
                if sentiment['label'].upper() == 'NEGATIVE' and sentiment['score'] > 0.8:
                    return True
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        return False

    def find_best_match(self, user_query: str) -> dict:
        processed_query = self.preprocess_text(user_query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        return {
            'answer': self.df.iloc[best_idx]['answer'],
            'confidence': float(best_score),
            'matched_question': self.df.iloc[best_idx]['question'],
            'method': 'TF-IDF Cosine Similarity'
        }

    # def get_response(self, user_query: str, history: list = None) -> dict:
    #     self.analytics['total_queries'] += 1
    #     if self.check_escalation_needed(user_query):
    #         self.analytics['escalations'] += 1
    #         return {
    #             'response': "This sounds serious. Please contact a student advisor.",
    #             'confidence': 1.0,
    #             'escalated': True,
    #             'method': 'Escalation Detection'
    #         }

    #     result = self.find_best_match(user_query)
    #     if result['confidence'] >= 0.3:
    #         self.analytics['successful_matches'] += 1
    #         return {
    #             'response': result['answer'],
    #             'confidence': result['confidence'],
    #             'escalated': False,
    #             'method': result['method']
    #         }
    #     else:
    #         self.analytics['escalations'] += 1
    #         return {
    #             'response': "I'm not sure about that. Let me connect you with a human advisor.",
    #             'confidence': result['confidence'],
    #             'escalated': True,
    #             'method': 'Low Confidence Escalation'
    #         }
