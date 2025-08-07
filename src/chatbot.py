import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from datetime import datetime
import logging
import streamlit as st
from transformers import pipeline
from src.llm_fallback import ask_llm

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
context_buffer = []
class EnhancedStudentAffairsChatbot:
    def __init__(self, csv_path: str = "data/conestoga_faqs.csv"):
        self.csv_path = csv_path
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
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

        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        except Exception as e:
            logger.warning(f"Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None

    def load_and_process_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} Q&A pairs")

            if 'questions' in self.df.columns and 'answers' in self.df.columns:
                self.df = self.df.rename(columns={'questions': 'question', 'answers': 'answer'})

            self.df['processed_question'] = self.df['question'].apply(self.preprocess_text)
            self.create_tfidf_index()
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
        self.create_tfidf_index()

    def preprocess_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)

    def create_tfidf_index(self):
        questions = self.df['processed_question'].fillna('').tolist()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(questions)

    def check_escalation_needed(self, text: str) -> bool:
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.escalation_keywords):
            return True
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text)[0]
                if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
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
        context_buffer.append((user_query, self.df.iloc[best_idx]['answer']))
        return {
            'answer': self.df.iloc[best_idx]['answer'],
            'confidence': float(best_score),
            'matched_question': self.df.iloc[best_idx]['question'],
            'method': 'TF-IDF Cosine Similarity'
        }

    def get_response(self, user_query: str, history: list = None) -> dict:
        self.analytics['total_queries'] += 1
        if self.check_escalation_needed(user_query):
            self.analytics['escalations'] += 1
            return {
                'response': "This sounds serious. Please contact a student advisor.",
                'confidence': 1.0,
                'escalated': True,
                'method': 'Escalation Detection'
            }
        print(f"Processing query: {user_query}")
        result = self.find_best_match(user_query)
        confidence_threshold = 0.9
        print (f"Best match: {result['matched_question']} (Confidence: {result['confidence']:.2f})")
        if result['confidence'] >= confidence_threshold:
            self.analytics['successful_matches'] += 1
            return {
                'response': result['answer'],
                'confidence111 ': result['confidence'],
                'escalated': False,
                'method': result['method']
            }
        else:
            # --- LLM Fallback ---
            llm_reply = ask_llm(user_query, context_buffer=context_buffer)
            context_buffer.append((llm_reply))
            return {
                'response': llm_reply,
                'confidence': result['confidence'],
                'escalated': False,
                'method': "LLM Fallback"
            }

def run_chat_ui():
    st.set_page_config(page_title="Student Affairs Chatbot", layout="centered")
    chatbot = EnhancedStudentAffairsChatbot()

    st.title("🎓 Student Affairs Chatbot")
    st.write("Ask me a question related to student services.")

    user_input = st.text_input("Your question:")
    if user_input:
        with st.spinner("Thinking..."):
            response = chatbot.get_response(user_input)
            if response['escalated']:
                st.warning(response['response'])
            else:
                st.success(response['response'])
            st.caption(f"Confidence: {response['confidence']:.2f} | Method: {response['method']}")

if __name__ == "__main__":
    run_chat_ui()