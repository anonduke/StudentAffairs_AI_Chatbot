import streamlit as st
from transformers import pipeline

@st.cache_resource
def get_relevance_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

def is_relevant_to_student_affairs(text, threshold=0.15):
    candidate_labels = [
        "student affairs","fees", "university support", "academic services",
        "fee payment", "admissions", "course registration", "scholarships", "exams",
        "grades", "mental health", "campus life", "student finance", "student services", "college support","dues", 
        "student dues", "financial aid", "student counseling", "student health","Hi","hello","hello","hi","how are you","what is your name","who are you",
        "what is your purpose","what can you do", "how can you help me", "what is student affairs", "what is campus support"
    ]
    classifier = get_relevance_classifier()
    try:
        result = classifier(text, candidate_labels)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        print(f"Relevance: {top_label} ({top_score:.2f})")
        return top_score >= threshold
    except Exception as e:
        print(f"[Relevance Filter] Classifier error: {e}")
        return True
