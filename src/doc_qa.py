# student_affairs_bot/core/doc_qa.py

# Option 1: Use HuggingFace QA pipeline (runs locally, slower, but free)
from transformers import pipeline

qa_model = None

def get_qa_pipeline():
    global qa_model
    if qa_model is None:
        qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return qa_model

def answer_from_context(question, context):
    """
    Answers a question based on the supplied context (text).
    :param question: User's question (str)
    :param context: Document text (str)
    :return: Answer (str)
    """
    pipe = get_qa_pipeline()
    try:
        result = pipe(question=question, context=context, top_k=1)
        answer = result['answer'] if isinstance(result, dict) else result[0]['answer']
    except Exception:
        answer = "Sorry, I couldn't find an answer in the document."
    return answer

import openai
def answer_from_context_openai(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering ONLY from the context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        temperature=0.2,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

