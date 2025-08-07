import streamlit as st
import random
from src.chatbot import EnhancedStudentAffairsChatbot
#from src.faq_matcher import EnhancedStudentAffairsChatbot
from src.doc_parser import extract_text_from_pdf
from src.doc_qa import answer_from_context
from src.relevance_filter import is_relevant_to_student_affairs
from src.translation import translate_text

st.set_page_config(page_title="Student Affairs AI Chatbot", layout="wide")

# --- Custom CSS & Branding ---
st.markdown("""
    <style>
    .chat-bubble-user {
        background: #E6EDFF;
        color: #20213D;
        padding: 12px 16px;
        border-radius: 16px 16px 0 16px;
        margin-bottom: 6px;
        margin-left: 80px;
        max-width: 70%;
        align-self: flex-end;
        font-size: 1.1rem;
    }
    .chat-bubble-bot {
        background: #5C67F2;
        color: white;
        padding: 12px 16px;
        border-radius: 16px 16px 16px 0;
        margin-bottom: 6px;
        margin-right: 80px;
        max-width: 70%;
        align-self: flex-start;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(92, 103, 242, 0.10);
        min-height: 24px;
    }
    .chat-row {
        display: flex;
        align-items: flex-end;
        margin-bottom: 4px;
    }
    .chat-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        margin: 0 12px 6px 0;
        object-fit: cover;
        box-shadow: 0 1px 4px rgba(60, 60, 60, 0.08);
    }
    .user-avatar {
        margin-left: 12px;
        margin-right: 0;
    }
    .floating-header {
        background: linear-gradient(90deg,#5C67F2 60%,#7681FF 100%);
        color: white;
        border-radius: 1.3rem;
        margin: 0 0 1.5rem 0;
        padding: 1.2rem 2rem;
        box-shadow: 0 3px 12px rgba(44,50,178,0.13);
        text-align: left;
    }
    .sidebar-footer {
        font-size: 0.97rem;
        color: #6B70A8;
        margin-top: 2rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Modern Floating Header ---
st.markdown(
    '<div class="floating-header">'
    '<h2 style="margin-bottom:0;">🎓 Student Affairs AI Chatbot</h2>'
    '<div style="font-size:1.07rem;opacity:0.92;">Ask about campus, academics, or upload a PDF for instant answers!</div>'
    '</div>',
    unsafe_allow_html=True
)

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=90)
st.sidebar.header("How can I help you today?")
st.sidebar.write(
    "- Switch response language below.\n"
    "- Upload student-related PDF for document-based Q&A.\n"
    "- All answers are confidential and AI-powered."
)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("🌐 Language")
lang_code = st.sidebar.selectbox(
    "Choose response language:",
    options=[("English", "en"), ("French", "fr"), ("Hindi", "hi")],
    index=0,
    format_func=lambda x: x[0]
)[1]
st.sidebar.markdown('<div class="sidebar-footer">© 2025 Your School or Team Name<br>Built for Student Success.</div>', unsafe_allow_html=True)

# --- Greetings ---
greeting_inputs = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "who are you", "what is your name", "what can you do",
    "greetings"
]
greeting_responses = [
    "Hello! How can I help you today?",
    "Hi there! Feel free to ask me anything about student services.",
    "Hey! I'm here to assist you with any campus or academic questions.",
    "Hi! How can I assist you?",
    "Hello, I'm your Student Affairs AI Assistant!"
]

# --- PDF Upload and Extraction ---
uploaded_pdf = st.file_uploader("Upload a Student Affairs PDF (optional, for document-based Q&A)", type=["pdf"])
if uploaded_pdf:
    if 'pdf_context' not in st.session_state:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_pdf)
        if not is_relevant_to_student_affairs(pdf_text[:2000]):
            st.error("Sorry, this document doesn't seem related to Student Affairs. Please upload a relevant file.")
            st.stop()
        st.session_state['pdf_context'] = pdf_text
        st.success("PDF uploaded and processed.")

# --- Mode Toggle ---
mode = "FAQ Bot"
if uploaded_pdf and 'pdf_context' in st.session_state:
    mode = st.radio(
        "Choose where you want to search for answers:",
        ["PDF Doc-QA", "FAQ Bot"], index=0
    )

# --- Cache FAQ bot for fast loading ---
@st.cache_resource
def load_faq_bot():
    return EnhancedStudentAffairsChatbot()
if 'chatbot_instance' not in st.session_state:
    st.session_state.chatbot_instance = load_faq_bot()
chatbot = st.session_state.chatbot_instance

# --- Chat History State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- News/Notifications (Sample Data, replace with your live feed) ---
def get_latest_news():
    return [
        {"title": "IRCC: Fall intake updates", "link": "https://www.canada.ca/en/immigration-refugees-citizenship/news.html"},
        {"title": "Campus Health Week starts Monday!", "link": "#"},
        {"title": "Scholarship application deadline: Aug 20", "link": "#"}
    ]

# --- Layout: Chat (left) & News (right) ---
chat_col, news_col = st.columns([3, 1.1], gap="large")

with chat_col:
    # --- Render Chat History ---
    def render_chat():
        for speaker, message in st.session_state.chat_history:
            if speaker == "user":
                st.markdown(f"""
                    <div class="chat-row" style="justify-content:flex-end;">
                        <div class="chat-bubble-user">{message}</div>
                        <img src="https://cdn-icons-png.flaticon.com/512/1144/1144760.png" class="chat-avatar user-avatar">
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-row" style="justify-content:flex-start;">
                        <img src="https://cdn-icons-png.flaticon.com/512/387/387561.png" class="chat-avatar">
                        <div class="chat-bubble-bot">{message}</div>
                    </div>
                """, unsafe_allow_html=True)
    render_chat()

    # --- Input Box: Always at Bottom ---
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your message and press Send...",
            key="user_input",
            value="",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
        send_btn = st.form_submit_button("Send")

    # --- Typing Effect: Show "Bot is typing..." instantly, then replace with answer ---
    if send_btn and user_input.strip():
        st.session_state.chat_history.append(("user", user_input))
        # Show typing bubble immediately
        st.session_state.chat_history.append(("bot", "typing..."))
        st.rerun()

    # --- On rerun, if last bot message is "...typing...", generate answer ---
    if len(st.session_state.chat_history) >= 2 and st.session_state.chat_history[-1] == ("bot", "typing..."):
        user_input = st.session_state.chat_history[-2][1]
        prompt_lower = user_input.strip().lower()
        # Process as before
        if any(greet in prompt_lower for greet in greeting_inputs):
            bot_reply = translate_text(random.choice(greeting_responses), lang_code)
        elif not is_relevant_to_student_affairs(user_input):
            bot_reply = translate_text("Sorry, I can only answer questions about student affairs or campus support.", lang_code)
        else:
            if mode == "PDF Doc-QA" and uploaded_pdf and 'pdf_context' in st.session_state:
                answer = answer_from_context(user_input, st.session_state['pdf_context'])
                bot_reply = translate_text(answer, lang_code)
            else:
                response = chatbot.get_response(user_input, history=st.session_state.chat_history)
                print(f"Bot response: {response}")
                bot_reply = translate_text(response['response'], lang_code)
        # Replace "...typing..." with real answer and rerun to display
        st.session_state.chat_history[-1] = ("bot", bot_reply)
        st.rerun()

with news_col:
    st.markdown("""
        <div style="background:#F4F7FE;border-radius:16px;padding:1.2rem 1rem 1.1rem 1rem;margin-top:30px;box-shadow:0 2px 12px #d3dbfb;">
        <h4 style="margin-top:0;margin-bottom:0.7rem;color:#3740AA;">Notifications & News</h4>
    """, unsafe_allow_html=True)
    for item in get_latest_news():
        st.markdown(f"""
            <div style="margin-bottom:0.8rem;padding-bottom:0.3rem;">
                <b><a href="{item['link']}" target="_blank" style="color:#4A53E8;text-decoration:none;">{item['title']}</a></b>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
