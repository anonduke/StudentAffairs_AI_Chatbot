import streamlit as st
import fitz  # PyMuPDF

@st.cache_data(show_spinner="Extracting PDF text...")
def extract_text_from_pdf(pdf_file):
    """
    Extracts all text from a PDF file uploaded via Streamlit.
    :param pdf_file: A file-like object (e.g., from st.file_uploader)
    :return: Extracted text (str)
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text