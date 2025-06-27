import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment ---
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# --- LangChain Components ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please give a very detailed, point-wise response to the question, like student notes."),
    ("user", "Question: {question}")
])
llm = Ollama(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# --- Streamlit UI ---
st.set_page_config(page_title="StudyBot with Gemma", layout="centered")
st.markdown("<h1 style='text-align:center;'>&lt;/StudyBot &gt;</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px;'>Ask detailed questions and get student-style notes, powered by the Gemma model.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input ---
input_text = st.text_input("Ask your doubt:", placeholder="e.g., What is the difference between BFS and DFS?")

# --- Response ---
if input_text:
    with st.spinner("Generating answer..."):
        try:
            result = chain.invoke({"question": input_text})
            st.markdown("### 📖 Answer")
            st.success(result)
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
