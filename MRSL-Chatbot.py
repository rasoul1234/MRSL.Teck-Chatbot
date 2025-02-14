import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd  # For data manipulation and analysis using DataFrames.
import random  # For generating random numbers and making random selections.
import streamlit.components.v1 as components  # For embedding custom HTML components
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACEHUB_ACCESS_TOKEN = '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'

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm=llm)

def generate_response(question):
    """Generate a response using the Hugging Face model."""
    try:
        return model.invoke(question).content
    except Exception as e:
        return f"Error: {e}"

# Configurations
icons = {
    "assistant": "https://raw.githubusercontent.com/rasoul1234/MRSL.Teck-Chatbot/main/img/assistant-done.svg",
    "user": "https://raw.githubusercontent.com/rasoul1234/MRSL.Teck-Chatbot/main/img/user-done.svg",
}
st.set_page_config(page_title="MRSL.Teck", page_icon="✨", layout="wide")

welcome_messages = [
    "Hello! I'm MRSL, your AI assistant. How can I help you today?",
    "Hi there! I’m MRSL, an AI-powered assistant here to assist you. What can I do for you?",
    "Hey! I'm MRSL, your AI chatbot. Feel free to ask me anything!",
    "Greetings! I’m MRSL, your friendly AI assistant. How can I assist you today?",
    "Welcome! I'm MRSL, your AI helper. Let me know how I can assist you!",
]
message = random.choice(welcome_messages)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": message}]
if "show_animation" not in st.session_state:
    st.session_state.show_animation = True

# Sidebar Content
with st.sidebar:
    image_url = "https://raw.githubusercontent.com/rasoul1234/MRSL.Teck-Chatbot/main/img/logo.png"
    st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src='{image_url}' style='width: 100px; height: 100px; border-radius: 50%; margin-right: 20px; margin-bottom: 10px'>
            <h1 style="margin: 0; color: white; font-size: 22px; text-shadow: 2px 2px 5px #ff0000, 2px -2px 5px #ff4500; font-family: 'Arial', sans-serif; letter-spacing: 2px; background: linear-gradient(90deg, #ff7f50, #ff4500, #ff0000); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">MRSL.Tech</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown(
    """
    <div style="font-weight: bold; font-size: 16px; text-align: center; color: #333;">
        <span style="color: #28a745;">Developed by</span> 
        <a href="http://www.linkedin.com/in/muhammad-rasoul-sahibzadah-b97a47218/" style="color: #0077b5; text-decoration: underline;">Muhammad Rasoul</a>. 
        <span style="color: #ffcc00;">Like this?</span> 
        <a href="mailto:rasoul.sahibbzadah@auaf.edu.af" style="color: #d14836; text-decoration: underline;">Hire me!</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=icons[message["role"]]):
        st.write(message["content"])

# User Input and Response Handling
if prompt := st.chat_input("Ask me anything:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=icons["user"]):
        st.write(prompt)

    # Generate and Display Hugging Face Model Response
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar=icons["assistant"]):
        st.write(response)

# Animation Effect
if "has_snowed" not in st.session_state:
    st.snow()
    st.session_state["has_snowed"] = True

# Optional Particle Animation
with open("particles.html", "r") as f:
    particles_html = f.read()

if st.session_state.show_animation:
    components.html(particles_html, height=370, scrolling=False)
