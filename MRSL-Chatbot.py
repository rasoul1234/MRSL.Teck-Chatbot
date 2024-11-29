import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import random
import streamlit.components.v1 as components
import os

# Configurations
icons = {
    "assistant": "https://raw.githubusercontent.com/rasoul1234/MRSL.Teck-Chatbot/main/img/assistant-done.svg",
    "user": "https://raw.githubusercontent.com/rasoul1234/MRSL.Teck-Chatbot/main/img/user-done.svg",
}
st.set_page_config(page_title="MRSL.Teck", page_icon="✨", layout="wide")

welcome_messages = [
    "Hello! I'm MRSL, an AI assistant designed to make image metadata meaningful. Ask me anything!",
    "Hi! I'm MRSL, an AI-powered assistant for extracting and explaining EXIF data. How can I help you today?",
    "Hey! I'm MRSL, your AI-powered guide to understanding the metadata in your images. What would you like to explore?",
]
message = random.choice(welcome_messages)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": message}]
if "show_animation" not in st.session_state:
    st.session_state.show_animation = True

# Ollama Integration
try:
    llm = Ollama(model="llama3.2")  # Replace with your installed model
    output_parser = StrOutputParser()

    # Define the main prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user's queries."),
            ("user", "Question: {question}"),
        ]
    )
    chain = prompt_template | llm | output_parser
except Exception as e:
    st.error(f"Failed to initialize Ollama: {e}")

def generate_response(question):
    """Generate a response using the Ollama chain."""
    try:
        return chain.invoke({"question": question})
    except Exception as e:
        return f"Error: {e}"

# Check if Ngrok URL should be used
ngrok_url = "https://2876-103-42-2-135.ngrok-free.app"  # Replace with actual Ngrok URL
local_url = "http://localhost:8080"

# Environment variable check: Set USE_NGROK to "true" when using Ngrok
url_to_use = ngrok_url if os.getenv("USE_NGROK", "false") == "true" else local_url
print(f"Using service at: {url_to_use}")

# Sidebar Content
with st.sidebar:
    image_url = "https://raw.githubusercontent.com/rasoul1234/MRSL.Teck-Chatbot/main/img/Rasoul.jpg"
    st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src='{image_url}' style='width: 100px; height: 100px; border-radius: 50%; margin-right: 20px; margin-bottom: 10px'>
            <h1 style="margin: 0; color: white; font-size: 22px; text-shadow: 2px 2px 5px #ff0000, 2px -2px 5px #ff4500; font-family: 'Arial', sans-serif; letter-spacing: 2px; background: linear-gradient(90deg, #ff7f50, #ff4500, #ff0000); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">MRSL.Tech</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=icons[message["role"]] ):
        st.write(message["content"])

# User Input and Response Handling
if prompt := st.chat_input("Ask me anything:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=icons["user"]):
        st.write(prompt)

    # Generate and Display Ollama Response
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar=icons["assistant"]):
        st.write(response)

# Animation Effect
if "has_snowed" not in st.session_state:
    st.snow()
    st.session_state["has_snowed"] = True

# Optional Particle Animation
try:
    with open("particles.html", "r") as f:
        particles_html = f.read()

    if st.session_state.show_animation:
        components.html(particles_html, height=370, scrolling=False)

except FileNotFoundError:
    st.warning("particles.html not found. Ensure the file is in the correct directory.")
