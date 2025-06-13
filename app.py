import streamlit as st
from assistant.speech_to_text import listen_and_transcribe
from assistant.text_to_speech import speak
from assistant.conversation import chat_with_ai
from models.local_llm import chat_history
import time

# Page config
st.set_page_config(page_title="Open Source Voice Assistant")
st.title("🧠 Open Source VAPI-style Voice Assistant")

# Initialize session state
if "active" not in st.session_state:
    st.session_state.active = False

# Start conversation
if st.button("🎙️ Start Talking"):
    st.session_state.active = True
    st.rerun()

# Stop conversation
if st.button("⏹️ Stop Talking"):
    st.session_state.active = False
    st.success("Stopped listening.")

# Reset memory
if st.button("🔄 Reset Conversation"):
    chat_history.clear()
    st.success("Conversation reset.")

# Auto-loop conversation while active
if st.session_state.active:
    st.info("🎧 Listening...")
    user_input = listen_and_transcribe()

    if user_input:
        st.write(f"🗣️ You: {user_input}")
        response = chat_with_ai(user_input)
        st.write(f"🤖 Assistant: {response}")
        speak(response)

        # Small delay before next listen to prevent hot loop
        time.sleep(1.0)

        # Automatically continue the loop
        st.rerun()
