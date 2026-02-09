import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="AI Mental Health Therapist", layout="wide")
st.title("🧠 SAFESPACE – AI Mental Health Therapist")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("What's on your mind today?")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    try:
        response = requests.post(BACKEND_URL, json={"message": user_input}, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes (e.g., 404, 500)
        data = response.json()
        assistant_content = f'{data.get("response", "Sorry, I couldn\'t process that.")} WITH TOOL: [{data.get("tool_called", "None")}]'
    except requests.exceptions.RequestException as e:
        assistant_content = f"Error connecting to backend: {str(e)}. Please ensure the backend is running."
    except ValueError:
        assistant_content = "Error: Invalid response from backend. Please check the server."
    
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_content})

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])