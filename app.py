import streamlit as st
# మనం __init__.py లో రాసిన క్లీన్ ఇంపోర్ట్స్ ఇవి
from src import classify_user_persona, generate_support_response

# 1. Page Setup
st.set_page_config(page_title="Persona Support Agent", page_icon="🤖")
st.title("🤖 AI Customer Support Agent")
st.markdown("Welcome! How can I help you today?")

# 2. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display past chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Get user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message on the screen
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # --- MAIN BACKEND INTEGRATION ---
    
    # Step A: Detect User Persona
    with st.spinner("Analyzing persona..."):
        # __init__.py ద్వారా వచ్చిన ఫంక్షన్ ని వాడుతున్నాం
        persona_data = classify_user_persona(user_input)
        detected_persona = persona_data["persona"]
        # Show a small pop-up on the screen
        st.toast(f"Detected Persona: {detected_persona}", icon="🔍")

    # Step B: Generate Response from Database
    with st.spinner("Searching database and generating answer..."):
        # __init__.py ద్వారా వచ్చిన ఫంక్షన్ ని వాడుతున్నాం
        bot_reply = generate_support_response(user_input, detected_persona)
    
    # Display AI response on the screen
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})