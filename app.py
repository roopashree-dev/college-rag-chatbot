import streamlit as st
from rag_chain import answer_question

st.set_page_config(page_title="College Notes RAG Chatbot")
st.title("College Notes RAG Chatbot")

# st.session_state persists data across reruns of the script.
# Streamlit reruns the whole file top-to-bottom on every interaction,
# so without session_state, your chat history would reset and vanish
# every time you sent a new message.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the existing conversation so it stays visible
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Source: {', '.join(msg['sources'])}")

# Get new input from the user
user_question = st.chat_input("Ask a question about your notes...")

if user_question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Generate and show the bot's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(user_question)
        st.write(answer)
        if sources:
            st.caption(f"Source: {', '.join(sources)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })