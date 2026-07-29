import streamlit as st
from rag_chain import answer_question

st.title("College Notes RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            for source in message["sources"]:
                st.caption(f"Source: {source}")

# Chat input
if prompt := st.chat_input("Ask a question about your college notes..."):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching notes and generating answer..."):
            result = answer_question(prompt)

            # Handle answer_question returning either a string or a tuple
            if isinstance(result, tuple) and len(result) == 2:
                answer, sources = result
                if isinstance(sources, str):
                    sources = [sources]
            else:
                answer = result
                sources = []

        st.markdown(answer)

        # Display source filenames
        if sources:
            for source in sources:
                st.caption(f"Source: {source}")

        # Save assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })