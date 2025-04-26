from typing import Set
import streamlit as st
from backend.core import runllm

st.markdown(
    """
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding: 2rem 1rem;
        }
        .chat-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #4A90E2;
            text-align: center;
            margin-bottom: 2rem;
        }
        .source-list {
            font-size: 0.9rem;
            color: #888;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='chat-title'>🤖 Documentation Helper Chatbot</div>",
    unsafe_allow_html=True,
)

prompt = st.chat_input("Ask me something about your documentation...")

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_url: Set[str]) -> str:
    if not source_url:
        return ""
    sources_list = list(source_url)
    sources_list.sort()
    sources_string = "<br><br><div class='source-list'><b>Sources:</b><ul>"
    for source in sources_list:
        sources_string += f"<li>{source}</li>"
    sources_string += "</ul></div>"
    return sources_string


if prompt:
    with st.spinner("🧠 Thinking..."):
        generated_response = runllm(prompt, st.session_state["chat_history"])
        sources = set(
            doc.metadata.get("source", "Unknown")
            for doc in generated_response.get("context", [])
        )
        formatted_response = f"{generated_response.get('answer', 'No answer found.')} {create_sources_string(sources)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(
            ("ai", generated_response.get("answer", ""))
        )

for user_msg, bot_msg in zip(
    st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]
):
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg, unsafe_allow_html=True)
