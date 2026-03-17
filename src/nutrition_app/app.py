import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .agent import get_agent_executor

load_dotenv()

st.set_page_config(page_title="Nutrition AI", page_icon="🥗", layout="centered")
st.markdown("""
<style>
.stApp { max-width: 800px; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

st.title("🥗 Nutrition AI Assistant")
st.write("Ask questions about nutrition — I'll search your personal notes and the USDA database.")


@st.cache_resource
def load_agent():
    return get_agent_executor()


@st.cache_resource
def load_llm():
    return ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=os.environ["NVIDIA_API_KEY"],
        temperature=0.0,
    )


def check_input(user_input: str) -> tuple[bool, str | None]:
    """Returns (is_food_related, polite_reply_or_None)."""
    llm = load_llm()
    system = (
        "You are a classifier for a nutrition assistant app. "
        "Respond with exactly one word: FOOD if the user message is related to food, "
        "nutrition, diet, health, ingredients, or meals. "
        "Respond with GREETING if it is a greeting or social message (hello, thanks, bye, etc.). "
        "Respond with OFF_TOPIC otherwise."
    )
    result = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]).content.strip().upper()

    if result == "FOOD":
        return True, None
    if result == "GREETING":
        reply = llm.invoke([
            {"role": "system", "content": "You are a friendly nutrition assistant. Reply briefly and invite the user to ask a nutrition question."},
            {"role": "user", "content": user_input},
        ]).content
        return False, reply
    # OFF_TOPIC
    return False, "I'm a nutrition assistant — I can only help with questions about food, diet, and nutrition."


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about nutrition..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                is_food_related, guard_reply = check_input(prompt)

                if not is_food_related:
                    placeholder.markdown(guard_reply)
                    st.session_state.messages.append({"role": "assistant", "content": guard_reply})
                else:
                    # Rebuild chat history from session state (excluding current message)
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))

                    response = load_agent().invoke({
                        "input": prompt,
                        "chat_history": chat_history,
                    })
                    answer = response["output"]
                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"**Error:** {str(e)}"
                placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
