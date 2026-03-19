import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from agent import get_agent_executor
from tools import get_retriever

load_dotenv()

st.set_page_config(page_title="Nutrition AI", page_icon="🥗", layout="centered")
st.markdown(
    """
<style>
.stApp { max-width: 800px; margin: 0 auto; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🥗 Nutrition AI Assistant")
st.write(
    "Ask questions about nutrition — I'll search your personal notes and the USDA database."
)


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
        "nutrition, diet, health, ingredients, meals, OR if it is a short selection or number "
        "(e.g., 'number 3', 'the first one', '1') that might answer a previous food list question. "
        "Respond with GREETING if it is a greeting or social message (hello, thanks, bye, etc.). "
        "Respond with OFF_TOPIC otherwise."
    )
    result = (
        llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ]
        )
        .content.strip()
        .upper()
    )

    if result == "FOOD":
        return True, None
    if result == "GREETING":
        reply = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a friendly nutrition assistant. Reply briefly and invite the user to ask a nutrition question.",
                },
                {"role": "user", "content": user_input},
            ]
        ).content
        return False, reply
    # OFF_TOPIC
    return (
        False,
        "I'm a nutrition assistant — I can only help with questions about food, diet, and nutrition.",
    )


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
                    st.session_state.messages.append(
                        {"role": "assistant", "content": guard_reply}
                    )
                else:
                    # Retrieve relevant docs and inject as context
                    rag_docs = get_retriever().invoke(prompt)
                    if rag_docs:
                        context = "\n\n---\n\n".join(d.page_content for d in rag_docs)
                        enriched_input = f"Relevant context from knowledge base:\n{context}\n\nUser question: {prompt}"
                    else:
                        enriched_input = prompt

                    # Rebuild chat history from session state (only keep last 6 messages to stay fast)
                    chat_history = []
                    recent_messages = st.session_state.messages[-7:-1] if len(st.session_state.messages) > 6 else st.session_state.messages[:-1]
                    for msg in recent_messages:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))

                    response = load_agent().invoke(
                        {
                            "input": enriched_input,
                            "chat_history": chat_history,
                        }
                    )
                    answer = response["output"]

                    # If agent looped and stopped without a proper answer,
                    # synthesize from all gathered info (RAG context + any USDA tool results)
                    if not answer or "stopped" in answer.lower() or len(answer) < 30:
                        tool_context = ""
                        for action, observation in response.get("intermediate_steps", []):
                            tool_context += f"\nTool: {action.tool}\nResult: {observation}\n"

                        synthesis_input = enriched_input
                        if tool_context:
                            synthesis_input += f"\n\nAdditional data retrieved:\n{tool_context}"

                        answer = load_llm().invoke(
                            [
                                {"role": "system", "content": (
                                    "You are a knowledgeable, conversational nutrition assistant. "
                                    "Answer the user's question using the provided context and data. "
                                    "Respond in the same language as the user's question. "
                                    "Be concise, practical, and conversational — no raw nutrient dumps."
                                )},
                                {"role": "user", "content": synthesis_input},
                            ]
                        ).content
                    placeholder.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

            except Exception as e:
                error_msg = f"**Error:** {str(e)}"
                placeholder.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
