import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .tools import get_rag_tool, get_available_usda_food, get_detailed_nutritional_content

load_dotenv()

SYSTEM_PROMPT = """You are a knowledgeable nutrition assistant with access to three tools:

1. search_personal_docs — searches the user's personal uploaded documents (notes, summaries)
2. get_available_usda_food — searches the USDA FoodData Central database by keyword
3. get_detailed_nutritional_content — fetches full nutrient data for a food (requires fdcId from tool 2)

Rules:
- For questions about the user's own documents or notes, use search_personal_docs.
- For precise nutritional data, always call get_available_usda_food first, then get_detailed_nutritional_content with the relevant fdcId.
- You can combine both sources when relevant — make it clear which information comes from where.
- Be concise and well-formatted in your final answers."""


def get_agent_executor() -> AgentExecutor:
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=os.environ["NVIDIA_API_KEY"],
        temperature=0.0,
    )

    tools = [get_rag_tool(), get_available_usda_food, get_detailed_nutritional_content]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
