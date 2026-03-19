import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import get_rag_tool, get_available_usda_food, get_detailed_nutritional_content

load_dotenv()

SYSTEM_PROMPT = """You are a knowledgeable nutrition assistant with access to three tools:

1. search_personal_docs — searches the user's personal nutrition knowledge base (concepts, food profiles, guides, personal nutrient sources)
2. get_available_usda_food — searches the USDA FoodData Central database by keyword
3. get_detailed_nutritional_content — fetches full nutrient data for a food (requires an integer fdcId from tool 2)

Rules:
- For questions about the user's own documents or notes, use search_personal_docs. ALWAYS include the name of the source document in your answer (e.g., "[Based on notes.docx]").
- For queries about nutritional data for a general food item (e.g., "apple", "fruits"):
  Step 1: ALWAYS use the `get_available_usda_food` tool to search the database. You must invoke the tool normally, do NOT reply with raw JSON text representing a function call.
  Step 2: Read the tool output, present the list of resulting food options to the user, and ask them which exact item they meant. **CRITICAL: You MUST include the `fdcId` in parentheses next to each food name in your message so it stays in the conversation history.** Stop and wait for their reply. DO NOT invoke `get_detailed_nutritional_content` yet.
  Step 3: Once the user specifies their choice (by name or ID), invoke the `get_detailed_nutritional_content` tool using the associated memory of the `fdcId` (an integer).
  CRITICAL: Never nest tool calls inside other tool calls.
- You can combine both sources when relevant — make it clear which information comes from where.
- Respond in the same language as the user's question (English or French).
- Be concise and well-formatted in your final answers."""


def get_agent_executor() -> AgentExecutor:
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=os.environ["NVIDIA_API_KEY"],
        temperature=0.0,
    )

    tools = [get_rag_tool(), get_available_usda_food, get_detailed_nutritional_content]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
