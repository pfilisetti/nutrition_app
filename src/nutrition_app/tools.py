import os
import requests
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

COLLECTION_NAME = "nutrition_docs"


@lru_cache(maxsize=1)
def _get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})


def get_retriever():
    return _get_retriever()


def get_rag_tool():
    return create_retriever_tool(
        _get_retriever(),
        name="search_personal_docs",
        description=(
            "Search the user's personal nutrition knowledge base. "
            "Contains: concepts (vitamins, minerals, proteins, fats, carbohydrates, fiber, antioxidants, omega-3, grains), "
            "food profiles (vegetables, fish, meat, dairy, legumes, nuts/seeds, fruits), "
            "guides (raw vs cooked, food storage, weight loss and muscle gain), "
            "and personal nutrient sources with quantities and % daily intake. "
            "Use this for any question about nutrition, diet, foods, health, or the user's personal notes."
        ),
    )


@tool
def get_available_usda_food(food_query: str, dataType: str = "Foundation") -> list | str:
    """Search for foods in the USDA FoodData Central database by keyword.
    Returns a list of matching food descriptions with their fdcId."""
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "dataType": dataType,
        "query": food_query,
        "api_key": os.environ["USDA_API_KEY"],
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    foods = response.json().get("foods", [])
    if not foods:
        return f"No foods found for '{food_query}'."
    return [{"description": f["description"], "fdcId": f["fdcId"]} for f in foods]


@tool
def get_detailed_nutritional_content(fdcId: int) -> list:
    """Get the full nutrient breakdown for a specific food item.
    Requires the fdcId returned by get_available_usda_food."""
    url = f"https://api.nal.usda.gov/fdc/v1/food/{int(fdcId)}"
    params = {"api_key": os.environ["USDA_API_KEY"], "format": "abridged"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return [
        {
            "name": n.get("name"),
            "unit": n.get("unitName"),
            "amount": n.get("amount"),
        }
        for n in data.get("foodNutrients", [])
    ]
