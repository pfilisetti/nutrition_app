# Nutrition AI Assistant

A RAG-powered nutrition chatbot built with LangChain, NVIDIA LLMs, and Qdrant. Ask questions about your personal nutrition documents or get precise data from the USDA FoodData Central database.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | `meta/llama-3.3-70b-instruct` via NVIDIA NIM |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector store | Qdrant Cloud |
| External data | USDA FoodData Central API |
| UI | Streamlit |

---

## Installation

### 1. Clone and install

```bash
git clone <repo-url>
cd 28_personal_project
uv sync
```

### 2. Configure environment variables

Create a `.env` file at the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
USDA_API_KEY=your_usda_api_key
```

- **NVIDIA API key**: [build.nvidia.com](https://build.nvidia.com)
- **Qdrant**: [cloud.qdrant.io](https://cloud.qdrant.io) — create a free cluster
- **USDA API key**: [fdc.nal.usda.gov/api-guide](https://fdc.nal.usda.gov/api-guide.html)

### 3. Ingest your personal documents

Place your `.docx` and `.xlsx` files in a `docs/` folder at the project root, then run:

```bash
uv run python src/nutrition_app/ingest.py
```

This only needs to be done once (or whenever your documents change).

### 4. Launch the app

```bash
nutrition
```

---

## Pipeline — what happens at each prompt

```
User types a message
        │
        ▼
┌───────────────────────┐
│   Input guard (LLM)   │  Classifies the message as FOOD / GREETING / OFF_TOPIC
└───────────┬───────────┘
            │
    ┌───────┴────────┐
    │                │
GREETING /       FOOD-related
OFF_TOPIC             │
    │                 ▼
    │   ┌─────────────────────────┐
    │   │   LangChain Agent       │  Llama 3.3 70B decides which tool(s) to call
    │   └────────────┬────────────┘
    │                │
    │     ┌──────────┼──────────────────┐
    │     │          │                  │
    │     ▼          ▼                  ▼
    │  search_    get_available_    get_detailed_
    │  personal_  usda_food()       nutritional_
    │  docs()     (USDA search)     content()
    │  (Qdrant    Returns list of   (USDA detail,
    │   RAG)      foods + fdcIds)   requires fdcId)
    │     │          │                  │
    │     └──────────┴──────────────────┘
    │                │
    │                ▼
    │   ┌─────────────────────────┐
    │   │   Agent synthesizes     │  Combines tool results into a final answer
    │   │   final answer          │
    │   └─────────────────────────┘
    │                │
    └────────────────┘
                     │
                     ▼
            Answer displayed in chat
```

### Step-by-step detail

1. **Input guard** — a first LLM call classifies the prompt. Off-topic and greeting messages are short-circuited here, no agent is invoked.

2. **Agent invocation** — for food-related prompts, the LangChain `AgentExecutor` is called with the full chat history for context.

3. **Tool selection** — the LLM reasons about which tool(s) to call:
   - `search_personal_docs` — semantic search over your ingested `.docx`/`.xlsx` files stored in Qdrant (top 4 chunks by cosine similarity)
   - `get_available_usda_food` — keyword search against the USDA FoodData Central API, returns food names and `fdcId`s
   - `get_detailed_nutritional_content` — fetches the full nutrient breakdown for a specific `fdcId`

4. **Answer synthesis** — the LLM receives the tool outputs and writes the final response, making clear which information comes from personal documents vs. the USDA database.

---

## Ingestion pipeline (one-time setup)

```
docs/*.docx + docs/*.xlsx
        │
        ▼
  Document loaders
  (Docx2txt / Unstructured)
        │
        ▼
  RecursiveCharacterTextSplitter
  chunk_size=1000, overlap=200
        │
        ▼
  HuggingFace embeddings
  (all-MiniLM-L6-v2, 384 dims)
        │
        ▼
  Qdrant Cloud collection
  "nutrition_docs" (cosine similarity)
```
