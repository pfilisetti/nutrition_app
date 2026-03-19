# Nutrition AI Assistant

A RAG-powered nutrition chatbot built with LangChain, NVIDIA LLMs, and Qdrant. Ask questions in French or English about nutrition, foods, and diet — the assistant searches a curated personal knowledge base and optionally queries the USDA FoodData Central database for precise nutritional data.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | `meta/llama-3.3-70b-instruct` via NVIDIA NIM |
| Embeddings | `intfloat/multilingual-e5-large` (HuggingFace, local, 1024 dims) |
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

### 3. Ingest the knowledge base

```bash
uv run python src/nutrition_app/ingest.py
```

This only needs to be done once (or whenever the knowledge base changes). It will delete and recreate the Qdrant collection.

### 4. Launch the app

```bash
nutrition
```

---

## Knowledge Base

The knowledge base lives in `docs_md/` and is organized into 4 categories:

```
docs_md/
├── concepts/     — vitamins, minerals, proteins, fats, carbohydrates, fiber,
│                   antioxidants, omega-3, grains, energy metabolism, cholesterol
├── foods/        — per-food profiles with key nutrients and % daily value
│                   (vegetables, fish, meat, eggs, dairy, legumes, nuts/seeds, fruits)
├── guides/       — raw vs cooked, food storage & safety, weight loss & muscle gain
└── personal/     — personal nutrient sources with quantities and % daily intake
```

Each file uses `##` sections (one per food or concept) so every RAG chunk is self-contained and meaningful.

---

## Pipeline — what happens at each prompt

```
User types a message
        │
        ▼
┌───────────────────────┐
│   Input guard (LLM)   │  Classifies as FOOD / GREETING / OFF_TOPIC
└───────────┬───────────┘
            │
    ┌───────┴────────┐
    │                │
GREETING /       FOOD-related
OFF_TOPIC             │
    │                 ▼
    │   ┌─────────────────────────┐
    │   │  RAG retrieval (Qdrant) │  Top 6 chunks by cosine similarity
    │   └────────────┬────────────┘
    │                │ context injected into message
    │                ▼
    │   ┌─────────────────────────┐
    │   │   LangChain Agent       │  Llama 3.3 70B (max 5 iterations)
    │   └────────────┬────────────┘
    │                │ (optionally calls USDA tools)
    │     ┌──────────┴──────────┐
    │     ▼                     ▼
    │  get_available_       get_detailed_
    │  usda_food()          nutritional_
    │  (USDA search)        content()
    │     │                     │
    │     └──────────┬──────────┘
    │                │
    │         ┌──────┴──────────────────────────┐
    │         │                                 │
    │    Proper answer                    Looped / stopped
    │         │                                 │
    │         │                    ┌────────────────────────┐
    │         │                    │  Fallback synthesis     │
    │         │                    │  RAG context +          │
    │         │                    │  intermediate tool data │
    │         │                    └────────────┬───────────┘
    │         │                                 │
    └─────────┴─────────────────────────────────┘
                                  │
                                  ▼
                         Answer displayed in chat
```

### Step-by-step detail

1. **Input guard** — a first LLM call classifies the prompt. Off-topic and greeting messages are short-circuited here; no agent is invoked.

2. **RAG retrieval** — for food-related prompts, the top 6 most relevant chunks are retrieved from Qdrant using cosine similarity and injected as context into the user message.

3. **Agent invocation** — the LangChain `AgentExecutor` receives the enriched message (context + question) and the chat history. Max 5 iterations.

4. **Tool selection (optional)** — if the knowledge base context is insufficient, the agent can call:
   - `get_available_usda_food` — keyword search against USDA FoodData Central, returns food names and `fdcId`s
   - `get_detailed_nutritional_content` — fetches the full nutrient breakdown for a specific `fdcId`

5. **Answer synthesis** — the LLM writes a natural, conversational response in the user's language (French or English). If the agent looped without producing a proper answer, a fallback direct LLM call synthesizes the final answer using the RAG context and any USDA data collected during the loop — no error is ever shown to the user.

---

## Ingestion pipeline

```
docs_md/**/*.md  (23 files, 4 categories)
        │
        ▼
  TextLoader (UTF-8)
        │
        ▼
  MarkdownHeaderTextSplitter
  splits on #, ##, ### headers
  → 279 self-contained chunks
        │
        ▼
  HuggingFace embeddings
  (intfloat/multilingual-e5-large, 1024 dims)
        │
        ▼
  Qdrant Cloud collection
  "nutrition_docs" (cosine similarity)
```
