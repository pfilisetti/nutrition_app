# Changes — RAG Knowledge Base Overhaul

This document summarizes all the improvements made to the project in this iteration.

---

## 1. Data Sources — Complete Rebuild

### Before
- Raw `.docx` and `.xlsx` files in `docs/`
- Word tables lost column-value alignment when extracted by `docx2txt` → chunks were unusable
- Excel data treated as raw unstructured text by `UnstructuredExcelLoader`
- Mix of French and English content in the same files

### After
- All content converted to structured Markdown in `docs_md/`, fully in English
- Each file is organized with `##` sections — one per food or concept — so each RAG chunk is self-contained
- 23 Markdown files across 4 categories, 279 chunks total

### New file structure

```
docs_md/
├── concepts/
│   ├── antioxidants.md
│   ├── carbohydrates-sugars.md
│   ├── daily-intake-recommendations.md
│   ├── energy-metabolism.md
│   ├── fats-and-oils.md
│   ├── fiber.md
│   ├── grains-and-flours.md
│   ├── minerals.md
│   ├── omega3-and-fats.md
│   ├── other-compounds.md
│   ├── proteins.md
│   └── vitamins.md
├── foods/                          ← NEW
│   ├── dairy.md
│   ├── fish-seafood.md
│   ├── fruits.md
│   ├── legumes.md
│   ├── meat-eggs.md
│   ├── nuts-seeds.md
│   └── vegetables.md
├── guides/
│   ├── food-storage-safety.md
│   ├── raw-vs-cooked.md
│   └── weight-loss-muscle-gain.md
└── personal/
    └── my-nutrient-sources.md
```

### Notable content additions (from source document images)
- `proteins.md`: WHO amino acid daily requirements table
- `vitamins.md`: Folate DFE conversion formula
- `fats-and-oils.md`: Fat label reading guide + fatty acid chain lengths table
- `other-compounds.md`: Phytic acid content by food (14 foods, % dry weight)
- `grains-and-flours.md`: Oats macros + minerals, spelt profile, almond/coconut flour profiles, flour comparison table
- `antioxidants.md`: Anthocyanin daily intake by country
- `weight-loss-muscle-gain.md`: Ghrelin/lean mass loss warning + bulking/cutting pros & cons tables

---

## 2. `ingest.py`

| | Before | After |
|--|--------|-------|
| Source directory | `docs/` (flat) | `docs_md/` (recursive via `Path.rglob`) |
| Loaders | `Docx2txtLoader` + `UnstructuredExcelLoader` | `TextLoader` for all `.md` files |
| Splitter | `RecursiveCharacterTextSplitter(1000, 200)` | `MarkdownHeaderTextSplitter` on `#`, `##`, `###` |
| Embedding model | `all-MiniLM-L6-v2` (384 dims, English-only) | `intfloat/multilingual-e5-large` (1024 dims, FR+EN) |
| Chunk metadata | source file only | source file + h1/h2/h3 header context |

**Why `MarkdownHeaderTextSplitter`**: each `##` section (e.g. `## Salmon`, `## Vitamin C`) becomes its own chunk with full context. No more mid-sentence splits that lose the food name or nutrient label.

**Why `multilingual-e5-large`**: the user writes queries in French; `all-MiniLM-L6-v2` is English-optimized and performs poorly on French semantic search.

---

## 3. `tools.py`

| | Before | After |
|--|--------|-------|
| Embedding model | `all-MiniLM-L6-v2` | `intfloat/multilingual-e5-large` + `normalize_embeddings=True` |
| Retrieval k | 4 | 6 |
| RAG tool description | generic | detailed list of what the knowledge base contains |
| Public API | `get_rag_tool()` only | added `get_retriever()` for direct use in `app.py` |

---

## 4. `agent.py` — Architecture Change

### Core problem
Using `search_personal_docs` as an agent tool caused an infinite loop: Llama 3.3 70B (via NVIDIA NIM) would call the tool repeatedly without ever producing a final answer. This is a known incompatibility between this model and LangChain's `create_tool_calling_agent` — the model does not reliably decide when to stop calling tools.

### Solution
RAG retrieval is now done **before** the agent, in `app.py`. The agent receives the retrieved context as part of the user message and only manages the USDA tools.

| | Before | After |
|--|--------|-------|
| Agent tools | `search_personal_docs` + 2 USDA tools | 2 USDA tools only |
| RAG invocation | Inside the agent (tool call) | Outside the agent (pre-retrieval in `app.py`) |
| `max_iterations` | Default (15) | 5 |
| `return_intermediate_steps` | False | True (used for fallback synthesis) |

### System prompt changes
- Language rule moved to the top and made explicit: *"if they write in French, you MUST answer in French"*
- Added answer format rules: no raw nutrient dumps, synthesize into natural conversational answers
- USDA tools described as fallback only (when context is insufficient)

---

## 5. `app.py`

### Pre-agent RAG retrieval

```python
rag_docs = get_retriever().invoke(prompt)
if rag_docs:
    context = "\n\n---\n\n".join(d.page_content for d in rag_docs)
    enriched_input = f"Relevant context from knowledge base:\n{context}\n\nUser question: {prompt}"
else:
    enriched_input = prompt
```

The agent always receives relevant knowledge base context, even before deciding whether to call USDA tools.

### Fallback synthesis on agent loop

If the agent hits `max_iterations` without producing a proper answer, the app collects all intermediate tool results (USDA data gathered during the loop) and makes a direct LLM call to synthesize a final answer. No error is ever shown to the user.

```python
if not answer or "stopped" in answer.lower() or len(answer) < 30:
    tool_context = ""
    for action, observation in response.get("intermediate_steps", []):
        tool_context += f"\nTool: {action.tool}\nResult: {observation}\n"

    synthesis_input = enriched_input
    if tool_context:
        synthesis_input += f"\n\nAdditional data retrieved:\n{tool_context}"

    answer = load_llm().invoke([...synthesis prompt...]).content
```

This means even a looping agent is useful — any USDA data it managed to collect before stopping feeds into the final answer.
