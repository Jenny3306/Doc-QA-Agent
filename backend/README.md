# Document Q&A Agent

An intelligent document assistant powered by **NVIDIA Nemotron 3 Nano** and **LangGraph** that answers questions from any PDF with source citations, conversation memory, and honest fallback behavior.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-76B900)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-purple)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)

---

## Demo

> Upload any PDF → ask questions → get cited, grounded answers
```
You: What attention mechanism does Falcon use?
Agent [RETRIEVE]: Falcon uses multigroup attention — an extension of 
multiquery attention that improves inference scalability [Chunk 2].

You: Summarize what you just told me
Agent [META]: Falcon uses multigroup attention to reduce the K,V-cache 
size during inference, making large model deployment more efficient.

You: What is the stock price of Falcon?
Agent [RETRIEVE]: I could not find this in the document.
```

---

## Architecture
```
User Question
     │
     ▼
┌─────────────┐
│   Router    │ ──── META ────► Conversation History
│  (LangGraph)│ ──── CLARIFY ─► Ask for more detail  
└─────────────┘ ──── RETRIEVE ─► ChromaDB Search
                                      │
                                      ▼
                              NVIDIA NIM Embeddings
                              (nv-embedqa-e5-v5)
                                      │
                                      ▼
                              Top-5 Relevant Chunks
                                      │
                                      ▼
                              Nemotron 3 Nano (30B)
                              (nvidia/nemotron-3-nano-30b-a3b)
                                      │
                                      ▼
                           Grounded Answer + Citation
```

---

## Key Features

- **Semantic search** — finds relevant content by meaning, not keywords
- **Smart routing** — LangGraph decides whether to search, summarize, or clarify
- **Conversation memory** — remembers context across multiple questions
- **Source citations** — every answer references which chunk it came from
- **Honest fallback** — refuses to hallucinate when information isn't in the document
- **Web UI** — Streamlit interface with PDF upload and real-time chat

---

## Evaluation Results

| Metric | Result |
|--------|--------|
| Retrieval accuracy (5-question benchmark) | **5/5 (100%)** |
| Automated eval set score | **6/6 (100%)** |
| Honest fallback (out-of-scope questions) | **3/3 (100%)** |
| Answer quality score (Nemotron-as-judge) | **3.0/5.0** |
| Source verification (manual PDF check) | **100% factual** |

### Chunk Size Optimization

Tested chunk sizes of 300, 500, and 800 tokens:

| Chunk Size | Embedding Score | Retrieval Accuracy |
|------------|----------------|--------------------|
| 300 tokens | 0.536 | 4/5 (80%) |
| **500 tokens** | **0.526** | **5/5 (100%)** ← chosen |
| 800 tokens | 0.504 | 5/5 (100%) |

> Embedding similarity scores favored smaller chunks, but end-to-end retrieval accuracy benchmarks showed chunk_size=500 achieved 100% keyword recall vs 80% for chunk_size=300 — demonstrating that proxy metrics don't always reflect real-world performance.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | NVIDIA Nemotron 3 Nano 30B (via NIM API) |
| Embeddings | NVIDIA nv-embedqa-e5-v5 (via NIM API) |
| Vector DB | ChromaDB (local persistent storage) |
| Agent framework | LangGraph |
| PDF processing | PyMuPDF (fitz) |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Web UI | Streamlit |

---

## Project Structure
```
nvidia-doc-agent/
├── app.py                  # Streamlit web UI
├── agent_state.py          # LangGraph state definition
├── agent_nodes.py          # Agent decision nodes (Router, Retriever, Generator, Meta, Clarifier)
├── langgraph_agent.py      # Terminal agent interface
├── rag_chain.py            # Core RAG pipeline
├── rag_chat.py             # Interactive terminal chat
├── embed_and_store.py      # PDF ingestion + ChromaDB storage
├── evaluator.py            # Automated evaluation (6/6 100%)
├── retrieval_benchmark.py  # Retrieval strategy comparison
├── chunk_optimizer.py      # Chunk size optimization
├── quality_scorer.py       # Nemotron-as-judge quality scoring
├── pdf_loader.py           # PDF text extraction
├── text_chunker.py         # Text chunking
├── hello_nemotron.py       # Week 1: first API call
├── chat_loop.py            # Week 1: conversation memory
└── doc_agent.py            # Week 1: document persona
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Jenny3306/Doc-QA-Agent.git
cd Doc-QA-Agent
```

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install openai langchain-nvidia-ai-endpoints langchain-text-splitters
pip install pymupdf chromadb langchain-chroma langgraph streamlit python-dotenv
```

### 3. Add your NVIDIA API key

Create a `.env` file:
```
NVIDIA_API_KEY=nvapi-your-key-here
```

Get your free API key at [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b)

### 4. Ingest a PDF
```bash
python embed_and_store.py
```

### 5. Launch the web UI
```bash
streamlit run app.py
```

Or use the terminal agent:
```bash
python langgraph_agent.py
```

---

## Agent Decision Logic

The LangGraph router classifies every question into one of three paths:

| Route | Trigger | Action |
|-------|---------|--------|
| RETRIEVE | Normal questions | Search ChromaDB → generate grounded answer |
| META | "summarize", "recap", "tldr" | Use conversation history directly |
| CLARIFY | Vague input (< 10 chars) | Ask user for more detail |

---

## Design Decisions

**Why chunk_size=500?**
Empirical testing showed 500-token chunks achieve 100% retrieval accuracy vs 80% for 300-token chunks, despite smaller chunks scoring slightly higher on embedding similarity metrics. Real-world accuracy beats proxy metrics.

**Why top_k=5?**
Retrieving 5 chunks ensures coverage of facts that may be spread across the document (e.g., author names split across page boundaries), while keeping the context window manageable for the LLM.

**Why honest fallback?**
Production AI systems must know what they don't know. The agent explicitly refuses to answer questions not found in the document rather than hallucinating — verified with 3/3 out-of-scope test cases.

---

## Built With

- [NVIDIA NIM](https://build.nvidia.com) — inference microservices
- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration
- [ChromaDB](https://www.trychroma.com) — vector database
- [Streamlit](https://streamlit.io) — web interface

---

## Author

Built by Lê Như Nhã Uyên ([@Jenny3306](https://github.com/Jenny3306)) as a portfolio project, inspired by the NVIDIA Nemotron models for language understanding and generation.

*Second-year Computer Science student passionate about LLMs, RAG systems, and AI infrastructure.*
