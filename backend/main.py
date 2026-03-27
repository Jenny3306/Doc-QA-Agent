from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
import fitz
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

import sys
sys.path.append(os.path.dirname(__file__))
from agent_state import AgentState
from agent_nodes import router_node, retriever_node, generator_node, meta_node, clarifier_node

load_dotenv()

app = FastAPI(title="NVIDIA Document Agent API")

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# ── Build LangGraph agent ─────────────────────────────────────
def route_decision(state: AgentState) -> str:
    d = state["decision"]
    return "meta" if d == "meta" else "clarifier" if d == "clarify" else "retriever"

def build_agent():
    g = StateGraph(AgentState)
    g.add_node("router",    router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("generator", generator_node)
    g.add_node("meta",      meta_node)
    g.add_node("clarifier", clarifier_node)
    g.set_entry_point("router")
    g.add_conditional_edges("router", route_decision,
        {"retriever":"retriever","meta":"meta","clarifier":"clarifier"})
    g.add_edge("retriever", "generator")
    g.add_edge("generator", END)
    g.add_edge("meta",      END)
    g.add_edge("clarifier", END)
    return g.compile()

agent = build_agent()

# ── Request/Response models ───────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    chat_history: List[dict] = []

class ChatResponse(BaseModel):
    answer: str
    decision: str
    chunks: List[str]
    confidence: float
    trace: List[dict]

class UploadResponse(BaseModel):
    chunk_count: int
    filename: str
    message: str

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "NVIDIA Document Agent API running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Extract text
    doc = fitz.open(tmp_path)
    full_text = ""
    for i, page in enumerate(doc):
        full_text += f"\n--- Page {i+1} ---\n{page.get_text()}"
    doc.close()
    os.unlink(tmp_path)

    # Chunk
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_text(full_text)

    # Store in ChromaDB
    db = chromadb.PersistentClient(path="./chroma_db")
    try:    db.delete_collection("nvidia_docs")
    except: pass
    col = db.get_or_create_collection("nvidia_docs")

    for i, chunk in enumerate(chunks):
        emb = client.embeddings.create(
            model="nvidia/nv-embedqa-e5-v5",
            input=chunk, encoding_format="float",
            extra_body={"input_type":"passage","truncate":"NONE"}
        ).data[0].embedding
        col.add(
            ids=[f"chunk_{i}"],
            embeddings=[emb],
            documents=[chunk],
            metadatas=[{"chunk_index": i, "source": file.filename}]
        )

    return UploadResponse(
        chunk_count=len(chunks),
        filename=file.filename,
        message=f"Successfully indexed {len(chunks)} chunks"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a question and get a grounded answer."""

    state = {
        "question":             request.question,
        "retrieved_chunks":     [],
        "answer":               "",
        "decision":             "",
        "iterations":           0,
        "chat_history":         request.chat_history[-12:],
        "retrieval_confidence": 0.0
    }

    result   = agent.invoke(state)
    answer   = result["answer"]
    decision = result["decision"]
    chunks   = result["retrieved_chunks"]
    conf     = result["retrieval_confidence"]

    # Build trace
    GRN = "#76B900"
    trace = [{"icon":"▶","label":"Router","value":decision.upper(),"color":GRN}]
    if decision == "retrieve":
        trace += [
            {"icon":"◎","label":"Vector search","value":"Top-5 chunks retrieved","color":GRN},
            {"icon":"◎","label":"Confidence","value":f"{conf:.2f}","color":GRN},
            {"icon":"◉","label":"Generator","value":"Answer produced","color":GRN},
        ]
    elif decision == "meta":
        trace += [
            {"icon":"◎","label":"Memory","value":"Conversation history used","color":"#a855f7"},
            {"icon":"◉","label":"Summary","value":"Generated from context","color":"#a855f7"},
        ]
    else:
        trace += [
            {"icon":"◉","label":"Clarifier","value":"Asking for more detail","color":"#f59e0b"},
        ]

    return ChatResponse(
        answer=answer,
        decision=decision,
        chunks=chunks,
        confidence=conf,
        trace=trace
    )

@app.get("/status")
def status():
    """Check if database has documents loaded."""
    try:
        db = chromadb.PersistentClient(path="./chroma_db")
        col = db.get_or_create_collection("nvidia_docs")
        return {"loaded": col.count() > 0, "chunk_count": col.count()}
    except:
        return {"loaded": False, "chunk_count": 0}