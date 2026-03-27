import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from agent_state import AgentState
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# NEW — fetches fresh connection every query
db_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection():
    """Always fetch a fresh collection reference."""
    return db_client.get_or_create_collection(name="nvidia_docs")

# ── Node 1: Router ──────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    """
    Decides what the agent should do based on the question.
    This is the brain of the agent.
    """
    question = state["question"]
    
    META_KEYWORDS = ["summarize", "summary", "bullet", "recap",
                     "what did you", "told me", "tldr", "wrap up"]
    
    CLARIFY_KEYWORDS = ["what do you mean", "can you explain",
                        "i don't understand", "clarify"]
    
    question_lower = question.lower()
    
    if any(k in question_lower for k in META_KEYWORDS):
        decision = "meta"
    elif any(k in question_lower for k in CLARIFY_KEYWORDS):
        decision = "clarify"
    elif len(question.strip()) < 10:
        decision = "clarify"  # Too vague
    else:
        decision = "retrieve"
    
    print(f"[Router] Decision: {decision}")
    
    return {**state, "decision": decision}


# ── Node 2: Retriever ────────────────────────────────────────
def retriever_node(state: AgentState) -> AgentState:
    """
    Searches ChromaDB for relevant chunks.
    Also scores how confident we are in the retrieval.
    """
    question = state["question"]
    
    # Embed the question
    embedding_response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=question,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    query_embedding = embedding_response.data[0].embedding
    
    # Get fresh collection reference every time
    collection = get_collection()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "distances"]
    )
    
    chunks = results["documents"][0]
    distances = results["distances"][0]
    
    avg_distance = sum(distances) / len(distances)
    confidence = max(0, 1 - (avg_distance / 2))
    
    print(f"[Retriever] Found {len(chunks)} chunks | confidence: {confidence:.2f}")
    
    return {
        **state,
        "retrieved_chunks": chunks,
        "retrieval_confidence": confidence
    }


# ── Node 3: Generator ────────────────────────────────────────
def generator_node(state: AgentState) -> AgentState:
    """
    Generates a grounded answer using retrieved chunks.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]
    chat_history = state["chat_history"]
    confidence = state["retrieval_confidence"]
    
    # Low confidence — warn the user
    if confidence < 0.05:
        answer = ("I could not find reliable information about this in the "
                  "document. Please try rephrasing or ask about a different topic.")
        return {**state, "answer": answer}
    
    context = "\n\n---\n\n".join([f"Chunk {i+1}:\n{chunk}" 
                                   for i, chunk in enumerate(chunks)])
    
    system_prompt = f"""You are a precise document assistant powered by NVIDIA Nemotron.
    Your answers must be:
    - Direct and concise — state the answer immediately
    - Factually grounded — only use information from the context below
    - Well cited — always mention which Chunk your answer comes from
    - Complete — include all relevant details from the context

    If the answer is not in the context below, say exactly:
    'I could not find this in the document.'

    Document context:
    {context}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-6:])  # Last 3 exchanges
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b",
            messages=messages,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        if not answer or not answer.strip():
            return {**state, "answer": "I had trouble generating a response. Please try rephrasing."}
        
        print(f"[Generator] Answer generated ({len(answer)} chars)")
        return {**state, "answer": answer}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {**state, "answer": "An error occurred. Please try again."}


# ── Node 4: Meta Handler ─────────────────────────────────────
def meta_node(state: AgentState) -> AgentState:
    """
    Handles summarization and reflection questions
    using conversation history instead of document search.
    """
    question = state["question"]
    chat_history = state["chat_history"]
    
    if not chat_history:
        answer = "We haven't discussed anything yet. Please ask me a question about the document first!"
        return {**state, "answer": answer}
    
    messages = [
        {"role": "system", "content": """You are a helpful assistant.
The user wants you to summarize or reflect on the conversation so far.
Use ONLY the conversation history to answer — do not add new information."""}
    ]
    messages.extend(chat_history[-8:])
    messages.append({"role": "user", "content": question})
    
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=messages,
        max_tokens=600
    )
    
    answer = response.choices[0].message.content or "I had trouble summarizing."
    
    print(f"[Meta] Summary generated")
    
    return {**state, "answer": answer}


# ── Node 5: Clarifier ────────────────────────────────────────
def clarifier_node(state: AgentState) -> AgentState:
    """
    Asks the user to clarify when the question is too vague.
    """
    question = state["question"]
    
    answer = (f"Your question '{question}' seems a bit broad. "
              f"Could you be more specific? For example:\n"
              f"- What specific aspect are you asking about?\n"
              f"- Are you referring to a particular model, technique, or result?\n"
              f"- What context are you looking for?")
    
    print(f"[Clarifier] Asking for clarification")
    
    return {**state, "answer": answer}