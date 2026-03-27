import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def get_query_embedding(question):
    """Embed the user question for searching ChromaDB."""
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=question,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    return response.data[0].embedding

def retrieve_chunks(question, collection, top_k=3):
    """Find the most relevant chunks from ChromaDB."""
    query_embedding = get_query_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

def generate_answer(question, chunks):
    """Send question + retrieved chunks to Nemotron to generate a grounded answer."""
    
    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([f"Chunk {i+1}:\n{chunk}" 
                                   for i, chunk in enumerate(chunks)])
    
    system_prompt = """You are a precise document assistant. 
Answer questions ONLY based on the context provided below.
Always cite which chunk your answer comes from (e.g. 'According to Chunk 2...').
If the answer is not in the context, say: 'I could not find this in the document.'
Never make up information."""

    user_message = f"""Context from document:
{context}

Question: {question}

Answer based only on the context above:"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=600
    )
    
    return response.choices[0].message.content

def rag_query(question, collection):
    """Full RAG pipeline: retrieve then generate."""
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Step 1: Retrieve relevant chunks
    print("Retrieving relevant chunks...")
    chunks = retrieve_chunks(question, collection)
    
    print(f"Found {len(chunks)} relevant chunks")
    
    # Step 2: Generate grounded answer
    print("Generating answer with Nemotron...\n")
    answer = generate_answer(question, chunks)
    
    print(f"Answer:\n{answer}")
    print("=" * 50)
    
    return answer


if __name__ == "__main__":
    # Connect to ChromaDB
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="nvidia_docs")
    
    print(f"Connected to database: {collection.count()} chunks available\n")
    
    # Test questions
    questions = [
        "How was the model trained?",
        "What benchmark results did the model achieve?",
        "What is the capital of France?"  # This should trigger the fallback!
    ]
    
    for question in questions:
        rag_query(question, collection)