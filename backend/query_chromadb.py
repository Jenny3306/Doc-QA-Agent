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
    """Embed the user's question for searching."""
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=question,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    return response.data[0].embedding

def search_documents(question, collection, top_k=3):
    """Find the most relevant chunks for a question."""
    
    # Embed the question
    query_embedding = get_query_embedding(question)
    
    # Search ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return results


if __name__ == "__main__":
    # Connect to existing database
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="nvidia_docs")
    
    print(f"Connected to database with {collection.count()} chunks")
    print("Semantic search ready!\n")
    
    # Test questions
    test_questions = [
        "What is Nemotron and what are its capabilities?",
        "How was the model trained?",
        "What benchmark results did the model achieve?"
    ]
    
    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 50)
        
        results = search_documents(question, collection)
        
        for i, doc in enumerate(results["documents"][0]):
            print(f"Result {i+1}: {doc[:200]}...")
            print()
        
        print("=" * 50 + "\n")