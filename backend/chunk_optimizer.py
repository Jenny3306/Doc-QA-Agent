import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

TEST_QUESTIONS = [
    "What attention mechanism does Falcon use?",
    "Who built the Falcon language models?",
    "How many tokens was Falcon-180B trained on?",
    "What is RefinedWeb?",
    "What benchmarks did Falcon achieve?"
]

def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for i, page in enumerate(doc):
        text += f"\n--- Page {i+1} ---\n{page.get_text()}"
    doc.close()
    return text

def get_embedding(text, input_type="passage"):
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=text,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"}
    )
    return response.data[0].embedding

def build_collection(text, chunk_size, chunk_overlap, collection_name):
    """Build a ChromaDB collection with given chunk settings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    
    db = chromadb.PersistentClient(path="./chroma_db_test")
    
    try:
        db.delete_collection(collection_name)
    except:
        pass
    collection = db.get_or_create_collection(collection_name)
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk, "passage")
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"chunk_index": i}]
        )
    
    return collection, len(chunks)

def test_retrieval(collection, questions, top_k=3):
    """Score how relevant the top retrieved chunks are."""
    total_score = 0
    
    for question in questions:
        query_emb = get_embedding(question, "query")
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["distances"]
        )
        
        distances = results["distances"][0]
        avg_distance = sum(distances) / len(distances)
        # Lower distance = better retrieval
        score = max(0, 1 - (avg_distance / 2))
        total_score += score
    
    return total_score / len(questions)

def run_optimization():
    print("=" * 55)
    print("  Chunk Size Optimization")
    print("=" * 55)
    
    text = load_pdf("test.pdf")
    
    configs = [
        {"chunk_size": 300,  "overlap": 30,  "name": "small"},
        {"chunk_size": 500,  "overlap": 50,  "name": "medium"},
        {"chunk_size": 800,  "overlap": 80,  "name": "large"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting chunk_size={config['chunk_size']}...")
        
        collection, num_chunks = build_collection(
            text,
            config["chunk_size"],
            config["overlap"],
            f"test_{config['name']}"
        )
        
        score = test_retrieval(collection, TEST_QUESTIONS)
        
        results.append({
            "name": config["name"],
            "chunk_size": config["chunk_size"],
            "num_chunks": num_chunks,
            "score": score
        })
        
        print(f"  Chunks created: {num_chunks}")
        print(f"  Retrieval score: {score:.3f}")
    
    # Find best
    best = max(results, key=lambda x: x["score"])
    
    print("\n" + "=" * 55)
    print("  Results Summary")
    print("=" * 55)
    for r in results:
        marker = " ← BEST" if r["name"] == best["name"] else ""
        print(f"  {r['name']:8} | size={r['chunk_size']} | chunks={r['num_chunks']:4} | score={r['score']:.3f}{marker}")
    
    print(f"\n  Recommendation: Use chunk_size={best['chunk_size']}")
    print("=" * 55)
    
    # Cleanup test collections
    db = chromadb.PersistentClient(path="./chroma_db_test")
    for config in configs:
        try:
            db.delete_collection(f"test_{config['name']}")
        except:
            pass
    
    return best["chunk_size"]

if __name__ == "__main__":
    best_size = run_optimization()
    print(f"\nUpdate embed_and_store.py with chunk_size={best_size}")