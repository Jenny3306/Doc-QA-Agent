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

# Test questions with known correct answers
BENCHMARK = [
    {
        "question": "What attention mechanism does Falcon use?",
        "must_contain": ["multigroup", "multi-query"],
    },
    {
        "question": "Who built the Falcon language models?",
        "must_contain": ["technology innovation institute", "tii"],
    },
    {
        "question": "How many tokens was Falcon-180B trained on?",
        "must_contain": ["3,500", "3500", "trillion"],
    },
    {
        "question": "What is RefinedWeb?",
        "must_contain": ["common crawl", "web", "filtered"],
    },
    {
        "question": "What GPU infrastructure was used to train Falcon-180B?",
        "must_contain": ["a100", "4,096", "aws"],
    }
]

def get_embedding(text):
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=text,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    return response.data[0].embedding

def benchmark_retrieval(collection, top_k=5):
    """Check if retrieved chunks contain the expected keywords."""
    
    passed = 0
    total = len(BENCHMARK)
    
    print(f"\n{'Question':<45} {'Result':<10} {'Keywords found'}")
    print("-" * 80)
    
    for test in BENCHMARK:
        question = test["question"]
        must_contain = test["must_contain"]
        
        query_emb = get_embedding(question)
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        
        all_chunks_text = " ".join(results["documents"][0]).lower()
        
        found = [k for k in must_contain if k.lower() in all_chunks_text]
        
        if found:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        short_q = question[:43] + ".." if len(question) > 43 else question
        print(f"  {short_q:<45} {status:<10} {found}")
    
    score = (passed / total) * 100
    return score, passed, total

def run_benchmark():
    print("=" * 55)
    print("  Retrieval Benchmark")
    print("=" * 55)
    
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_or_create_collection(name="nvidia_docs")
    
    print(f"\nDatabase: {collection.count()} chunks")
    
    # Test with top_k=3 (old setting)
    print("\n--- Strategy 1: top_k=3 (original) ---")
    score3, passed3, total = benchmark_retrieval(collection, top_k=3)
    
    # Test with top_k=5 (current setting)
    print("\n--- Strategy 2: top_k=5 (optimized) ---")
    score5, passed5, total = benchmark_retrieval(collection, top_k=5)
    
    print("\n" + "=" * 55)
    print("  Benchmark Summary")
    print("=" * 55)
    print(f"  top_k=3: {passed3}/{total} ({score3:.0f}%)")
    print(f"  top_k=5: {passed5}/{total} ({score5:.0f}%)")
    improvement = score5 - score3
    if improvement > 0:
        print(f"  Improvement: +{improvement:.0f}% from increasing top_k")
    else:
        print(f"  top_k=3 and top_k=5 perform equally on this dataset")
    print("=" * 55)

if __name__ == "__main__":
    run_benchmark()