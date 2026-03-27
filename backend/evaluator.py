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

# Ground truth Q&A pairs — answers you manually verified from the PDF
EVAL_SET = [
    {
        "question": "What attention mechanism does Falcon use?",
        "expected_keywords": ["multigroup", "attention"],
        "should_answer": True
    },
    {
        "question": "What is RefinedWeb?",
        "expected_keywords": ["common crawl", "filtered", "deduplicated"],
        "should_answer": True
    },
    {
        "question": "How many tokens was Falcon-180B trained on?",
        "expected_keywords": ["3,500", "3500", "billion"],
        "should_answer": True
    },
    {
        "question": "Who built the Falcon language models?",
        "expected_keywords": ["technology innovation institute"],
        "should_answer": True
    },
    {
        "question": "What is the price of the Falcon API?",
        "expected_keywords": [],
        "should_answer": False
    },
    {
        "question": "Who is the CEO of NVIDIA?",
        "expected_keywords": [],
        "should_answer": False
    }
]

def get_query_embedding(question):
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=question,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    return response.data[0].embedding

def retrieve_chunks(question, collection, top_k=2):
    query_embedding = get_query_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

def generate_answer(question, chunks):
    context = "\n\n---\n\n".join([f"Chunk {i+1}:\n{chunk}" 
                                   for i, chunk in enumerate(chunks)])
    
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=[
            {"role": "system", "content": """You are a precise document assistant.
Answer ONLY based on the context. If not found say exactly: 'I could not find this in the document.'"""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=400
    )
    return response.choices[0].message.content

def evaluate():
    """Run evaluation and score the agent."""
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="nvidia_docs")
    
    print("=" * 55)
    print("  RAG Agent Evaluation")
    print("=" * 55)
    
    passed = 0
    total = len(EVAL_SET)
    
    for i, test in enumerate(EVAL_SET):
        question = test["question"]
        expected_keywords = test["expected_keywords"]
        should_answer = test["should_answer"]
        
        print(f"\nTest {i+1}: {question}")
        print("-" * 40)
        
        chunks = retrieve_chunks(question, collection, top_k=5)
        answer = generate_answer(question, chunks)
        answer_lower = answer.lower()
        
        print(f"Answer: {answer[:200]}...")
        
        if should_answer:
            # Check if answer contains expected keywords
            keywords_found = [k for k in expected_keywords 
                            if k.lower() in answer_lower]
            
            if keywords_found and "could not find" not in answer_lower:
                print(f"PASS — keywords found: {keywords_found}")
                passed += 1
            else:
                print(f"FAIL — missing keywords: {expected_keywords}")
        else:
            # Should have triggered fallback
            if "could not find" in answer_lower:
                print("PASS — correctly refused to answer")
                passed += 1
            else:
                print("FAIL — should have said 'could not find'")
    
    print("\n" + "=" * 55)
    score = (passed / total) * 100
    print(f"  Final Score: {passed}/{total} ({score:.0f}%)")
    
    if score == 100:
        print("  Perfect score! Agent is working correctly.")
    elif score >= 75:
        print("  Good performance. Review failed tests.")
    else:
        print("  Needs improvement. Check retrieval quality.")
    print("=" * 55)

if __name__ == "__main__":
    evaluate()