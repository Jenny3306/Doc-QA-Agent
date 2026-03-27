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

EVAL_QUESTIONS = [
    {
        "question": "What attention mechanism does Falcon use?",
        "reference": "Falcon uses multigroup attention, an extension of multiquery attention, to improve inference scalability."
    },
    {
        "question": "Who built the Falcon language models?",
        "reference": "The Falcon models were built by the Falcon LLM Team at the Technology Innovation Institute in Abu Dhabi."
    },
    {
        "question": "How many tokens was Falcon-180B trained on?",
        "reference": "Falcon-180B was trained on 3,500 billion tokens (3.5 trillion tokens)."
    },
    {
        "question": "What is the license for Falcon-7B?",
        "reference": "Falcon-7B is released under the Apache 2.0 license."
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

def get_agent_answer(question, collection):
    """Get answer from the RAG pipeline."""
    query_emb = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=5
    )
    chunks = results["documents"][0]
    context = "\n\n---\n\n".join([f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks)])
    
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b",
            messages=[
                {"role": "system", "content": f"""Answer ONLY based on this context:
{context}
If not found, say: 'I could not find this in the document.'"""},
                {"role": "user", "content": question}
            ],
            max_tokens=400
        )
        
        answer = response.choices[0].message.content
        
        if not answer or not answer.strip():
            return "I could not generate a response."
        
        return answer
    
    except Exception as e:
        print(f"API error: {str(e)}")
        return "I could not generate a response."

def score_answer(question, agent_answer, reference_answer):
    """Use Nemotron to judge the quality of the answer on a 1-5 scale."""
    
    judge_prompt = f"""Score this answer 1-5. Reply with ONE digit only.

Question: {question}
Reference: {reference_answer}
Answer: {agent_answer}

Rules:
- If the answer contains the same key facts as the reference → score 5
- If the answer is mostly correct but adds unnecessary info → score 4  
- If the answer is partially correct → score 3
- If the answer is mostly wrong → score 2
- If the answer is completely wrong or refused → score 1

Does the answer contain the same core facts as the reference? 
Reply with a single digit 1-5:"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=5,
        temperature=0.0  # Make judge deterministic
    )
    
    try:
        score = int(response.choices[0].message.content.strip()[0])
        return min(max(score, 1), 5)
    except:
        return 3

def run_quality_scoring():
    print("=" * 55)
    print("  Answer Quality Scorer")
    print("  (Nemotron judging Nemotron)")
    print("=" * 55)
    
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_or_create_collection(name="nvidia_docs")
    
    total_score = 0
    results = []
    
    for test in EVAL_QUESTIONS:
        question = test["question"]
        reference = test["reference"]
        
        print(f"\nQ: {question}")
        
        # Get agent answer
        answer = get_agent_answer(question, collection)
        print(f"A: {answer[:150] if answer else 'No answer generated'}...")
        
        # Score it
        score = score_answer(question, answer, reference)
        total_score += score
        results.append({"question": question, "score": score})
        
        stars = "★" * score + "☆" * (5 - score)
        print(f"Score: {stars} ({score}/5)")
    
    avg_score = total_score / len(EVAL_QUESTIONS)
    
    print("\n" + "=" * 55)
    print("  Quality Score Summary")
    print("=" * 55)
    for r in results:
        stars = "★" * r["score"] + "☆" * (5 - r["score"])
        print(f"  {stars}  {r['question'][:45]}")
    print(f"\n  Average Quality Score: {avg_score:.1f}/5.0")
    
    if avg_score >= 4.0:
        print("  Excellent! Production-ready quality.")
    elif avg_score >= 3.0:
        print("  Good. Minor improvements possible.")
    else:
        print("  Needs improvement. Review retrieval settings.")
    print("=" * 55)

if __name__ == "__main__":
    run_quality_scoring()