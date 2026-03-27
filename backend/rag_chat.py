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


def trim_history(conversation_history, max_exchanges=6):
    """Keep only the last N exchanges to avoid token limit issues."""
    max_messages = max_exchanges * 2
    if len(conversation_history) > max_messages:
        conversation_history = conversation_history[-max_messages:]
    return conversation_history

META_KEYWORDS = [
    "summarize", "summary", "bullet point", "recap",
    "what did you", "everything you", "told me", "just said",
    "in short", "tldr", "wrap up", "overview of our"
]

def is_meta_question(question):
    """Detect if the user is asking about the conversation itself."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in META_KEYWORDS)

def generate_answer(question, chunks, conversation_history):
    """Generate answer with full conversation memory."""
    
    # For meta questions, skip document context entirely
    if is_meta_question(question):
        system_prompt = """You are a helpful assistant. 
The user is asking you to summarize or reflect on your previous answers.
Use ONLY the conversation history provided to answer.
Do not make up new information."""
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": question})
    
    else:
        # Normal RAG flow — use document chunks
        context = "\n\n---\n\n".join([f"Chunk {i+1}:\n{chunk}" 
                                       for i, chunk in enumerate(chunks)])
        
        system_prompt = f"""You are a precise document assistant powered by NVIDIA Nemotron.
Answer questions ONLY based on the context provided.
Always cite which chunk your answer comes from (e.g. 'According to Chunk 2...').
If the answer is not in the context, say: 'I could not find this in the document.'
Never make up information.

Relevant document context:
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b",
            messages=messages,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        if not answer or not answer.strip():
            return "I had trouble generating a response. Please try rephrasing."
        
        return answer

    except Exception as e:
        print(f"Error: {str(e)}")
        return "An error occurred. Please try again."

def print_welcome(chunk_count):
    """Print a nice welcome message."""
    print("\n" + "=" * 55)
    print("       NVIDIA Document Q&A Agent")
    print("       Powered by Nemotron + ChromaDB")
    print("=" * 55)
    print(f"  Knowledge base: {chunk_count} chunks loaded")
    print("  Commands:")
    print("    'quit'  — exit the agent")
    print("    'clear' — reset conversation memory")
    print("    'chunks'— show how many chunks are loaded")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    # Connect to ChromaDB
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="nvidia_docs")
    
    conversation_history = []
    
    print_welcome(collection.count())
    
    while True:
        user_input = input("You: ").strip()
        
        # Handle commands
        if user_input.lower() == "quit":
            print("\nGoodbye! Great work today.")
            break
            
        if user_input.lower() == "clear":
            conversation_history = []
            print("Conversation memory cleared!\n")
            continue
        
        if user_input.lower() == "chunks":
            print(f"Knowledge base has {collection.count()} chunks loaded.\n")
            continue
            
        if not user_input:
            continue
        
        # Trim history to avoid token limit issues
        conversation_history = trim_history(conversation_history)
        
        # Retrieve relevant chunks
        chunks = retrieve_chunks(user_input, collection, top_k=2)
        
        # Generate grounded answer
        answer = generate_answer(user_input, chunks, conversation_history)
        
        # Update conversation memory
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})
        
        # Show exchange count so user knows memory depth
        exchange_count = len(conversation_history) // 2
        print(f"\nAgent [{exchange_count} exchanges in memory]: {answer}\n")