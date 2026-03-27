import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def load_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        full_text += f"\n--- Page {page_num + 1} ---\n{page.get_text()}"
    doc.close()
    return full_text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def get_embedding(text):
    """Convert a text chunk into an embedding vector using NVIDIA NIM."""
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=text,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"}
    )
    return response.data[0].embedding

def store_chunks(chunks, collection):
    """Embed each chunk and store it in ChromaDB."""
    print(f"\nEmbedding and storing {len(chunks)} chunks...")
    print("This may take a minute...\n")
    
    for i, chunk in enumerate(chunks):
        # Get embedding from NVIDIA
        embedding = get_embedding(chunk)
        
        # Store in ChromaDB with a unique ID
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"chunk_index": i, "source": "test.pdf"}]
        )
        
        # Progress update every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"  Stored {i + 1}/{len(chunks)} chunks...")
    
    print(f"\nAll {len(chunks)} chunks stored successfully!")


if __name__ == "__main__":
    # Set up ChromaDB
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="nvidia_docs")
    
    # Check if already populated
    if collection.count() > 0:
        print(f"Database already has {collection.count()} chunks.")
        print("Clearing and re-ingesting...")
        db_client.delete_collection("nvidia_docs")
        collection = db_client.get_or_create_collection(name="nvidia_docs")
    
    # Load and chunk PDF
    print("Loading PDF...")
    text = load_pdf("test.pdf")
    
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Embed and store
    store_chunks(chunks, collection)
    
    print(f"\nFinal count in database: {collection.count()} chunks")
    print("Your knowledge base is ready!")