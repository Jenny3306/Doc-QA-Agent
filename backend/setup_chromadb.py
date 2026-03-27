import chromadb
import os

def setup_database():
    """Create a local ChromaDB database in your project folder."""
    
    # This creates a 'chroma_db' folder in your project directory
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create a collection (like a table in a regular database)
    collection = client.get_or_create_collection(
        name="nvidia_docs",
        metadata={"description": "NVIDIA document knowledge base"}
    )
    
    print("ChromaDB set up successfully!")
    print(f"Collection name: {collection.name}")
    print(f"Documents stored so far: {collection.count()}")
    
    return client, collection


if __name__ == "__main__":
    client, collection = setup_database()
    print("\nDatabase is ready at ./chroma_db")
    print("You should see a new 'chroma_db' folder in your project!")