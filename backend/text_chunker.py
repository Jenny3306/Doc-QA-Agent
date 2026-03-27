from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import os

def load_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        full_text += f"\n--- Page {page_num + 1} ---\n{page.get_text()}"
    doc.close()
    return full_text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split text into overlapping chunks.
    
    chunk_size    = how many characters per chunk
    chunk_overlap = how many characters overlap between chunks
                    (overlap helps avoid cutting sentences mid-thought)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    print("Loading PDF...")
    text = load_pdf("test.pdf")
    
    print("Chunking text...")
    chunks = chunk_text(text)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} characters")
    
    print("\n--- Sample Chunk 1 ---")
    print(chunks[0])
    print("\n--- Sample Chunk 2 ---")
    print(chunks[1])
    print("\n--- Sample Chunk 3 ---")
    print(chunks[2])