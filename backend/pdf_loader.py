import fitz  # PyMuPDF
import os

def load_pdf(file_path):
    """Load a PDF and extract all text page by page."""
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None
    
    doc = fitz.open(file_path)
    full_text = ""
    
    print(f"PDF loaded: {doc.page_count} pages found\n")
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    
    doc.close()
    return full_text


if __name__ == "__main__":
    text = load_pdf("test.pdf")
    
    if text:
        print(f"Total characters extracted: {len(text)}")
        print(f"\nFirst 500 characters preview:")
        print("-" * 40)
        print(text[:500])