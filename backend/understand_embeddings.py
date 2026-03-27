from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# A sample chunk of text
sample_chunk = """NVIDIA was founded in 1993 and has become a leader 
in GPU computing, AI infrastructure, and large language model development."""

print("Sending chunk to NVIDIA embedding model...")

response = client.embeddings.create(
    model="nvidia/nv-embedqa-e5-v5",
    input=sample_chunk,
    encoding_format="float",
    extra_body={"input_type": "passage", "truncate": "NONE"}
)

embedding = response.data[0].embedding

print(f"\nChunk text: {sample_chunk[:80]}...")
print(f"\nEmbedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
print(f"\nWhat this means: this chunk is now represented as {len(embedding)} numbers.")
print("Similar chunks will have similar numbers — that's how AI search works!")