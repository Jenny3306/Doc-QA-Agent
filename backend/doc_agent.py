from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# This is what shapes the agent's personality and behavior
SYSTEM_PROMPT = """You are a helpful document assistant powered by NVIDIA Nemotron.
Your job is to answer questions based on documents provided to you.

Rules you must follow:
- Only answer based on information given in the documents
- If the answer is not in the documents, say clearly: "I couldn't find this in the provided documents."
- Always cite which part of the document your answer comes from
- Keep answers concise and factual
- Never make up information
"""

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# Simulate a document being "given" to the agent
sample_doc = """
--- Document: NVIDIA Company Overview ---
NVIDIA was founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem.
The company is headquartered in Santa Clara, California.
NVIDIA is best known for designing graphics processing units (GPUs).
In recent years, NVIDIA has expanded into AI, data center computing, and autonomous vehicles.
The NVIDIA H100 GPU is widely used for training large language models.
"""

print("Document Q&A Agent ready. Ask questions about the loaded document.")
print("(type 'quit' to exit)\n")
print(f"Loaded document preview: {sample_doc[:80]}...\n")

while True:
    user_input = input("Your question: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye! Great work today.")
        break

    if not user_input:
        continue

    # Include the document as context in each query
    message_with_context = f"""Here is the document to reference:

{sample_doc}

User question: {user_input}"""

    conversation_history.append({
        "role": "user",
        "content": message_with_context
    })

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=conversation_history,
        max_tokens=600
    )

    reply = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    print(f"\nAgent: {reply}\n")
