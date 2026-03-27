from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

conversation_history = []

print("Chat with Nemotron (type 'quit' to exit)\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye! Great work today.")
        break

    if not user_input:
        continue

    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    # Send full history so the model remembers context
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=conversation_history,
        max_tokens=500
    )

    assistant_reply = response.choices[0].message.content

    # Add assistant reply to history too
    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    print(f"\nNemotron: {assistant_reply}\n")