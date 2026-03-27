from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-nano-30b-a3b",
    messages=[
        {"role": "user", "content": "In one sentence, what is a RAG pipeline?"}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)