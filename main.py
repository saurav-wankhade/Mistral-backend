from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1"
)

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="mistral-small-latest",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )

    reply = response.choices[0].message.content
    return {"reply": reply}
