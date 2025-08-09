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
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store chat history (for demo, keyed by user_id)
chat_history = {}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    user_id = data.get("user_id", "default_user")  # Pass from frontend

    # Initialize history if new user
    if user_id not in chat_history:
        chat_history[user_id] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    # Append the new user message
    chat_history[user_id].append({"role": "user", "content": user_message})

    # Send full history to Mistral
    response = client.chat.completions.create(
        model="mistral-small-latest",
        messages=chat_history[user_id],
        temperature=0.7,
        max_tokens=512,
    )

    reply = response.choices[0].message.content

    # Append assistant reply to history
    chat_history[user_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}
