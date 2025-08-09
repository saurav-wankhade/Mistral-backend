from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "mistral-small-latest"
MODEL_CONTEXT_WINDOW = 8192  # Adjust if model supports more/less
RESPONSE_TOKEN_RESERVE = 512  # Reserve space for bot's reply

client = OpenAI(api_key=MISTRAL_API_KEY, base_url="https://api.mistral.ai/v1")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store history in memory (dict keyed by user_id)
chat_history = {}

# Token counting helper
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(messages):
    total = 0
    for m in messages:
        total += len(encoding.encode(f"{m['role']}: {m['content']}"))
    return total

def trim_history(messages):
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    while count_tokens(system_msgs + other_msgs) + RESPONSE_TOKEN_RESERVE > MODEL_CONTEXT_WINDOW:
        if other_msgs:
            other_msgs.pop(0)  # remove oldest user/assistant message
        else:
            break
    return system_msgs + other_msgs

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default_user")
    user_message = data.get("message", "")

    if not isinstance(user_message, str):
        raise HTTPException(400, "Message must be a string")

    # Initialize history for this user if needed
    if user_id not in chat_history:
        chat_history[user_id] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    # Add user message
    chat_history[user_id].append({"role": "user", "content": user_message})

    # Trim history to fit context window
    chat_history[user_id] = trim_history(chat_history[user_id])

    # Send to Mistral
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=chat_history[user_id],
        temperature=0.7,
        max_tokens=RESPONSE_TOKEN_RESERVE,
    )

    reply = response.choices[0].message.content

    # Add assistant reply to history
    chat_history[user_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}
