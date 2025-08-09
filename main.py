# server.py
import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import redis
from cryptography.fernet import Fernet
import tiktoken  # for token estimation

load_dotenv()

# ---------- Config ----------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FERNET_KEY = os.getenv("FERNET_KEY")  # set this to a 32-byte base64 urlsafe key
# If not set, generate one for dev (but DO NOT do this in prod)
if not FERNET_KEY:
    print("WARNING: No FERNET_KEY set. Generating ephemeral key for dev. Do NOT use in prod.")
    FERNET_KEY = Fernet.generate_key().decode()

# Model / token limits
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")
# Set a context window in tokens (set based on model you use). Example: 8192
MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "8192"))
# Reserve some tokens for the response
RESPONSE_TOKEN_RESERVE = int(os.getenv("RESPONSE_TOKEN_RESERVE", "512"))

# ---------- Clients ----------
client = OpenAI(api_key=MISTRAL_API_KEY, base_url=MISTRAL_BASE_URL)

# Redis client (optional)
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    use_redis = True
except Exception as e:
    print("Redis not available, falling back to in-memory store:", str(e))
    redis_client = None
    use_redis = False
    _in_memory_store = {}

fernet = Fernet(FERNET_KEY.encode())

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Token estimator ----------
# Use cl100k_base as a practical approx. If you find model-specific encoding, swap it.
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens_for_messages(messages: List[Dict[str, str]]) -> int:
    """
    Rough token count: encode the concatenated message contents plus role tokens.
    This is an approximation but works well in practice for trimming strategy.
    """
    total = 0
    for m in messages:
        # include role as small overhead
        # join role + content to count roughly
        text = (m.get("role", "") + ": " + m.get("content", ""))
        total += len(encoding.encode(text))
    return total

# ---------- Encryption helpers ----------
def encrypt_history_obj(obj: Any) -> bytes:
    raw = json.dumps(obj, ensure_ascii=False).encode()
    token = fernet.encrypt(raw)
    return token

def decrypt_history_obj(token_bytes: bytes) -> Any:
    raw = fernet.decrypt(token_bytes)
    return json.loads(raw.decode())

# ---------- Storage helpers ----------
def _redis_key(user_id: str) -> str:
    return f"chat_history:{user_id}"

def save_history(user_id: str, messages: List[Dict[str, str]]):
    if use_redis:
        enc = encrypt_history_obj(messages)
        redis_client.set(_redis_key(user_id).encode(), enc)
    else:
        _in_memory_store[user_id] = messages

def load_history(user_id: str) -> List[Dict[str, str]]:
    if use_redis:
        val = redis_client.get(_redis_key(user_id).encode())
        if not val:
            return []
        try:
            return decrypt_history_obj(val)
        except Exception:
            # If decryption fails, return empty to avoid crashing
            return []
    else:
        return _in_memory_store.get(user_id, [])

# ---------- Trimming helper ----------
def trim_history_to_fit(messages: List[Dict[str, str]],
                        model_limit: int = MODEL_CONTEXT_WINDOW,
                        reserve: int = RESPONSE_TOKEN_RESERVE) -> List[Dict[str, str]]:
    """
    Trim oldest non-system messages until token budget fits:
      total_tokens(messages) + reserve <= model_limit
    Keep system messages at the front always.
    """
    if not messages:
        return messages

    # Separate out the system messages to always keep them
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    # If even system messages exceed budget, we still keep them (they're small).
    # Now trim from the oldest of other_msgs
    current = system_msgs + other_msgs
    while True:
        tokens = count_tokens_for_messages(current)
        if tokens + reserve <= model_limit:
            break
        if not other_msgs:
            # nothing left to drop
            break
        # drop the oldest user/assistant message
        other_msgs.pop(0)
        current = system_msgs + other_msgs

    return current

# ---------- API endpoint ----------
@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    user_id = payload.get("user_id", "default_user")
    user_message = payload.get("message", "")
    if not isinstance(user_message, str):
        raise HTTPException(400, "message must be a string")

    # Load history, append new user message
    history = load_history(user_id)
    if not history:
        history = [{"role": "system", "content": "You are a helpful assistant."}]

    history.append({"role": "user", "content": user_message, "ts": int(time.time())})

    # Trim history to fit model limit
    history = trim_history_to_fit(history, model_limit=MODEL_CONTEXT_WINDOW, reserve=RESPONSE_TOKEN_RESERVE)

    # Send to Mistral
    # NOTE: adapt fields to your client library if different.
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        temperature=0.7,
        max_tokens=RESPONSE_TOKEN_RESERVE,
    )

    assistant_reply = resp.choices[0].message.content

    # Append assistant reply and save encrypted history
    history.append({"role": "assistant", "content": assistant_reply, "ts": int(time.time())})
    save_history(user_id, history)

    return {"reply": assistant_reply}
