"""
Ask-AI backend.

A thin streaming proxy in front of Gemini. It exists so the API key never
reaches the browser — that is the whole job. Everything else is kept minimal on
purpose: no database, no ORM, no auth, no framework beyond FastAPI itself.

Run:  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

from admin import public_router, router as admin_router  # noqa: E402  (needs env loaded first)

BASE = Path(__file__).parent
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))
MAX_CHARS = 600
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]

# Comma-separated. Each is tried in turn, so a free-tier quota limit on the
# first key does not take the chat down.
KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]

# Generated from the frontend's lib/content.ts by `npm run export-bio`.
# There is one content layer; never edit this file by hand.
BIO_PATH = BASE / "biography.txt"
SYSTEM_PROMPT = BIO_PATH.read_text(encoding="utf-8") if BIO_PATH.exists() else ""

app = FastAPI(title="Ishant portfolio — Ask AI", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["content-type", "authorization"],
)

app.include_router(admin_router)
app.include_router(public_router)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = []


# In-memory sliding window, keyed by IP. Good enough for portfolio traffic and
# it survives as long as the process does. Swap for Redis if this ever gets
# real traffic or runs on more than one worker.
_hits: dict[str, deque[float]] = defaultdict(deque)


def rate_limited(ip: str) -> bool:
    now = time.time()
    window = _hits[ip]
    while window and window[0] < now - 3600:
        window.popleft()
    window.append(now)
    if len(_hits) > 5000:
        _hits.clear()
    return len(window) > MAX_PER_HOUR


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "model": MODEL,
        "keys_configured": len(KEYS),
        "biography_chars": len(SYSTEM_PROMPT),
    }


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    if not KEYS:
        return PlainTextResponse(
            "The chat isn't configured yet — GEMINI_API_KEY is missing on the server. "
            "Everything else on this page works."
        )

    if not SYSTEM_PROMPT:
        return PlainTextResponse(
            "The chat has no biography loaded. Run `npm run export-bio` in the frontend "
            "to generate backend/biography.txt, then restart this server."
        )

    if rate_limited(client_ip(request)):
        return PlainTextResponse(
            f"That's {MAX_PER_HOUR} questions in an hour, which is a generous read of a "
            "portfolio. Try again shortly, or email Ishant directly — the address is at "
            "the bottom of the page."
        )

    contents = [
        {
            "role": "model" if m.role == "assistant" else "user",
            "parts": [{"text": m.content[:MAX_CHARS]}],
        }
        for m in body.messages[-10:]
        if m.content and m.content.strip()
    ]

    if not contents:
        return PlainTextResponse("Ask a question and I will answer it.")

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 500},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:streamGenerateContent?alt=sse"
    )

    async def stream():
        last_status = 0
        async with httpx.AsyncClient(timeout=60.0) as client:
            for key in KEYS:
                # These are Gemini API keys, so x-goog-api-key goes first — no
                # wasted round-trip. The Bearer retry is a cheap safety net for
                # the case where a credential is actually an OAuth token; it
                # only costs anything on a request that was failing anyway.
                for headers in (
                    {"x-goog-api-key": key},
                    {"Authorization": f"Bearer {key}"},
                ):
                    try:
                        async with client.stream(
                            "POST",
                            url,
                            json=payload,
                            headers={"content-type": "application/json", **headers},
                        ) as resp:
                            if resp.status_code != 200:
                                last_status = resp.status_code
                                await resp.aread()
                                if last_status == 429:
                                    break  # quota — move to the next key
                                continue
                            async for line in resp.aiter_lines():
                                if not line.startswith("data:"):
                                    continue
                                raw = line[5:].strip()
                                if not raw or raw == "[DONE]":
                                    continue
                                try:
                                    data = json.loads(raw)
                                except json.JSONDecodeError:
                                    continue
                                for cand in data.get("candidates", []):
                                    for part in cand.get("content", {}).get("parts", []):
                                        text = part.get("text")
                                        if text:
                                            yield text
                            return
                    except httpx.HTTPError:
                        last_status = 0
                        continue

        if last_status in (401, 403):
            yield (
                "The model rejected the credentials. That's a server-side configuration "
                "problem, not something you did — the site owner needs to check "
                "GEMINI_API_KEY. Everything else on this page works."
            )
        elif last_status == 429:
            yield (
                "The model is over its quota for now. Try again a little later, or email "
                "Ishant directly — the address is at the bottom of the page."
            )
        else:
            yield (
                "The model didn't respond. That's on the server, not on you — try again "
                "in a moment, or email Ishant directly."
            )

    return StreamingResponse(
        stream(),
        media_type="text/plain; charset=utf-8",
        headers={"cache-control": "no-store", "x-accel-buffering": "no"},
    )
