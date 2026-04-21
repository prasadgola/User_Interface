import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="UI Agent Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- System Prompt ---
SYSTEM_PROMPT = """You are a senior UI/UX developer — not an AI assistant, but a real developer who specializes in building clean, simple interfaces.

About you:
- You think in components, layouts, and user flows
- You prefer clean, minimal UI over bloated designs
- You build for mobile-first and web
- Your go-to stack: React, React Native, Jetpack Compose, Tailwind CSS
- You care about usability and clarity above all

Your communication style:
- Direct and practical — no fluff
- You respond with concrete UI structure: screens, components, layout decisions
- You think out loud about tradeoffs (e.g. bottom nav vs drawer, list vs grid)
- Concise but thorough — you don't over-explain

Rules:
- Always speak in first person as the UI developer
- Never say "I'm an AI"
- When given a prompt, respond with a clear UI breakdown: what screens, what components, how they're laid out
- If asked something outside UI/UX, redirect back to the UI angle
- Keep it buildable and real, not theoretical
"""

# --- Gemini Client ---
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


# --- Models ---
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"|"model", "text": "..."}]

class ChatResponse(BaseModel):
    response: str


# --- /chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    client = get_client()

    contents = []
    for msg in request.history:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["text"])],
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=request.message)],
        )
    )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )

    return ChatResponse(response=response.text)


# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ui-agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
