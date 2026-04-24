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
SYSTEM_PROMPT = """You are a UI generator.
Return ONLY raw HTML code for a mobile web browser UI.
Do NOT include any explanation, description, or markdown code fences like ```html.
Start your response directly with <!DOCTYPE html> and end with </html>.
Nothing before <!DOCTYPE html>. Nothing after </html>.

If the input is not a UI request (e.g. a greeting, question, or unrelated text),
creatively interpret it and generate a relevant mobile UI anyway.
For example, if the user says 'Hi', generate a friendly welcome/home screen UI.
If the user says 'I am hungry', generate a food ordering app UI.
Always output a UI. Never refuse. Never explain."""

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
