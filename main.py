import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str


# --- /chat Endpoint (text, fixed model) ---
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
            max_output_tokens=8192,
        ),
    )

    # Strip any preamble defensively
    text = response.text.strip()
    start = text.lower().find("<!doctype")
    if start == -1:
        start = text.lower().find("<html")
    if start != -1:
        text = text[start:]
    end = text.lower().rfind("</html>")
    if end != -1:
        text = text[:end + 7]

    return ChatResponse(response=text)


# --- /live WebSocket Endpoint (Gemini 2.0 Flash Live) ---
@app.websocket("/live")
async def live(websocket: WebSocket):
    await websocket.accept()
    client = get_client()

    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=SYSTEM_PROMPT)]
        ),
    )

    try:
        async with client.aio.live.connect(
            model="gemini-3.1-flash-live-preview",
            config=config,
        ) as session:
            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_text()
                        msg = json.loads(data)
                        text = msg.get("text", "")
                        if text:
                            await session.send_message(
                                types.LiveClientMessage(
                                    client_content=types.LiveClientContent(
                                        turns=[
                                            types.Content(
                                                role="user",
                                                parts=[types.Part.from_text(text=text)],
                                            )
                                        ],
                                        turn_complete=True,
                                    )
                                )
                            )
                except WebSocketDisconnect:
                    pass

            async def send_to_client():
                try:
                    async for response in session.receive():
                        if response.text:
                            await websocket.send_text(
                                json.dumps({"type": "text", "content": response.text})
                            )
                        if response.server_content and response.server_content.turn_complete:
                            await websocket.send_text(
                                json.dumps({"type": "done"})
                            )
                except WebSocketDisconnect:
                    pass

            await asyncio.gather(receive_from_client(), send_to_client())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
        except:
            pass


# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ui-agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)