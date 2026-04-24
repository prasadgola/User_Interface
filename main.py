import os
import asyncio
import json
import base64
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
UI_SYSTEM_PROMPT = """You are a UI generator.
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


# --- /chat Endpoint (text → HTML) ---
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
            system_instruction=UI_SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=8192,
        ),
    )

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


# --- /live WebSocket Endpoint (text in, text out) ---
@app.websocket("/live")
async def live(websocket: WebSocket):
    await websocket.accept()
    client = get_client()

    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=UI_SYSTEM_PROMPT)]
        ),
    )

    try:
        async with client.aio.live.connect(
            model="gemini-3.1-flash-live-preview",
            config=config,
        ) as session:
            print(f"[live] Gemini session connected: {session}")

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
                            await websocket.send_text(json.dumps({"type": "done"}))
                except WebSocketDisconnect:
                    pass

            await asyncio.gather(receive_from_client(), send_to_client())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[live] Error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
        except:
            pass


# --- /voice WebSocket Endpoint (audio in, audio out + UI HTML at end) ---
@app.websocket("/voice")
async def voice(websocket: WebSocket):
    await websocket.accept()
    client = get_client()

    transcript_parts = []

    try:
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=UI_SYSTEM_PROMPT)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Fenrir"
                    )
                )
            ),
        )

        async with client.aio.live.connect(
            model="gemini-3.1-flash-live-preview",
            config=config,
        ) as session:
            print(f"[voice] Gemini session connected: {session}")

            async def receive_and_forward():
                nonlocal transcript_parts
                try:
                    while True:
                        data = await websocket.receive()

                        if "bytes" in data:
                            print(f"[voice] Sending audio chunk: {len(data['bytes'])} bytes")
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        types.Blob(
                                            data=data["bytes"],
                                            mime_type="audio/pcm;rate=16000",
                                        )
                                    ]
                                )
                            )

                        elif "text" in data:
                            msg = json.loads(data["text"])

                            if msg.get("type") == "image":
                                image_bytes = base64.b64decode(msg["data"])
                                print(f"[voice] Sending image frame: {len(image_bytes)} bytes")
                                await session.send(
                                    input=types.LiveClientRealtimeInput(
                                        media_chunks=[
                                            types.Blob(
                                                data=image_bytes,
                                                mime_type="image/jpeg",
                                            )
                                        ]
                                    )
                                )

                            elif msg.get("type") == "end":
                                print(f"[voice] End received, transcript: {transcript_parts}")
                                combined = " ".join(transcript_parts).strip()
                                if combined:
                                    ui_html = await generate_ui(client, combined)
                                    await websocket.send_text(
                                        json.dumps({"type": "ui", "content": ui_html})
                                    )
                                else:
                                    print("[voice] No transcript collected, skipping UI generation")

                            elif msg.get("type") == "close":
                                await session.close()
                                return

                except WebSocketDisconnect:
                    pass

            async def receive_and_send_response():
                try:
                    async for response in session.receive():
                        print(f"[voice] Response: data={bool(response.data)} text={response.text}")
                        if response.data:
                            await websocket.send_bytes(response.data)
                        if response.text:
                            transcript_parts.append(response.text)
                            await websocket.send_text(
                                json.dumps({"type": "transcript", "text": response.text})
                            )
                        if response.server_content and response.server_content.turn_complete:
                            await websocket.send_text(json.dumps({"type": "turn_complete"}))
                except Exception as e:
                    print(f"[voice] Error receiving from Gemini: {e}")

            await asyncio.gather(receive_and_forward(), receive_and_send_response())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[voice] Session error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
        except:
            pass


async def generate_ui(client, transcript: str) -> str:
    prompt = f"Based on this conversation: '{transcript}' — generate a relevant mobile UI."
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3-flash-preview",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=UI_SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=8192,
        ),
    )
    text = response.text.strip()
    start = text.lower().find("<!doctype")
    if start == -1:
        start = text.lower().find("<html")
    if start != -1:
        text = text[start:]
    end = text.lower().rfind("</html>")
    if end != -1:
        text = text[:end + 7]
    return text


# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ui-agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)