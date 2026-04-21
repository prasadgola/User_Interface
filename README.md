# UI Agent Server

FastAPI backend that responds as a senior UI/UX developer using Gemini. Text in, text out.

## Endpoint

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /chat` | HTTP | Text conversation |
| `GET /health` | HTTP | Health check |

## Request

```json
POST /chat
{
    "message": "Design a login screen for a fitness app",
    "history": [
        {"role": "user", "text": "Hi"},
        {"role": "model", "text": "Hey, what are we building?"}
    ]
}
```

`history` is optional — omit it for single-turn calls.

## Response

```json
{
    "response": "For a fitness app login I'd go with..."
}
```

## Local Dev

```bash
export GEMINI_API_KEY="your-key"
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

Test:
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Design a simple onboarding flow for a finance app"}'
```

## Deploy to Cloud Run (from scratch)

1. Install [gcloud CLI](https://cloud.google.com/sdk/docs/install) and run `gcloud auth login`
2. Get your billing account ID:
   ```bash
   gcloud billing accounts list
   ```
3. Edit `deploy.sh` — fill in `PROJECT_ID`, `BILLING_ACCOUNT`, and `GEMINI_API_KEY`
4. Run:
   ```bash
   chmod +x deploy.sh && ./deploy.sh
   ```

## Changing the Agent Persona

Edit `SYSTEM_PROMPT` in `main.py`. That's it.
