# FitCoach AI

Fitness coaching web app with:
- React + Vite frontend
- Python backend (`ai_backend`)
- Optional Supabase Edge function (`supabase/functions/ai-coach`)

## Run frontend

```bash
npm install
npm run dev
```

## Run backend

```bash
cd ai_backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8010
```

## Free AI alternatives 

You can run the chat with free/local providers:
- Ollama local (recommended free): `http://127.0.0.1:11434`
- Groq free tier API
- Hugging Face Inference API (free tier)

### Example (Ollama)

1. Install Ollama.
2. Pull a model:
```bash
ollama pull llama3.1:8b
```
3. Keep Ollama running, then set backend env:
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## Supabase Edge function env (optional)

If using `supabase/functions/ai-coach`, set:
- `AI_GATEWAY_URL` (e.g. `http://127.0.0.1:11434/v1/chat/completions` or other OpenAI-compatible endpoint)
- `AI_MODEL` (e.g. `llama3.1:8b`)
- `AI_API_KEY` (only if your provider requires it)
