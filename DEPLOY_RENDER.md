# Deploy FitCoach AI for Free (Render)

This repo includes `render.yaml` so Render can deploy both services:

- `fit-coach-ai-backend` (FastAPI)
- `fit-coach-ai-frontend` (Vite static app)

## 1) Push latest code

If you are not already up to date:

```bash
git push origin main
```

## 2) One-click deploy

Open:

`https://render.com/deploy?repo=https://github.com/mohannad-121/fit_coach_ai`

Render will read `render.yaml` and create both services.

## 3) Set required frontend env vars in Render

On service `fit-coach-ai-frontend`, set:

- `VITE_AI_BACKEND_URL` (set this to your backend Render URL, for example `https://fit-coach-ai-backend.onrender.com`)
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY` (or `VITE_SUPABASE_PUBLISHABLE_KEY`)

## 4) Optional AI provider key

Backend is configured with `CHAT_RESPONSE_MODE=dataset_only` so it can run free without paid LLM APIs.

If you want OpenAI responses, set on backend service:

- `OPENAI_API_KEY`

and change:

- `CHAT_RESPONSE_MODE=ai_hybrid`

## 5) Verify

- Backend health: `https://<backend-domain>/health`
- Frontend app: `https://<frontend-domain>`
