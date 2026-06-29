# Contract RAG — Local-first Q&A demo

Upload contracts (PDF / DOCX / TXT), ask natural-language questions, get answers
with document + page citations. Runs entirely local; the only external calls are
to the OpenAI API (embeddings + answer generation).

Stack: FastAPI · FAISS · OpenAI (text-embedding-3-small + gpt-4o-mini) · pdfplumber / python-docx

## Run locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

## Deploy on a VPS (quick)

```bash
# on the server
git clone <repo> rag-demo && cd rag-demo
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."

# run it (0.0.0.0 so it's reachable; pick any port)
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then point the client at `http://YOUR_VPS_IP:8000`.

To keep it running after you log out:
```bash
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &
```

Optional: put nginx in front for a clean domain + HTTPS.

## Adding documents

Just upload them in the UI — they're chunked, embedded, and added to the local
FAISS index immediately. Data persists in `data/` (`index.faiss` + `store.pkl`).

## Notes

- Search scope dropdown = ask across all docs or restrict to one.
- `POST /reset` clears the index if you want a clean demo.
- This is an MVP/POC: pragmatic code, no test suite (by request).
