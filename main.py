"""
Local-first RAG Contract Q&A — MVP demo.
FastAPI + FAISS + OpenAI. Data stays local; the only external calls are to OpenAI
(embeddings + answer generation), exactly as the spec requires.
"""
import os
import io
import json
import uuid
import pickle

import numpy as np
import faiss
import pdfplumber
import docx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

# ---------------------------------------------------------------- config
EMBED_MODEL = "text-embedding-3-small"      # 1536 dims
CHAT_MODEL = "gpt-4o-mini"
EMBED_DIM = 1536
TOP_K = 5
CHUNK_SIZE = 900        # chars
CHUNK_OVERLAP = 150

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
STORE_PATH = os.path.join(DATA_DIR, "store.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# Key is provided at runtime via the OPENAI_API_KEY env var. We fall back to a
# placeholder so the server still boots for a UI walkthrough; real API calls
# (upload/chat) need a valid key set.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "sk-set-your-key")

# ---------------------------------------------------------------- storage
# FAISS index + a parallel list of chunk metadata (same order as vectors).
if os.path.exists(INDEX_PATH) and os.path.exists(STORE_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(STORE_PATH, "rb") as f:
        STORE = pickle.load(f)   # {"chunks": [ {id, doc_id, doc_name, page, text} ], "docs": {doc_id: name}}
else:
    index = faiss.IndexFlatIP(EMBED_DIM)
    STORE = {"chunks": [], "docs": {}}


def persist():
    faiss.write_index(index, INDEX_PATH)
    with open(STORE_PATH, "wb") as f:
        pickle.dump(STORE, f)


# ---------------------------------------------------------------- helpers
def embed(texts):
    """Return a normalized float32 matrix of embeddings."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def chunk_text(text):
    """Simple overlapping character chunks."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def extract_pages(filename, raw):
    """Return list of (page_label, text). PDFs keep real page numbers."""
    name = filename.lower()
    if name.endswith(".pdf"):
        out = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                out.append((f"p.{i}", page.extract_text() or ""))
        return out
    if name.endswith(".docx"):
        d = docx.Document(io.BytesIO(raw))
        full = "\n".join(p.text for p in d.paragraphs)
        return [("—", full)]
    if name.endswith(".txt"):
        return [("—", raw.decode("utf-8", errors="ignore"))]
    raise HTTPException(400, "Unsupported file type. Use PDF, DOCX, or TXT.")


# ---------------------------------------------------------------- app
app = FastAPI(title="Contract RAG Demo")


@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/documents")
def documents():
    return [{"id": did, "name": name} for did, name in STORE["docs"].items()]


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    pages = extract_pages(file.filename, raw)

    doc_id = uuid.uuid4().hex[:8]
    new_texts, new_meta = [], []
    for page_label, page_text in pages:
        for ch in chunk_text(page_text):
            new_texts.append(ch)
            new_meta.append({
                "id": uuid.uuid4().hex,
                "doc_id": doc_id,
                "doc_name": file.filename,
                "page": page_label,
                "text": ch,
            })

    if not new_texts:
        raise HTTPException(400, "Could not extract any text from this file.")

    # embed in batches of 100
    for i in range(0, len(new_texts), 100):
        batch = new_texts[i:i + 100]
        index.add(embed(batch))

    STORE["chunks"].extend(new_meta)
    STORE["docs"][doc_id] = file.filename
    persist()
    return {"doc_id": doc_id, "name": file.filename, "chunks": len(new_meta)}


@app.post("/chat")
async def chat(question: str = Form(...), doc_id: str = Form("")):
    if index.ntotal == 0:
        raise HTTPException(400, "No documents uploaded yet.")

    qvec = embed([question])
    # over-fetch so we can filter by document if requested
    D, I = index.search(qvec, min(30, index.ntotal))

    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = STORE["chunks"][idx]
        if doc_id and meta["doc_id"] != doc_id:
            continue
        hits.append(meta)
        if len(hits) >= TOP_K:
            break

    if not hits:
        return {"answer": "I couldn't find anything relevant in the selected document(s).", "citations": []}

    context = "\n\n".join(
        f"[{h['doc_name']} {h['page']}]\n{h['text']}" for h in hits
    )
    system = (
        "You are a contract analysis assistant. Answer ONLY from the provided context. "
        "If the answer isn't in the context, say you couldn't find it. "
        "Always cite the source as [document name, page] inline where relevant."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.1,
    )
    answer = resp.choices[0].message.content

    citations = [{"doc": h["doc_name"], "page": h["page"]} for h in hits]
    # de-dup citations, keep order
    seen, uniq = set(), []
    for c in citations:
        key = (c["doc"], c["page"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return {"answer": answer, "citations": uniq}


@app.post("/reset")
def reset():
    global index
    index = faiss.IndexFlatIP(EMBED_DIM)
    STORE["chunks"], STORE["docs"] = [], {}
    persist()
    return {"ok": True}
