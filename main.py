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
from datetime import datetime, timezone

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


SYSTEM_PROMPT = (
    "You are Clause, a friendly assistant that helps people understand their contracts.\n\n"
    "Decide how to respond based on what the user says:\n"
    "1. If they are greeting you, making small talk, or thanking you — reply naturally and "
    "briefly, like a real person would, in one short sentence. Do NOT tack on a phrase like "
    "'let me know if you have questions about your contracts' — it's repetitive and pushy. Only "
    "mention contracts if it's the very first message or they explicitly ask what you do. Vary "
    "your wording and never repeat the same closing line. Do NOT use the document context. Set "
    "mode to \"chat\".\n"
    "2. If they ask which documents/files/contracts you have or can see, answer from the "
    "'Documents currently loaded' list given to you — just name the files. Set mode to "
    "\"chat\".\n"
    "3. If they ask a question about the content of their documents — answer using ONLY the "
    "provided context. Quote and explain the relevant part. If the answer is genuinely not in "
    "the context, say you couldn't find it. Set mode to \"document\".\n\n"
    "Always respond as JSON: {\"mode\": \"chat\" | \"document\", \"answer\": \"<your reply>\"}."
)


@app.post("/chat")
async def chat(question: str = Form(...), doc_id: str = Form(""), history: str = Form("")):
    q = question.strip()
    if not q:
        return {"answer": "Ask me something about your contracts.", "citations": []}

    # Recent conversation, so follow-ups like "what about termination?" make sense.
    try:
        prior = json.loads(history) if history else []
        prior = [m for m in prior if m.get("role") in ("user", "assistant")][-8:]
    except json.JSONDecodeError:
        prior = []

    # Retrieve context only if there are documents.
    hits = []
    if index.ntotal > 0:
        qvec = embed([q])
        D, I = index.search(qvec, min(30, index.ntotal))
        for idx in I[0]:
            if idx < 0:
                continue
            meta = STORE["chunks"][idx]
            if doc_id and meta["doc_id"] != doc_id:
                continue
            hits.append(meta)
            if len(hits) >= TOP_K:
                break

    context = "\n\n".join(f"[{h['doc_name']} {h['page']}]\n{h['text']}" for h in hits) \
        or "(nothing retrieved)"
    loaded = ", ".join(STORE["docs"].values()) or "(none yet)"
    user = (f"Documents currently loaded: {loaded}\n\n"
            f"Relevant excerpts for this question:\n{context}\n\nUser message: {q}")

    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
    system = f"{SYSTEM_PROMPT}\n\nFor reference, today's date is {today}."
    messages = [{"role": "system", "content": system}]
    messages += prior
    messages.append({"role": "user", "content": user})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
        answer = parsed.get("answer", "").strip()
        mode = parsed.get("mode", "chat")
    except (json.JSONDecodeError, AttributeError):
        answer = resp.choices[0].message.content
        mode = "document"

    # Only show citations for real document answers that actually found something.
    low = answer.lower()
    failed = any(p in low for p in ["couldn't find", "could not find", "not in the context",
                                    "no relevant", "isn't in", "is not in", "don't have"])
    if mode != "document" or not hits or failed:
        return {"answer": answer, "citations": []}

    seen, uniq = set(), []
    for h in hits:
        key = (h["doc_name"], h["page"])
        if key not in seen:
            seen.add(key)
            uniq.append({"doc": h["doc_name"], "page": h["page"]})
    return {"answer": answer, "citations": uniq}


@app.post("/delete")
def delete_doc(doc_id: str = Form(...)):
    global index
    if doc_id not in STORE["docs"]:
        raise HTTPException(404, "Document not found.")

    keep_mask = [c["doc_id"] != doc_id for c in STORE["chunks"]]

    # Rebuild the index from the chunks we're keeping. We reconstruct the existing
    # vectors straight from FAISS, so there's no need to re-embed (no API cost).
    new_index = faiss.IndexFlatIP(EMBED_DIM)
    if index.ntotal > 0 and any(keep_mask):
        all_vecs = index.reconstruct_n(0, index.ntotal)
        kept = all_vecs[np.array(keep_mask)]
        new_index.add(kept.astype("float32"))
    index = new_index

    STORE["chunks"] = [c for c, k in zip(STORE["chunks"], keep_mask) if k]
    del STORE["docs"][doc_id]
    persist()
    return {"ok": True}


@app.post("/reset")
def reset():
    global index
    index = faiss.IndexFlatIP(EMBED_DIM)
    STORE["chunks"], STORE["docs"] = [], {}
    persist()
    return {"ok": True}
