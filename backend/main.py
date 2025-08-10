"""Kuber backend - FastAPI app that optionally serves a built frontend

Run backend only (development API):
  pip install -r backend/requirements.txt
  uvicorn backend.main:app --reload --port 8000

Run frontend (dev):
  cd frontend && npm install && npm run dev

Build frontend for production and let backend serve it:
  cd frontend && npm install && npm run build
  uvicorn backend.main:app --port 8000

This file provides:
- API endpoints (/api/*) for ingest, query, analytics
- Optional static serving of a built frontend placed at frontend/dist
- Unit-test friendly functions
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import uuid
from pathlib import Path
import os
import openai
import cohere

app = FastAPI(title="Kuber Agent API")

# mount static after api (we mount at /static to avoid swallowing /api)
DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST_DIR.exists():
    # serve built frontend assets at /static
    app.mount("/static", StaticFiles(directory=str(DIST_DIR)), name="static")

# -------------------------
# Simple in-memory vector store
# -------------------------
class TinyVectorStore:
    def __init__(self, dim=128):
        self.dim = dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}

    def add(self, text: str, metadata: Dict[str, Any] = None) -> str:
        vec = self._fake_embed(text)
        doc_id = str(uuid.uuid4())
        self.vectors[doc_id] = vec
        self.metadatas[doc_id] = {"text": text, **(metadata or {})}
        return doc_id

    def _fake_embed(self, text: str) -> np.ndarray:
        # deterministic pseudo-embedding for demo only (replace with real embeddings)
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        return rng.normal(size=(self.dim,)).astype(float)

    def query(self, q_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        qv = self._fake_embed(q_text)
        dists = []
        for doc_id, vec in self.vectors.items():
            sim = np.dot(qv, vec) / (np.linalg.norm(qv) * (np.linalg.norm(vec) + 1e-9))
            dists.append((sim, doc_id))
        dists.sort(reverse=True)
        results = []
        for sim, doc_id in dists[:top_k]:
            md = self.metadatas[doc_id].copy()
            md.update({"score": float(sim), "id": doc_id})
            results.append(md)
        return results

vector_store = TinyVectorStore(dim=128)

# -------------------------
# Pydantic models
# -------------------------
class IngestRequest(BaseModel):
    rows: List[Dict[str, Any]]

class QueryRequest(BaseModel):
    query: str
    context_k: int = 3

# -------------------------
# Analytics module (toy)
# -------------------------

def compute_simple_exposure(positions_df: pd.DataFrame) -> Dict[str, float]:
    # positions_df expected columns: ['ticker','quantity','price','beta']
    # crude exposure: sum(quantity * price * beta)
    positions_df = positions_df.copy()
    # coerce numeric columns
    for c in ['quantity', 'price', 'beta']:
        if c not in positions_df.columns:
            raise ValueError(f"positions_df must include column '{c}'")
        positions_df[c] = pd.to_numeric(positions_df[c], errors='coerce').fillna(0.0)
    positions_df['market_value'] = positions_df['quantity'] * positions_df['price']
    exposure = (positions_df['market_value'] * positions_df['beta']).sum()
    return {"total_exposure": float(exposure)}

# -------------------------
# LLM adapter (placeholder)
# -------------------------

# Make sure you have set one of these environment variables:
#   export OPENAI_API_KEY="your-key-here"
#   export COHERE_API_KEY="your-key-here"

def call_llm(prompt: str, max_tokens: int = 256, model: str = None) -> str:
    """
    Calls an LLM from OpenAI or Cohere depending on available API keys.
    Defaults to OpenAI GPT-4 if key is available, else Cohere Command R.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")

    if openai_key:
        openai.api_key = openai_key
        if not model:
            model = "gpt-4o-mini"  # Faster & cheaper, change to gpt-4o for higher quality
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"[Error calling OpenAI API: {e}]"

    elif cohere_key:
        co = cohere.Client(cohere_key)
        if not model:
            model = "command-r-plus"  # Cohere's reasoning model
        try:
            response = co.generate(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response.generations[0].text.strip()
        except Exception as e:
            return f"[Error calling Cohere API: {e}]"

    else:
        return "[Error: No API key found for OpenAI or Cohere]"

# Example usage:
if __name__ == "__main__":
    print(call_llm("Summarize the latest S&P 500 market trends in 3 bullet points."))

# -------------------------
# Endpoints
# -------------------------
@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    added = []
    for row in req.rows:
        text = row.get('text') or str(row)
        doc_id = vector_store.add(text, metadata=row)
        added.append(doc_id)
    return {"ingested": len(added), "ids": added}

@app.post("/api/query")
async def query(req: QueryRequest):
    contexts = vector_store.query(req.query, top_k=req.context_k)
    # build prompt
    prompt = "\\n---\\n".join([c['text'] for c in contexts]) + "\\n\\nUser: " + req.query
    llm_out = call_llm(prompt)
    return {"answer": llm_out, "contexts": contexts}

@app.post("/api/analytics/exposure")
async def analytics_exposure(positions: List[Dict[str, Any]]):
    # positions: JSON list of objects with ticker, quantity, price, beta
    df = pd.DataFrame(positions)
    required = {'ticker', 'quantity', 'price', 'beta'}
    if not required.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail="positions must include ticker, quantity, price, beta")
    return compute_simple_exposure(df)

# health
@app.get("/api/health")
async def health():
    return {"status": "ok"}

# Serve frontend index if present
@app.get("/")
async def root():
    index_path = Path(__file__).resolve().parent.parent / "frontend" / "dist" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    # helpful development message (prevents serving markdown/raw docs to /)
    return {"status": "backend running - build the frontend with 'cd frontend && npm run build' to serve the UI from /"}
