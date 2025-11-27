from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from chunked_pooling import chunked_pooling, chunk_by_sentences, chunk_semantically
import torch

app = FastAPI()
model_name = 'jinaai/jina-embeddings-v2-base-de'

# Set the correct max length for Jina v2 models
JINA_MAX_LENGTH = 8192

# Load model/tokenizer once at startup (late chunking-friendly model)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, model_max_length=JINA_MAX_LENGTH)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()

class ChunkRequest(BaseModel):
    doc_id: str
    text: str
    chunk_size: int | None = None

class ChunkResponse(BaseModel):
    doc_id: str
    model_name: str
    chunks: list[str]
    embeddings: list[list[float]]  # JSON-friendly
    span_annotations: list[tuple[int, int]]    # optional: token spans per chunk

@app.post("/chunk", response_model=ChunkResponse)
def chunk_text(req: ChunkRequest):
    # Use chunk_size if provided, otherwise use max_tokens

    # Split into chunks and get span annotations (token counts per chunk)
    # chunks, span_annotations = chunk_by_sentences(req.text, tokenizer)
    chunks, span_annotations = chunk_semantically(req.text, tokenizer, model_name, req.chunk_size)

    # Single full forward pass (late chunking)
    inputs = tokenizer(req.text, return_tensors='pt', max_length=JINA_MAX_LENGTH, truncation=True)
    with torch.no_grad():
        output = model(**inputs)

    # Pool token-level states into per-chunk embeddings
    pooled = chunked_pooling(output, [span_annotations])[0]  # list of vectors (torch/numpy)

    # Convert to plain lists for JSON
    embeddings = [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in pooled]

    return ChunkResponse(
        doc_id=req.doc_id,
        model_name=model_name,
        chunks=chunks,
        embeddings=embeddings,
        span_annotations=span_annotations
    )