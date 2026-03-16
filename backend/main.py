"""
YouTube AI Assistant — FastAPI Backend
Fast startup version — Nomic + ChromaDB + Supadata + Groq
"""

import os
import logging
import httpx
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from nomic import embed
import nomic

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
NOMIC_API_KEY    = os.getenv("NOMIC_API_KEY", "")
SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY", "")
CHROMA_PERSIST   = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
LLM_MODEL        = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K            = int(os.getenv("TOP_K", "4"))

# Initialize at module level — fast startup
nomic.login(NOMIC_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=LLM_MODEL,
    temperature=0.1,
    max_tokens=512,
)

app = FastAPI(title="YouTube AI Assistant", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    video_id: str

class IngestResponse(BaseModel):
    video_id: str
    status: str
    chunk_count: int

class ChatRequest(BaseModel):
    video_id: str
    question: str

class ChatResponse(BaseModel):
    video_id: str
    question: str
    answer: str


def _collection_name(video_id: str) -> str:
    return f"yt_{video_id}"


def _collection_exists(video_id: str) -> bool:
    existing = [c.name for c in chroma_client.list_collections()]
    return _collection_name(video_id) in existing


def _get_embeddings(texts: list) -> list:
    output = embed.text(
        texts=texts,
        model="nomic-embed-text-v1.5",
        task_type="search_document",
    )
    return output["embeddings"]


def _get_query_embedding(query: str) -> list:
    output = embed.text(
        texts=[query],
        model="nomic-embed-text-v1.5",
        task_type="search_query",
    )
    return output["embeddings"][0]


async def _fetch_transcript(video_id: str) -> str:
    url     = "https://api.supadata.ai/v1/youtube/transcript"
    headers = {"x-api-key": SUPADATA_API_KEY}
    params  = {"videoId": video_id, "text": "true"}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=headers, params=params)

    if response.status_code == 404:
        raise HTTPException(status_code=422, detail=f"No transcript for '{video_id}'.")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Supadata error: {response.text}")

    data = response.json()
    if isinstance(data.get("content"), str):
        return data["content"]
    elif isinstance(data.get("content"), list):
        return " ".join(item.get("text", "") for item in data["content"])
    else:
        raise HTTPException(status_code=500, detail="Unexpected transcript format.")


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions strictly based on the
YouTube video transcript provided below.
If the answer is not present in the transcript, respond with:
"I don't know based on the video content."
Never make up information.

Transcript context:
{context}

Question: {question}

Answer:"""
)


@app.get("/health")
async def health():
    return {"status": "ok", "llm": LLM_MODEL}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id required.")

    if _collection_exists(video_id):
        logger.info("'%s' already indexed.", video_id)
        col   = chroma_client.get_collection(_collection_name(video_id))
        count = col.count()
        return IngestResponse(video_id=video_id, status="already_exists", chunk_count=count)

    logger.info("Fetching transcript for '%s' …", video_id)
    transcript = await _fetch_transcript(video_id)
    logger.info("Transcript: %d chars", len(transcript))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(transcript)
    logger.info("Split into %d chunks.", len(chunks))

    logger.info("Generating embeddings via Nomic …")
    embeddings = _get_embeddings(chunks)

    col = chroma_client.create_collection(name=_collection_name(video_id))
    col.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{video_id}_{i}" for i in range(len(chunks))]
    )
    logger.info("Indexed %d chunks ✅", len(chunks))

    return IngestResponse(video_id=video_id, status="ingested", chunk_count=len(chunks))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    video_id = req.video_id.strip()
    question = req.question.strip()

    if not video_id or not question:
        raise HTTPException(status_code=400, detail="video_id and question required.")
    if not _collection_exists(video_id):
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not indexed.")

    query_embedding = _get_query_embedding(question)
    col     = chroma_client.get_collection(_collection_name(video_id))
    results = col.query(query_embeddings=[query_embedding], n_results=TOP_K)
    context = "\n\n".join(results["documents"][0])

    prompt   = RAG_PROMPT.format_messages(context=context, question=question)
    response = await llm.ainvoke(prompt)

    return ChatResponse(video_id=video_id, question=question, answer=response.content)
