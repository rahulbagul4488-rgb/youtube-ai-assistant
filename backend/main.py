"""
YouTube AI Assistant — FastAPI Backend
RAG pipeline: LangChain + ChromaDB + Groq (free & fast)

Embeddings : sentence-transformers/all-MiniLM-L6-v2  (free, runs locally)
LLM        : llama3-8b-8192 via Groq API             (free & fast)
"""

import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LangChain
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# YouTube transcript
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
CHROMA_PERSIST = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL      = os.getenv("LLM_MODEL", "llama3-8b-8192")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K          = int(os.getenv("TOP_K", "4"))

# ── Shared singletons ─────────────────────────────────────────────────────────
embeddings = None
llm        = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, llm

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    # Embeddings — runs locally
    logger.info("Loading embedding model '%s' …", EMBED_MODEL)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_name=EMBED_MODEL,
    )
    logger.info("Embeddings ready ✅")

    # LLM — Groq API
    logger.info("Connecting to Groq API — model '%s' …", LLM_MODEL)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=512,
    )
    logger.info("LLM ready ✅")

    yield

    logger.info("Shutting down …")


app = FastAPI(
    title="YouTube AI Assistant (Groq)",
    description="Chat with any YouTube video — fully open-source stack.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    video_id: str

class IngestResponse(BaseModel):
    video_id: str
    status: str
    chunk_count: int

class ChatRequest(BaseModel):
    video_id: str
    question: str
    stream: bool = False

class ChatResponse(BaseModel):
    video_id: str
    question: str
    answer: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _collection_name(video_id: str) -> str:
    return f"yt_{video_id}"


def _collection_exists(video_id: str) -> bool:
    import chromadb
    client   = chromadb.PersistentClient(path=CHROMA_PERSIST)
    existing = [c.name for c in client.list_collections()]
    return _collection_name(video_id) in existing


def _fetch_transcript(video_id: str) -> str:
    """Fetch transcript using new youtube-transcript-api."""
    try:
        ytt_api    = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        return " ".join(entry.text for entry in transcript)
    except TranscriptsDisabled:
        raise HTTPException(
            status_code=422,
            detail=f"Transcripts are disabled for video '{video_id}'.",
        )
    except NoTranscriptFound:
        raise HTTPException(
            status_code=422,
            detail=f"No transcript found for '{video_id}'. Try another video.",
        )
    except Exception as exc:
        logger.exception("Unexpected transcript error")
        raise HTTPException(status_code=500, detail=str(exc))


def _get_vectorstore(video_id: str) -> Chroma:
    return Chroma(
        collection_name=_collection_name(video_id),
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST,
    )


# ── RAG Prompt ────────────────────────────────────────────────────────────────
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


def _build_rag_chain(video_id: str):
    retriever = _get_vectorstore(video_id).as_retriever(
        search_kwargs={"k": TOP_K}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "embed_model": EMBED_MODEL, "llm": LLM_MODEL}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Index a YouTube video into ChromaDB."""
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id must not be empty.")

    # Already indexed? Skip
    if _collection_exists(video_id):
        logger.info("'%s' already indexed — skipping.", video_id)
        count = _get_vectorstore(video_id)._collection.count()
        return IngestResponse(
            video_id=video_id, status="already_exists", chunk_count=count
        )

    # Fetch transcript
    logger.info("Fetching transcript for '%s' …", video_id)
    raw_text = _fetch_transcript(video_id)
    logger.info("Transcript: %d characters", len(raw_text))

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents(
        texts=[raw_text],
        metadatas=[{"video_id": video_id}],
    )
    logger.info("Split into %d chunks.", len(chunks))

    # Embed and persist
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_collection_name(video_id),
        persist_directory=CHROMA_PERSIST,
    )
    logger.info("Ingestion complete for '%s' ✅", video_id)

    return IngestResponse(
        video_id=video_id, status="ingested", chunk_count=len(chunks)
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Answer a question about an already-indexed YouTube video."""
    video_id = req.video_id.strip()
    question = req.question.strip()

    if not video_id or not question:
        raise HTTPException(
            status_code=400, detail="Both video_id and question are required."
        )
    if not _collection_exists(video_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{video_id}' is not indexed. Call /ingest first.",
        )

    chain = _build_rag_chain(video_id)

    # Streaming
    if req.stream:
        async def token_stream():
            async for token in chain.astream(question):
                yield token
        return StreamingResponse(token_stream(), media_type="text/plain")

    # Standard
    try:
        answer = await chain.ainvoke(question)
    except Exception as exc:
        logger.exception("RAG chain error")
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(video_id=video_id, question=question, answer=answer)
