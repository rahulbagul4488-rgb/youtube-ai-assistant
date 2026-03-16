"""
YouTube AI Assistant — FastAPI Backend
RAG pipeline: LangChain + ChromaDB + Groq

Transcript fetched via Supadata API (avoids YouTube IP blocks on cloud)
Embeddings via HuggingFace Inference API
LLM via Groq API
"""

import os
import logging
import httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
HF_TOKEN          = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
SUPADATA_API_KEY  = os.getenv("SUPADATA_API_KEY", "")
CHROMA_PERSIST    = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL       = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL         = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K             = int(os.getenv("TOP_K", "4"))

embeddings = None
llm        = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, llm

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set.")
    if not SUPADATA_API_KEY:
        raise RuntimeError("SUPADATA_API_KEY is not set.")

    logger.info("Loading embeddings via HF Inference API …")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name=EMBED_MODEL,
    )
    logger.info("Embeddings ready ✅")

    logger.info("Connecting to Groq — model '%s' …", LLM_MODEL)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=512,
    )
    logger.info("LLM ready ✅")

    yield
    logger.info("Shutting down …")


app = FastAPI(title="YouTube AI Assistant", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
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


def _get_vectorstore(video_id: str) -> Chroma:
    return Chroma(
        collection_name=_collection_name(video_id),
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST,
    )


async def _fetch_transcript_supadata(video_id: str) -> str:
    """Fetch transcript using Supadata API — works from cloud servers."""
    url = f"https://api.supadata.ai/v1/youtube/transcript"
    headers = {"x-api-key": SUPADATA_API_KEY}
    params  = {"videoId": video_id, "text": "true"}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=headers, params=params)

    if response.status_code == 404:
        raise HTTPException(
            status_code=422,
            detail=f"No transcript found for video '{video_id}'. Try another video."
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Supadata API error: {response.status_code} — {response.text}"
        )

    data = response.json()

    # Supadata returns transcript as string or list
    if isinstance(data.get("content"), str):
        return data["content"]
    elif isinstance(data.get("content"), list):
        return " ".join(item.get("text", "") for item in data["content"])
    else:
        raise HTTPException(status_code=500, detail="Unexpected transcript format from Supadata.")


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
    retriever = _get_vectorstore(video_id).as_retriever(search_kwargs={"k": TOP_K})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "llm": LLM_MODEL}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Fetch transcript via Supadata API and index into ChromaDB."""
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required.")

    # Already indexed?
    if _collection_exists(video_id):
        logger.info("'%s' already indexed.", video_id)
        count = _get_vectorstore(video_id)._collection.count()
        return IngestResponse(video_id=video_id, status="already_exists", chunk_count=count)

    # Fetch transcript via Supadata
    logger.info("Fetching transcript for '%s' via Supadata …", video_id)
    transcript = await _fetch_transcript_supadata(video_id)
    logger.info("Transcript: %d chars", len(transcript))

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents(
        texts=[transcript],
        metadatas=[{"video_id": video_id}],
    )

    # Embed and persist
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_collection_name(video_id),
        persist_directory=CHROMA_PERSIST,
    )
    logger.info("Done — %d chunks ✅", len(chunks))

    return IngestResponse(video_id=video_id, status="ingested", chunk_count=len(chunks))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Answer a question about an already-indexed YouTube video."""
    video_id = req.video_id.strip()
    question = req.question.strip()

    if not video_id or not question:
        raise HTTPException(status_code=400, detail="video_id and question required.")
    if not _collection_exists(video_id):
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not indexed.")

    chain = _build_rag_chain(video_id)

    if req.stream:
        async def token_stream():
            async for token in chain.astream(question):
                yield token
        return StreamingResponse(token_stream(), media_type="text/plain")

    try:
        answer = await chain.ainvoke(question)
    except Exception as exc:
        logger.exception("RAG chain error")
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(video_id=video_id, question=question, answer=answer)
