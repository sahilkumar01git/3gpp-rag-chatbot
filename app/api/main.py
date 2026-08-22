from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.logging_config import configure_logging
from app.pipeline import RagPipeline

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting up — loading (or building) the FAISS index and models ...")
    app.state.pipeline = RagPipeline()
    logger.info("Startup complete. Index has %d vectors.", app.state.pipeline.vector_store.index.ntotal)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="3GPP RAG Chatbot",
    description="Retrieval-augmented Q&A over 3GPP technical specifications, with citation-backed, "
    "confidence-gated answers.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Check server logs."})


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
