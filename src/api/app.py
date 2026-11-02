"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.logger import configure_logging, get_logger
from src.api.routers import ingest, query, evaluate, graph

logger = get_logger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("app_starting", env=settings.app_env)

    # Eagerly connect and ensure Neo4j indexes exist
    try:
        from src.graph.client import get_neo4j_client  # noqa: PLC0415
        client = get_neo4j_client()
        client.create_indexes()
        logger.info("neo4j_ready")
    except Exception as exc:
        logger.warning("neo4j_startup_warning", error=str(exc))

    # Ping ChromaDB
    try:
        from src.vectorstore.store import get_vector_store  # noqa: PLC0415
        vs = get_vector_store()
        healthy = vs.health_check()
        logger.info("chroma_ready", healthy=healthy)
    except Exception as exc:
        logger.warning("chroma_startup_warning", error=str(exc))

    yield  # ← app is live

    # Graceful shutdown
    try:
        from src.graph.client import get_neo4j_client  # noqa: PLC0415
        get_neo4j_client().close()
    except Exception:
        pass
    logger.info("app_shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Evaluation Platform",
        description=(
            "End-to-end RAG system with LangChain, LangGraph, Neo4j knowledge graphs, "
            "ChromaDB vector store, and dual evaluation via Ragas & DeepEval."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(evaluate.router)
    app.include_router(graph.router)

    # ── Root health check ─────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"], summary="Platform liveness check.")
    async def health() -> dict:
        from src.graph.client import get_neo4j_client  # noqa: PLC0415
        from src.vectorstore.store import get_vector_store  # noqa: PLC0415

        neo4j_ok = False
        chroma_ok = False
        try:
            neo4j_ok = get_neo4j_client().health_check()
        except Exception:
            pass
        try:
            chroma_ok = get_vector_store().health_check()
        except Exception:
            pass

        return {
            "status": "ok",
            "neo4j": neo4j_ok,
            "chroma": chroma_ok,
        }

    return app


# ── Singleton for Uvicorn ─────────────────────────────────────────────────────
app = create_app()
