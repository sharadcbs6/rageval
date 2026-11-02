"""Neo4j driver singleton with connection pooling, keep-alive, and auto-reconnect."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import (
    ServiceUnavailable,
    SessionExpired,
    ClientError,
)

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)

# Max retries for transient connection failures
MAX_RETRIES = 3


class Neo4jClient:
    """Thread-safe Neo4j driver wrapper with auto-reconnect."""

    _driver: Driver | None = None

    def __init__(self) -> None:
        settings = get_settings()
        self._uri = settings.neo4j_uri
        self._auth = (settings.neo4j_username, settings.neo4j_password)
        self._database = settings.neo4j_database

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=self._auth,
                # AuraDB free tier drops idle connections after ~5 min.
                # Setting max_connection_lifetime below the idle timeout
                # forces the driver to proactively refresh stale connections
                # before they get killed by the server.
                max_connection_lifetime=180,       # 3 min — recycle before Aura drops
                max_connection_pool_size=10,
                connection_timeout=15,
                keep_alive=True,                    # send TCP keepalive probes
            )
            self._driver.verify_connectivity()
            logger.info("neo4j_connected", uri=self._uri)

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("neo4j_disconnected")

    def _reconnect(self) -> None:
        """Close the stale driver and create a fresh one."""
        logger.warning("neo4j_reconnecting")
        try:
            if self._driver:
                self._driver.close()
        except Exception:
            pass
        self._driver = None
        self.connect()

    # ── session helpers ───────────────────────────────────────────────────────

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self._database) as s:  # type: ignore[union-attr]
            yield s

    # ── query helpers ─────────────────────────────────────────────────────────

    def run_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query with automatic retry on transient connection failures.

        AuraDB free tier can drop idle connections, causing SessionExpired or
        ServiceUnavailable errors. We catch those, reconnect, and retry.
        """
        params = params or {}
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with self.session() as s:
                    result = s.run(cypher, **params)
                    return [record.data() for record in result]
            except (SessionExpired, ServiceUnavailable) as exc:
                last_exc = exc
                logger.warning(
                    "neo4j_retry",
                    attempt=attempt,
                    max_retries=MAX_RETRIES,
                    error=str(exc)[:120],
                )
                self._reconnect()
            except ClientError as exc:
                # Syntax / schema errors shouldn't be retried
                logger.error("neo4j_cypher_error", error=str(exc)[:200])
                raise

        # All retries exhausted
        logger.error("neo4j_query_failed_after_retries", error=str(last_exc)[:200] if last_exc else "unknown")
        raise last_exc  # type: ignore[misc]

    def health_check(self) -> bool:
        try:
            self.run_query("RETURN 1 AS ok")
            return True
        except Exception:
            return False

    # ── schema helpers ────────────────────────────────────────────────────────

    def create_indexes(self) -> None:
        """Create baseline full-text and vector indexes."""
        statements = [
            """
            CREATE CONSTRAINT entity_id IF NOT EXISTS
            FOR (e:Entity) REQUIRE (e.name, e.type) IS NODE KEY
            """,
            """
            CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
            FOR (e:Entity) ON EACH [e.name, e.description]
            """,
        ]
        with self.session() as s:
            for stmt in statements:
                s.run(stmt)
        logger.info("neo4j_indexes_created")


# ── module-level singleton ─────────────────────────────────────────────────────

_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
        _client.connect()
    return _client
