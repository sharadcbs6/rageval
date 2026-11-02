"""Graph-based context retriever using Groq LLM."""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_settings
from src.graph.client import get_neo4j_client
from src.logger import get_logger

logger = get_logger(__name__)

_ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract the most important named entities from the question. "
            "Return them as a comma-separated list with NO extra text.",
        ),
        ("human", "{question}"),
    ]
)


class GraphRetriever:
    """Retrieve graph context relevant to a user question."""

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )

    def _extract_question_entities(self, question: str) -> list[str]:
        chain = _ENTITY_EXTRACTION_PROMPT | self._llm
        response = chain.invoke({"question": question})
        raw = response.content if hasattr(response, "content") else str(response)
        entities = [e.strip() for e in raw.split(",") if e.strip()]
        logger.debug("question_entities", entities=entities)
        return entities

    def _fetch_subgraph(self, entities: list[str]) -> list[dict]:
        if not entities:
            return []
        client = get_neo4j_client()
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
        YIELD node, score
        WITH node LIMIT 10
        MATCH (node)-[r]-(neighbour:Entity)
        RETURN
            node.name        AS source,
            node.type        AS source_type,
            type(r)          AS relation,
            neighbour.name   AS target,
            neighbour.type   AS target_type
        LIMIT 50
        """
        query_str = " OR ".join(entities)
        try:
            records = client.run_query(cypher, {"query": query_str})
            logger.info("graph_records_fetched", count=len(records))
            return records
        except Exception as exc:
            logger.error("graph_retrieval_failed", error=str(exc))
            return []

    @staticmethod
    def _format_triples(records: list[dict]) -> str:
        if not records:
            return ""
        lines = ["Knowledge graph context:"]
        for r in records:
            lines.append(
                f"  ({r['source']}:{r['source_type']}) "
                f"-[{r['relation']}]-> "
                f"({r['target']}:{r['target_type']})"
            )
        return "\n".join(lines)

    def get_context(self, question: str) -> str:
        entities = self._extract_question_entities(question)
        records = self._fetch_subgraph(entities)
        return self._format_triples(records)

    def get_raw_records(self, question: str) -> list[dict]:
        entities = self._extract_question_entities(question)
        return self._fetch_subgraph(entities)
