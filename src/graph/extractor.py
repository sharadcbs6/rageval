"""Entity and relationship extractor using LangChain + Groq.

Extracts structured (entity, relation, entity) triples from raw text
and upserts them into Neo4j as a knowledge graph.
"""

from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graph.client import get_neo4j_client
from src.logger import get_logger

logger = get_logger(__name__)


# ── Pydantic models for structured LLM output ─────────────────────────────────

class Entity(BaseModel):
    name: str = Field(description="Canonical entity name.")
    type: str = Field(description="Entity type, e.g. Person, Organization, Concept.")
    description: str = Field(default="", description="Short description.")


class Relation(BaseModel):
    source: str = Field(description="Source entity name.")
    relation: str = Field(description="Relation type in UPPER_SNAKE_CASE.")
    target: str = Field(description="Target entity name.")


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a knowledge-graph extraction engine.
Given a passage of text, extract all named entities and the relationships between them.
Return ONLY valid JSON that matches the schema below — no prose, no markdown fences.

Schema:
{format_instructions}
"""

_HUMAN = "Text:\n{text}"


# ── Extractor class ───────────────────────────────────────────────────────────

class KnowledgeGraphExtractor:
    """Extract triples from text and persist them to Neo4j."""

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
        self._parser = PydanticOutputParser(pydantic_object=ExtractionResult)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
                ("human", _HUMAN),
            ]
        ).partial(format_instructions=self._parser.get_format_instructions())

    def extract(self, text: str) -> ExtractionResult:
        chain = self._prompt | self._llm | self._parser
        try:
            result: ExtractionResult = chain.invoke({"text": text})
            logger.info(
                "extraction_complete",
                entities=len(result.entities),
                relations=len(result.relations),
            )
            return result
        except Exception as exc:
            logger.error("extraction_failed", error=str(exc))
            return ExtractionResult()

    def upsert_to_graph(self, result: ExtractionResult) -> dict[str, int]:
        client = get_neo4j_client()
        nodes_created = 0
        rels_created = 0

        for entity in result.entities:
            rows = client.run_query(
                """
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET e.description = $description, e.created_at = timestamp()
                ON MATCH  SET e.description = $description
                RETURN e
                """,
                {"name": entity.name, "type": entity.type, "description": entity.description},
            )
            if rows:
                nodes_created += 1

        for rel in result.relations:
            rows = client.run_query(
                f"""
                MATCH (a:Entity {{name: $source}})
                MATCH (b:Entity {{name: $target}})
                MERGE (a)-[r:{rel.relation}]->(b)
                ON CREATE SET r.created_at = timestamp()
                RETURN r
                """,
                {"source": rel.source, "target": rel.target},
            )
            if rows:
                rels_created += 1

        logger.info("graph_upsert", nodes=nodes_created, relations=rels_created)
        return {"nodes_created": nodes_created, "relations_created": rels_created}

    def extract_and_store(self, text: str) -> dict[str, int]:
        result = self.extract(text)
        return self.upsert_to_graph(result)
