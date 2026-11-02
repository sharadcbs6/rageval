"""LangGraph agentic RAG workflow using Groq LLM."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.vectorstore.store import get_vector_store
from src.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2


class RAGState(TypedDict, total=False):
    question: str
    answer: str
    contexts: list[str]
    source_documents: list[dict[str, Any]]
    use_graph: bool
    retry_count: int
    route_decision: Literal["vector", "graph", "hybrid"]


def _get_llm() -> ChatGroq:
    s = get_settings()
    return ChatGroq(model=s.groq_model, temperature=0, api_key=s.groq_api_key)


_ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing agent. Given a question decide the best retrieval strategy:\n"
        '- "vector"  → factual lookup in document chunks\n'
        '- "graph"   → entity/relationship question best served by a knowledge graph\n'
        '- "hybrid"  → complex question needing both\n'
        "Reply with EXACTLY one word: vector, graph, or hybrid.",
    ),
    ("human", "{question}"),
])

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance grader. Given a document and a question, "
        "answer YES if the document is relevant, NO otherwise. One word only.",
    ),
    ("human", "Question: {question}\n\nDocument:\n{document}"),
])

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using ONLY the context below. "
        "If unsure say \"I don't know.\"\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])

_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a hallucination checker. Given an answer and the source context, "
        "does the answer contain claims NOT supported by the context? "
        "Answer YES (hallucination found) or NO (answer is grounded). One word only.",
    ),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
])


def node_route(state: RAGState) -> RAGState:
    chain = _ROUTE_PROMPT | _get_llm() | StrOutputParser()
    raw = chain.invoke({"question": state["question"]}).strip().lower()
    decision: Literal["vector", "graph", "hybrid"] = (
        raw if raw in ("vector", "graph", "hybrid") else "vector"
    )
    logger.info("route_decision", decision=decision)
    return {**state, "route_decision": decision, "retry_count": 0}


def node_retrieve(state: RAGState) -> RAGState:
    docs: list[Document] = get_vector_store().as_retriever(k=5).invoke(state["question"])
    logger.info("vector_retrieved", count=len(docs))
    return {
        **state,
        "contexts": [d.page_content for d in docs],
        "source_documents": [{"content": d.page_content, **d.metadata} for d in docs],
    }


def node_graph_retrieve(state: RAGState) -> RAGState:
    from src.graph.retriever import GraphRetriever
    graph_ctx = GraphRetriever().get_context(state["question"])
    existing = list(state.get("contexts") or [])
    if graph_ctx:
        existing.append(graph_ctx)
    logger.info("graph_retrieved", chars=len(graph_ctx))
    return {**state, "contexts": existing}


def node_grade_docs(state: RAGState) -> RAGState:
    chain = _GRADE_PROMPT | _get_llm() | StrOutputParser()
    question = state["question"]
    filtered = [
        ctx for ctx in state.get("contexts", [])
        if chain.invoke({"question": question, "document": ctx}).strip().upper().startswith("YES")
    ]
    logger.info("grading_done", before=len(state.get("contexts", [])), after=len(filtered))
    return {**state, "contexts": filtered or state.get("contexts", [])}


def node_generate(state: RAGState) -> RAGState:
    context_str = "\n\n---\n\n".join(state.get("contexts", []))
    answer = (_GENERATE_PROMPT | _get_llm() | StrOutputParser()).invoke(
        {"question": state["question"], "context": context_str}
    )
    logger.info("answer_generated", chars=len(answer))
    return {**state, "answer": answer}


def node_check_answer(state: RAGState) -> RAGState:
    context_str = "\n\n---\n\n".join(state.get("contexts", []))
    verdict = (_HALLUCINATION_PROMPT | _get_llm() | StrOutputParser()).invoke(
        {"context": context_str, "answer": state.get("answer", "")}
    ).strip().upper()
    hallucinated = verdict.startswith("YES")
    retry_count = state.get("retry_count", 0)
    if hallucinated:
        logger.warning("hallucination_detected", retry=retry_count)
    return {**state, "retry_count": retry_count + (1 if hallucinated else 0)}


def edge_route_decision(state: RAGState) -> str:
    decision = state.get("route_decision", "vector")
    return "graph_retrieve" if decision == "graph" else "retrieve"


def edge_after_retrieve(state: RAGState) -> str:
    return "graph_retrieve" if state.get("route_decision") == "hybrid" else "grade_docs"


def edge_check_answer(state: RAGState) -> str:
    retry = state.get("retry_count", 0)
    return "generate" if 0 < retry <= MAX_RETRIES else END


def build_rag_graph() -> StateGraph:
    g = StateGraph(RAGState)
    g.add_node("route", node_route)
    g.add_node("retrieve", node_retrieve)
    g.add_node("graph_retrieve", node_graph_retrieve)
    g.add_node("grade_docs", node_grade_docs)
    g.add_node("generate", node_generate)
    g.add_node("check_answer", node_check_answer)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", edge_route_decision, {"retrieve": "retrieve", "graph_retrieve": "graph_retrieve"})
    g.add_conditional_edges("retrieve", edge_after_retrieve, {"graph_retrieve": "graph_retrieve", "grade_docs": "grade_docs"})
    g.add_edge("graph_retrieve", "grade_docs")
    g.add_edge("grade_docs", "generate")
    g.add_edge("generate", "check_answer")
    g.add_conditional_edges("check_answer", edge_check_answer, {"generate": "generate", END: END})
    return g


def compile_workflow():
    return build_rag_graph().compile()


def run_workflow(question: str, use_graph: bool = False) -> dict[str, Any]:
    initial: RAGState = {
        "question": question, "use_graph": use_graph,
        "contexts": [], "source_documents": [],
        "answer": "", "retry_count": 0, "route_decision": "vector",
    }
    final: RAGState = compile_workflow().invoke(initial)
    return {
        "question": final.get("question", question),
        "answer": final.get("answer", ""),
        "contexts": final.get("contexts", []),
        "source_documents": final.get("source_documents", []),
        "route_decision": final.get("route_decision", "vector"),
    }
