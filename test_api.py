"""
End-to-end API test script.

Usage:
    python test_api.py                 # run all tests
    python test_api.py health          # only health check
    python test_api.py ingest          # only ingestion
    python test_api.py query           # only query
    python test_api.py evaluate        # only evaluation
    python test_api.py graph           # only graph endpoints
"""

import sys
import json
import urllib.request
import urllib.error
from typing import Any

BASE_URL = "http://localhost:8080"

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
HEAD = "\033[94m{}\033[0m"


def request(method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
    url = BASE_URL + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def check(label: str, status: int, body: Any, expect_status: int = 200, check_keys: list[str] | None = None):
    ok = status == expect_status
    if ok and check_keys:
        ok = all(k in body for k in check_keys)
    icon = PASS if ok else FAIL
    print(f"{icon}  [{status}] {label}")
    if not ok:
        print(f"        Response: {json.dumps(body, indent=2)[:400]}")
    else:
        # Print a short summary of the response
        summary = {k: v for k, v in body.items() if k not in ("results", "source_documents")} if isinstance(body, dict) else body
        print(f"        {json.dumps(summary)[:200]}")
    return ok


# ── test groups ───────────────────────────────────────────────────────────────

def test_health():
    print(HEAD.format("\n── 1. Health Check ────────────────────────────────────"))
    status, body = request("GET", "/health")
    check("GET /health", status, body, check_keys=["status", "neo4j", "chroma"])


def test_ingest():
    print(HEAD.format("\n── 2. Document Ingestion ───────────────────────────────"))

    # Ingest without graph building (faster)
    payload = {
        "texts": [
            "LangChain is a framework for building applications with large language models. "
            "It provides tools for chaining together LLM calls, managing prompts, and integrating with external data sources.",

            "Ragas is an evaluation framework for Retrieval-Augmented Generation (RAG) pipelines. "
            "It measures faithfulness, answer relevancy, and context precision using LLM-based metrics.",

            "Neo4j is a graph database that stores data as nodes and relationships. "
            "It uses the Cypher query language and is well-suited for knowledge graphs and connected data.",

            "DeepEval is an open-source LLM evaluation framework that provides metrics like hallucination detection, "
            "answer relevancy, and contextual precision for evaluating LLM outputs.",

            "ChromaDB is an open-source vector database designed for storing and querying embeddings. "
            "It supports similarity search and is commonly used in RAG applications.",
        ],
        "metadatas": [
            {"source": "langchain_docs", "topic": "frameworks"},
            {"source": "ragas_docs",     "topic": "evaluation"},
            {"source": "neo4j_docs",     "topic": "databases"},
            {"source": "deepeval_docs",  "topic": "evaluation"},
            {"source": "chromadb_docs",  "topic": "databases"},
        ],
        "build_graph": False,  # set True to also populate Neo4j
    }
    status, body = request("POST", "/ingest", payload)
    check("POST /ingest (5 docs, no graph)", status, body,
          expect_status=201, check_keys=["ingested", "collection"])

    # Ingest with graph building
    payload_graph = {
        "texts": [
            "LangGraph is built on top of LangChain and allows building stateful, "
            "multi-actor applications with LLMs using a graph-based workflow."
        ],
        "build_graph": True,
    }
    status, body = request("POST", "/ingest", payload_graph)
    check("POST /ingest (1 doc, with graph)", status, body,
          expect_status=201, check_keys=["ingested", "collection"])


def test_query():
    print(HEAD.format("\n── 3. RAG Query ─────────────────────────────────────────"))

    # Simple pipeline query
    payload = {
        "question": "What is Ragas and what metrics does it provide?",
        "top_k": 3,
        "use_graph": False,
    }
    status, body = request("POST", "/query", payload)
    check("POST /query (vector only)", status, body,
          check_keys=["question", "answer", "contexts"])

    # Graph-augmented query
    payload_graph = {
        "question": "How does LangChain relate to LangGraph?",
        "top_k": 3,
        "use_graph": True,
    }
    status, body = request("POST", "/query", payload_graph)
    check("POST /query (graph-augmented)", status, body,
          check_keys=["question", "answer", "contexts"])

    # LangGraph agentic workflow
    payload_wf = {
        "question": "What is ChromaDB used for?",
        "top_k": 3,
        "use_graph": False,
    }
    status, body = request("POST", "/query/workflow", payload_wf)
    check("POST /query/workflow (agentic)", status, body,
          check_keys=["question", "answer", "contexts"])


def test_evaluate():
    print(HEAD.format("\n── 4. Evaluation ────────────────────────────────────────"))

    samples = [
        {
            "question": "What is Ragas?",
            "answer": "Ragas is an evaluation framework for RAG pipelines that measures faithfulness and answer relevancy.",
            "contexts": [
                "Ragas is an evaluation framework for Retrieval-Augmented Generation (RAG) pipelines. "
                "It measures faithfulness, answer relevancy, and context precision using LLM-based metrics."
            ],
            "ground_truth": "Ragas is an open-source evaluation framework for RAG systems.",
        },
        {
            "question": "What database does Neo4j use for queries?",
            "answer": "Neo4j uses the Cypher query language.",
            "contexts": [
                "Neo4j is a graph database that stores data as nodes and relationships. "
                "It uses the Cypher query language and is well-suited for knowledge graphs."
            ],
            "ground_truth": "Neo4j uses Cypher as its query language.",
        },
    ]

    # List available metrics
    status, body = request("GET", "/evaluate/metrics")
    check("GET /evaluate/metrics", status, body,
          check_keys=["ragas", "deepeval"])

    # Ragas evaluation
    ragas_payload = {
        "samples": samples,
        "metrics": ["faithfulness", "answer_relevancy"],
    }
    status, body = request("POST", "/evaluate/ragas", ragas_payload)
    check("POST /evaluate/ragas", status, body,
          check_keys=["framework", "results", "aggregate"])

    # DeepEval evaluation
    deepeval_payload = {
        "samples": samples,
        "metrics": ["answer_relevancy", "faithfulness"],
    }
    status, body = request("POST", "/evaluate/deepeval", deepeval_payload)
    check("POST /evaluate/deepeval", status, body,
          check_keys=["framework", "results", "aggregate"])

    # Compare both frameworks
    compare_payload = {
        "samples": samples,
        "metrics": ["faithfulness", "answer_relevancy"],
        "deepeval_metrics": ["answer_relevancy", "faithfulness"],
    }
    status, body = request("POST", "/evaluate/compare", compare_payload)
    check("POST /evaluate/compare", status, body,
          check_keys=["ragas", "deepeval"])


def test_graph():
    print(HEAD.format("\n── 5. Knowledge Graph ───────────────────────────────────"))

    # Health check
    status, body = request("GET", "/graph/health")
    check("GET /graph/health", status, body, check_keys=["neo4j"])

    # Natural language graph search
    status, body = request("GET", "/graph/search?question=LangChain+LangGraph+relationship")
    check("GET /graph/search?question=...", status, body,
          check_keys=["records", "count"])

    # Raw Cypher query — list all Entity nodes
    cypher_payload = {
        "cypher": "MATCH (e:Entity) RETURN e.name AS name, e.type AS type LIMIT 10",
        "params": {},
    }
    status, body = request("POST", "/graph/query", cypher_payload)
    check("POST /graph/query (list entities)", status, body,
          check_keys=["records", "count"])


# ── runner ────────────────────────────────────────────────────────────────────

GROUPS = {
    "health":   test_health,
    "ingest":   test_ingest,
    "query":    test_query,
    "evaluate": test_evaluate,
    "graph":    test_graph,
}

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    print(f"\n{'='*55}")
    print(f"  RAG Eval Platform — API Tests  ({BASE_URL})")
    print(f"{'='*55}")

    if target == "all":
        for fn in GROUPS.values():
            fn()
    elif target in GROUPS:
        GROUPS[target]()
    else:
        print(f"Unknown group '{target}'. Choose from: {', '.join(GROUPS)} or 'all'")
        sys.exit(1)

    print(f"\n{'='*55}\n")
