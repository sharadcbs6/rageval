import json
import urllib.request

BASE = "http://localhost:8080"
payload = {
    "samples": [
        {
            "question": "What is Ragas?",
            "answer": "Ragas is an evaluation framework for RAG pipelines that measures faithfulness and answer relevancy.",
            "contexts": ["Ragas is an evaluation framework for Retrieval-Augmented Generation (RAG) pipelines. It measures faithfulness, answer relevancy, and context precision using LLM-based metrics."],
            "ground_truth": "Ragas is an open-source evaluation framework for RAG systems.",
        },
        {
            "question": "What database does Neo4j use for queries?",
            "answer": "Neo4j uses the Cypher query language.",
            "contexts": ["Neo4j is a graph database that stores data as nodes and relationships. It uses the Cypher query language and is well-suited for knowledge graphs."],
            "ground_truth": "Neo4j uses Cypher as its query language.",
        },
    ],
    "metrics": ["faithfulness", "answer_relevancy"],
    "deepeval_metrics": ["answer_relevancy", "faithfulness"],
}

data = json.dumps(payload).encode()
req = urllib.request.Request(
    BASE + "/evaluate/compare",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
print("Sending compare request (this takes ~60s)...")
with urllib.request.urlopen(req, timeout=300) as resp:
    result = json.loads(resp.read())
    ragas_agg = result.get("ragas", {}).get("aggregate", {})
    deepeval_agg = result.get("deepeval", {}).get("aggregate", {})
    print()
    print("RAGAS:    ", ragas_agg)
    print("DEEPEVAL: ", deepeval_agg)
    print()
    print("COMPARE: PASS")
