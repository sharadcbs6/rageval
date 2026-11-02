"""Smoke-test that all src modules import without error."""
import sys
sys.path.insert(0, ".")

tests = [
    ("src.config",              "get_settings"),
    ("src.models.schemas",      "QueryRequest"),
    ("src.graph.client",        "get_neo4j_client"),
    ("src.graph.extractor",     "KnowledgeGraphExtractor"),
    ("src.graph.retriever",     "GraphRetriever"),
    ("src.vectorstore.store",   "get_vector_store"),
    ("src.vectorstore.ingestor","DocumentIngestor"),
    ("src.rag.pipeline",        "RAGPipeline"),
    ("src.rag.workflow",        "run_workflow"),
    ("src.evaluation.ragas_eval",   "RagasEvaluator"),
    ("src.evaluation.deepeval_eval","DeepEvaluator"),
    ("src.api.routers.ingest",   "router"),
    ("src.api.routers.query",    "router"),
    ("src.api.routers.evaluate", "router"),
    ("src.api.routers.graph",    "router"),
    ("src.api.app",              "app"),
]

all_ok = True
for module, symbol in tests:
    try:
        mod = __import__(module, fromlist=[symbol])
        getattr(mod, symbol)
        print(f"  OK  {module}")
    except Exception as e:
        print(f"  ERR {module}: {e}")
        all_ok = False

print()
print("All src modules OK" if all_ok else "SOME MODULES FAILED")
