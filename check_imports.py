"""Verify all core dependencies import correctly."""
import importlib.metadata as meta

packages = {
    "langchain":            "langchain",
    "langgraph":            "langgraph",
    "langchain_groq":       "langchain-groq",
    "langchain_huggingface":"langchain-huggingface",
    "langchain_chroma":     "langchain-chroma",
    "chromadb":             "chromadb",
    "ragas":                "ragas",
    "deepeval":             "deepeval",
    "fastapi":              "fastapi",
    "numpy":                "numpy",
    "pandas":               "pandas",
    "neo4j":                "neo4j",
    "sentence_transformers":"sentence-transformers",
    "groq":                 "groq",
}

all_ok = True
for module, dist in packages.items():
    try:
        __import__(module)
        version = meta.version(dist)
        print(f"  {dist:<22} {version}")
    except Exception as e:
        print(f"  {dist:<22} FAILED — {e}")
        all_ok = False

print()
print("All imports OK" if all_ok else "SOME IMPORTS FAILED — see above")
