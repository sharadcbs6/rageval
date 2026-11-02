from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
print("ragas 0.2.x API: OK")
print("metrics:", [faithfulness.name, answer_relevancy.name, context_precision.name])
