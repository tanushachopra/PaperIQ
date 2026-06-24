import numpy as np
from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank(query: str, papers_df, top_n: int = 10):
    reranker  = get_reranker()
    pairs     = [[query, abstract]
                 for abstract in papers_df["abstract"].tolist()]
    scores    = reranker.predict(pairs)
    result    = papers_df.copy()
    result["rerank_score"] = scores
    result    = result.sort_values("rerank_score", ascending=False).head(top_n)
    return result.reset_index(drop=True)