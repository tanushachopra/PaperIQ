import numpy as np
from sentence_transformers import SentenceTransformer, util

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts: list) -> np.ndarray:
    return get_model().encode(texts, convert_to_numpy=True)

def shortlist_by_embedding(query: str, papers_df, top_n: int = 20):
    model      = get_model()
    query_vec  = model.encode(query, convert_to_numpy=True)
    paper_vecs = model.encode(
        papers_df["abstract"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores      = util.cos_sim(query_vec, paper_vecs)[0].numpy()
    top_indices = np.argsort(scores)[::-1][:top_n]
    result      = papers_df.iloc[top_indices].copy()
    result["embedding_score"] = scores[top_indices]
    return result.reset_index(drop=True)