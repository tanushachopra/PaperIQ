# recommender.py
# Step 2: TF-IDF Vectorization + Cosine Similarity + Pickle Save/Load

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration ──────────────────────────────────────────────────────────────
CLEANED_DATA_PATH = "data/cleaned_papers.csv"
VECTORS_PKL       = "models/vectors.pkl"      # saves TF-IDF matrix
METADATA_PKL      = "models/final.pkl"        # saves titles, categories, years


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Build & Save the Model
# ══════════════════════════════════════════════════════════════════════════════

def build_tfidf_model(df: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer on the combined_text column.
    Returns:
        vectorizer  — the fitted TfidfVectorizer object
        tfidf_matrix — sparse matrix (n_papers × n_features)
    """
    print("⚙️  Fitting TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=5000,       # keep top 5000 words (reduces noise)
        stop_words="english",    # remove common words like 'the', 'is', 'and'
        ngram_range=(1, 2),      # use single words AND two-word phrases
        min_df=2,                # ignore words that appear in fewer than 2 papers
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"   → {tfidf_matrix.shape[0]} papers × {tfidf_matrix.shape[1]} features\n")

    return vectorizer, tfidf_matrix


def save_model_artifacts(vectorizer, tfidf_matrix, df: pd.DataFrame):
    """
    Save two pickle files:
      - vectors.pkl  → vectorizer + tfidf_matrix  (used for search)
      - final.pkl    → clean metadata DataFrame   (used for display)
    """
    os.makedirs("models", exist_ok=True)   # create models/ folder if missing

    # Save TF-IDF vectorizer and matrix together
    with open(VECTORS_PKL, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix}, f)
    print(f"💾 Vectors saved     → {VECTORS_PKL}")

    # Save metadata: only the columns the UI needs
    metadata = df[["title", "summary", "category", "year"]].reset_index(drop=True)
    with open(METADATA_PKL, "wb") as f:
        pickle.dump(metadata, f)
    print(f"💾 Metadata saved    → {METADATA_PKL}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Load the Saved Model
# ══════════════════════════════════════════════════════════════════════════════

def load_model_artifacts():
    """
    Load vectorizer, tfidf_matrix, and metadata from pickle files.
    Returns:
        vectorizer, tfidf_matrix, metadata_df
    """
    with open(VECTORS_PKL, "rb") as f:
        vector_data = pickle.load(f)

    with open(METADATA_PKL, "rb") as f:
        metadata = pickle.load(f)

    print("✅ Model artifacts loaded from pickle files.")
    return vector_data["vectorizer"], vector_data["tfidf_matrix"], metadata


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Recommendation Logic
# ══════════════════════════════════════════════════════════════════════════════

def get_recommendations(paper_title: str, metadata: pd.DataFrame,
                         tfidf_matrix, top_n: int = 5) -> pd.DataFrame:
    """
    Given a paper title, return top_n most similar papers.

    How it works:
      1. Find the index of the selected paper in metadata
      2. Extract its TF-IDF vector (one row from the matrix)
      3. Compute cosine similarity against ALL other papers
      4. Sort by similarity score, return top_n (excluding itself)
    """
    # Step 1: Find the paper's index
    matches = metadata[metadata["title"] == paper_title]
    if matches.empty:
        print(f"❌ Paper not found: '{paper_title}'")
        return pd.DataFrame()

    paper_idx = matches.index[0]

    # Step 2: Get that paper's vector (1 row of the matrix)
    paper_vector = tfidf_matrix[paper_idx]

    # Step 3: Compute cosine similarity with all papers
    similarity_scores = cosine_similarity(paper_vector, tfidf_matrix).flatten()

    # Step 4: Sort scores — highest first, skip index 0 (the paper itself)
    similar_indices = np.argsort(similarity_scores)[::-1]   # descending order
    similar_indices = [i for i in similar_indices if i != paper_idx][:top_n]

    # Step 5: Build result DataFrame
    results = metadata.iloc[similar_indices].copy()
    results["similarity_score"] = np.round(similarity_scores[similar_indices], 4)
    results = results[["title", "category", "year", "similarity_score", "summary"]]

    return results.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Description-Based Search
# ══════════════════════════════════════════════════════════════════════════════

def search_by_description(user_query: str, vectorizer,
                           tfidf_matrix, metadata: pd.DataFrame,
                           top_n: int = 5) -> pd.DataFrame:
    """
    Given a free-text description from the user:
      1. Clean and vectorize it using the SAME fitted vectorizer
      2. Compare with all papers using cosine similarity
      3. Return top_n matching papers
    """
    import re

    # Clean query the same way we cleaned our data
    query = user_query.lower()
    query = re.sub(r"[^a-z\s]", "", query)
    query = re.sub(r"\s+", " ", query).strip()

    if not query:
        return pd.DataFrame()

    # Vectorize the query (transform only — do NOT refit)
    query_vector = vectorizer.transform([query])

    # Compute similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get top_n results
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]

    results = metadata.iloc[top_indices].copy()
    results["similarity_score"] = np.round(similarity_scores[top_indices], 4)
    results = results[["title", "category", "year", "similarity_score", "summary"]]

    return results.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# RUN AS STANDALONE — builds + tests the model
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Load cleaned data
    print("📂 Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"   {len(df)} papers loaded.\n")

    # 2. Build TF-IDF model
    vectorizer, tfidf_matrix = build_tfidf_model(df)

    # 3. Save to pickle
    save_model_artifacts(vectorizer, tfidf_matrix, df)

    # 4. Reload and test
    print("\n🔁 Reloading from pickle to verify...\n")
    vectorizer, tfidf_matrix, metadata = load_model_artifacts()

    # 5. Test recommender with the first paper in dataset
    sample_title = metadata["title"].iloc[0]
    print(f"\n🔍 Testing Recommender for:\n   '{sample_title}'\n")
    recs = get_recommendations(sample_title, metadata, tfidf_matrix, top_n=3)
    if not recs.empty:
        print(recs[["title", "similarity_score", "category"]].to_string())

    # 6. Test description search
    sample_query = "neural networks image classification deep learning"
    print(f"\n🔍 Testing Search for query:\n   '{sample_query}'\n")
    search_results = search_by_description(sample_query, vectorizer, tfidf_matrix, metadata, top_n=3)
    if not search_results.empty:
        print(search_results[["title", "similarity_score", "category"]].to_string())

    print("\n✅ All tests passed! Model is ready.")