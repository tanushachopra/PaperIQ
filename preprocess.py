# preprocess.py
# Step 1: Load and clean the dataset

import pandas as pd
import re
import os

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = "data/papers.csv"          # 👈 Change this to your CSV filename


# ── Helper Functions ───────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset into a Pandas DataFrame.
    Raises a clear error if the file is not found.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please place your CSV file inside the 'data/' folder."
        )
    
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns found: {list(df.columns)}\n")
    return df


def clean_text(text: str) -> str:
    """
    Clean a single string:
      - Convert to lowercase
      - Remove URLs
      - Remove special characters and digits
      - Remove extra whitespace
    """
    if not isinstance(text, str):      # handle missing / NaN values
        return ""
    text = text.lower()                          # UPPERCASE → lowercase
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)         # keep only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()     # collapse multiple spaces
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Rename columns to standard names (edit mapping below if needed)
      2. Drop rows missing title or summary
      3. Clean title and summary text
      4. Combine them into a single 'combined_text' field
      5. Parse publication_date and extract 'year'
    """

    # ── 1. Standardise column names ───────────────────────────────────────────
    # Edit this mapping if your CSV uses different column names
    column_mapping = {
        "title":            "title",
        "summary":          "summary",
        "category":         "category",
        "publication_date": "publication_date",
    }
    df = df.rename(columns=column_mapping)

    # Keep only the columns we need (ignore extras)
    required_cols = ["title", "summary", "category", "publication_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in dataset: {missing}\n"
            "Please check your CSV column names and update 'column_mapping' above."
        )
    df = df[required_cols].copy()

    # ── 2. Drop rows with missing title or summary ────────────────────────────
    before = len(df)
    df.dropna(subset=["title", "summary"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"⚠️  Dropped {dropped} rows with missing title/summary.")

    # ── 3. Clean text columns ─────────────────────────────────────────────────
    df["clean_title"]   = df["title"].apply(clean_text)
    df["clean_summary"] = df["summary"].apply(clean_text)
    print("✅ Text cleaning complete.")

    # ── 4. Combine title + summary → combined_text ────────────────────────────
    # We give the title a bit more weight by repeating it twice
    df["combined_text"] = df["clean_title"] + " " + df["clean_title"] + " " + df["clean_summary"]
    print("✅ Combined text field created.")

    # ── 5. Extract year from publication_date ─────────────────────────────────
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["year"] = df["publication_date"].dt.year
    print("✅ Year extracted from publication date.")

    # Final check
    print(f"\n📊 Preprocessed dataset: {df.shape[0]} rows ready.")
    print(df[["title", "combined_text", "year", "category"]].head(3).to_string())

    return df


# ── Run as standalone script ───────────────────────────────────────────────────
if __name__ == "__main__":
    raw_df      = load_data(DATA_PATH)
    clean_df    = preprocess(raw_df)

    # Save cleaned data so other modules can reuse it
    clean_df.to_csv("data/cleaned_papers.csv", index=False)
    print("\n💾 Cleaned data saved to → data/cleaned_papers.csv")