# 🧠 PaperIQ — Research Paper Exploration Dashboard
**Made by Tanusha Chopra**

---

## 📌 What is PaperIQ?

PaperIQ is a machine learning powered web application that lets you
explore, search, and analyze 136,000+ research papers in an interactive
and intelligent way. Instead of manually scanning through thousands of
papers, PaperIQ uses TF-IDF vectorization and Cosine Similarity to
understand the content of each paper and surface the most relevant ones
— instantly.

---

## 🎯 Why I Built This

Research papers are hard to navigate. There are thousands of them,
spread across dozens of categories, published over 30+ years. I wanted
to build something that makes exploring academic research feel fast,
visual, and intelligent — not like digging through a spreadsheet.

---

## ✨ Features

### 🤖 Smart Recommender
Select any paper from the dataset and PaperIQ instantly finds the most
similar papers using cosine similarity on TF-IDF vectors. Each result
shows a similarity score, category, year, and a summary — so you can
quickly judge relevance.

### 🔍 Description-Based Search
Type a topic or idea in plain English — PaperIQ converts it into a
TF-IDF vector and compares it against the entire dataset to return the
most relevant papers. No keywords needed, just describe what you're
looking for.

### 📈 Publication Trend Analysis
Explore how research output has evolved from 1993 to 2025. See which
categories are growing, which peaked, and how the landscape has shifted
— all through interactive Altair charts.

---

## 🧠 How It Works (The ML Behind It)

1. **Text Preprocessing** — Paper titles and summaries are cleaned
   (lowercased, noise removed) and combined into a single text field.
   The title is weighted more by repeating it, since titles carry more
   signal.

2. **TF-IDF Vectorization** — Each paper is converted into a numerical
   vector using TfidfVectorizer (top 5000 features, unigrams + bigrams,
   English stop words removed). This captures how important each word
   is relative to the whole dataset.

3. **Cosine Similarity** — To find similar papers, we compute the cosine
   of the angle between two paper vectors. A score close to 1 means very
   similar, close to 0 means unrelated.

4. **Pickle Serialization** — The fitted vectorizer and TF-IDF matrix
   are saved as .pkl files so the app loads instantly without
   recomputing on every run.

---

## 🗂️ Project Structure
paperIQ/
├── .streamlit/
│   └── config.toml         ← dark theme configuration
├── data/
│   ├── papers.csv           ← raw dataset (136,238 papers)
│   └── cleaned_papers.csv   ← preprocessed data
├── models/
│   ├── vectors.pkl          ← TF-IDF matrix + vectorizer
│   └── final.pkl            ← paper metadata
├── utils/
│   └── helpers.py           ← UI components + global CSS
├── app.py                   ← main Streamlit application
├── preprocess.py            ← data cleaning pipeline
├── recommender.py           ← ML model logic
├── requirements.txt         ← dependencies
└── README.md
---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| UI Framework | Streamlit |
| ML / NLP | scikit-learn (TF-IDF, Cosine Similarity) |
| Data Processing | Pandas, NumPy |
| Visualization | Altair |
| Model Saving | Pickle |
| Styling | Custom CSS via st.markdown |

---

## 📊 Dataset

- **136,238** research papers
- **138** unique categories
- **32 years** of publications (1993 – 2025)
- Fields covered: Machine Learning, Physics, Biology,
  Software Engineering, Astrophysics, and more

---

## 🧩 What I Learned Building This

- How TF-IDF works in practice on a large real-world dataset
- Why cosine similarity is preferred over Euclidean distance for
  high-dimensional text vectors
- How to architect a clean ML project with separation between
  data, model, and UI layers
- How Streamlit session_state works to persist results across reruns
- How to build a polished dark-themed UI purely with custom CSS
  injected through Streamlit
  ## ⚠️ Note
The dataset and model files are not included in this repo
due to GitHub's file size limits. Run these locally to
regenerate them:
```bash
python preprocess.py
python recommender.py
```

---

*PaperIQ — Built with 🧠 by Tanusha Chopra*