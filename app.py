# app.py  —  v2: polished Research Paper Exploration Dashboard
import os
import gdown

# Create folders if not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# Download files
download_file("18mA5Rw0uhQL5FFvymOz5_-CaC-JqLemc", "data/cleaned_papers.csv")
download_file("1N2_YVEGDScL6CXuQ7xgodDB7KbU2YQjp", "models/vectors.pkl")
download_file("1ZwKSxajZJQDgOynjHaLG3OCmu4XL-L-O", "models/final.pkl")

import streamlit as st
import pandas as pd
import altair as alt

from recommender import load_model_artifacts, get_recommendations, search_by_description
from utils.helpers import inject_global_css, show_paper_card, credit_footer

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PaperIQ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()   # ← apply all CSS


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔬 Loading ML model — please wait...")
def load_artifacts():
    return load_model_artifacts()

vectorizer, tfidf_matrix, metadata = load_artifacts()


# ══════════════════════════════════════════════════════════════════════════════
# ALTAIR THEME HELPER  (dark background charts)
# ══════════════════════════════════════════════════════════════════════════════

CHART_CONFIG = {
    "config": {
        "background":   "#13132a",
        "view":         {"stroke": "transparent"},
        "axis": {
            "domainColor": "#3a3a5c",
            "gridColor":   "#1e1e38",
            "tickColor":   "#3a3a5c",
            "labelColor":  "#9090b0",
            "titleColor":  "#b0b0d0",
            "labelFont":   "Inter",
            "titleFont":   "Inter",
        },
        "legend": {
            "labelColor":  "#b0b0d0",
            "titleColor":  "#c0c0e0",
            "labelFont":   "Inter",
        },
        "title": {"color": "#d0c4ff", "font": "Inter", "fontSize": 15},
    }
}

def themed(chart):
    return chart.configure(**CHART_CONFIG["config"])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-size:2.4rem;margin-bottom:4px'>🔬</div>"
        "<div style='font-size:1.4rem;font-weight:700;color:#c8b8ff'>"
        "Research Explorer</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        options=["🏠 Home", "🤖 Recommender", "📈 Trend Analysis", "🔍 Search"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
    "<div style='font-size:2.8rem;margin-bottom:2px'>🧠</div>"
    "<div style='font-size:2rem;font-weight:800;color:#c8b8ff;"
    "letter-spacing:0.04em;line-height:1.2'>PaperIQ</div>"
    "<div style='font-size:0.75rem;color:#6c63ff;font-weight:500;"
    "letter-spacing:0.12em;text-transform:uppercase;margin-top:4px'>"
    "Research Intelligence</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":

    st.markdown(
    "<div style='font-size:3.8rem;font-weight:900;color:#c8b8ff;"
    "letter-spacing:0.04em;line-height:1.1;margin-bottom:8px'>🧠 PaperIQ</div>"
    "<div style='font-size:1.15rem;color:#8888aa;font-weight:400;"
    "letter-spacing:0.02em;margin-bottom:1.5rem'>"
    "Discover, compare, and analyze 136K+ research papers — powered by TF-IDF & Cosine Similarity."
    "</div>",
    unsafe_allow_html=True,
)
    st.markdown("---")

    # ── Metrics ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Total Papers",   f"{len(metadata):,}")
    c2.metric("🏷️ Categories",     metadata["category"].nunique())
    c3.metric("📅 Earliest Year",  int(metadata["year"].min()))
    c4.metric("📅 Latest Year",    int(metadata["year"].max()))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature cards ──────────────────────────────────────────────────────────
    st.markdown("#### 🧭 What can you do here?")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.info("### 🤖 Recommender\nSelect any paper and instantly find the most similar ones using **TF-IDF + Cosine Similarity**.")
    with f2:
        st.success("### 📈 Trend Analysis\nExplore how research output has **grown or shifted** over the years and across categories.")
    with f3:
        st.warning("### 🔍 Search\nDescribe a topic in **plain English** and let the app surface the most relevant papers.")

    st.markdown("---")

    # ── Dataset Preview ────────────────────────────────────────────────────────
    st.markdown("#### 📋 Dataset Preview")
    st.dataframe(
        metadata[["title", "category", "year"]].head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── FIX: Top 20 categories chart (avoid 138-bar clutter) ──────────────────
    st.markdown("#### 🏷️ Top 20 Categories by Paper Count")

    top20 = (
        metadata["category"]
        .value_counts()
        .head(20)
        .reset_index()
    )
    top20.columns = ["category", "count"]

    bar = (
        alt.Chart(top20)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            y=alt.Y("category:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelLimit=200, labelFontSize=12)),
            x=alt.X("count:Q",
                    title="Number of Papers",
                    axis=alt.Axis(labelFontSize=11)),
            color=alt.Color(
                "count:Q",
                scale=alt.Scale(scheme="purpleblue"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q",    title="Papers"),
            ],
        )
        .properties(height=520, title="Top 20 Research Categories")
    )

    st.altair_chart(
        themed(bar),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Recommender":

    st.markdown(
        "<div class='page-title'>🤖 Paper Recommender</div>"
        "<div class='page-subtitle'>"
        "Select a paper and discover the most similar ones using cosine similarity on TF-IDF vectors."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([0.78, 0.22])
    with col1:
        selected_paper = st.selectbox(
            "📄 Select a Paper",
            options=metadata["title"].tolist(),
            index=0,
        )
    with col2:
        top_n = st.slider("Top N Results", 3, 15, 5)

    # Selected paper info
    selected_row = metadata[metadata["title"] == selected_paper].iloc[0]
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📌 Selected Paper Details", expanded=True):
        sc1, sc2 = st.columns([0.7, 0.3])
        with sc1:
            st.markdown(
                f"<div style='font-size:1.1rem;font-weight:600;"
                f"color:#c8b8ff'>{selected_row['title']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:#8888aa;margin-top:6px'>"
                f"🏷️ {selected_row['category']}  &nbsp;|&nbsp;  "
                f"📅 {int(selected_row['year'])}</div>",
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                f"<div style='text-align:right;color:#6c63ff;"
                f"font-size:0.85rem'>Category<br>"
                f"<b style='font-size:1.1rem;color:#9c8fff'>"
                f"{selected_row['category']}</b></div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<p style='color:#b0b0cc;margin-top:10px;"
            f"font-size:0.92rem;line-height:1.7'>"
            f"{selected_row['summary']}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Computing similarity scores..."):
            recs = get_recommendations(
                selected_paper, metadata, tfidf_matrix, top_n=top_n
            )

        if recs.empty:
            st.error("No recommendations found. Try a different paper.")
        else:
            st.markdown(
                f"<div style='color:#4ecca3;font-weight:600;"
                f"margin-bottom:12px'>✅ Top {len(recs)} papers similar to "
                f"<em>{selected_paper[:60]}…</em></div>",
                unsafe_allow_html=True,
            )
            for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                show_paper_card(row, rank=rank)
    else:
        st.markdown(
            "<div style='color:#666688;font-style:italic'>"
            "👆 Press the button above to generate recommendations.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Trend Analysis":

    st.markdown(
        "<div class='page-title'>📈 Publication Trend Analysis</div>"
        "<div class='page-subtitle'>"
        "How has research evolved over the years? Explore output by year and category."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    all_cats = sorted(metadata["category"].dropna().unique().tolist())
    selected_cats = st.multiselect(
        "🏷️ Filter by Category (leave empty = show all)",
        options=all_cats,
        default=[],
    )

    df_trend = metadata.copy()
    if selected_cats:
        df_trend = df_trend[df_trend["category"].isin(selected_cats)]
    df_trend["year"] = df_trend["year"].dropna().astype(int)

    st.markdown("---")

    # ── Chart 1: Papers per year (line) ───────────────────────────────────────
    st.markdown("#### 📊 Total Papers Published per Year")

    yearly = df_trend.groupby("year").size().reset_index(name="count")

    area = (
        alt.Chart(yearly)
        .mark_area(
            line={"color": "#6c63ff", "strokeWidth": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#6c63ff", offset=0),
                    alt.GradientStop(color="rgba(108,99,255,0.05)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        .encode(
            x=alt.X("year:O", title="Year",
                    axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("count:Q", title="Number of Papers"),
            tooltip=[
                alt.Tooltip("year:O",  title="Year"),
                alt.Tooltip("count:Q", title="Papers"),
            ],
        )
        .properties(height=320, title="Research Output Over Time")
    )

    st.altair_chart(themed(area), use_container_width=True)

    peak_year  = int(yearly.loc[yearly["count"].idxmax(), "year"])
    peak_count = int(yearly["count"].max())
    st.info(f"📌 **Peak year:** **{peak_year}** with **{peak_count:,}** papers published.")

    st.markdown("---")

    # ── Chart 2: Category breakdown (top 10 only to avoid clutter) ────────────
    st.markdown("#### 🏷️ Top 10 Categories Over Time")

    top10_cats = (
        df_trend["category"].value_counts().head(10).index.tolist()
    )
    df_top = df_trend[df_trend["category"].isin(top10_cats)]
    cat_yearly = (
        df_top.groupby(["year", "category"])
        .size()
        .reset_index(name="count")
    )

    stacked = (
        alt.Chart(cat_yearly)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year",
                    axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("count:Q", title="Papers", stack="zero"),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(scheme="tableau10"),
                title="Category",
            ),
            tooltip=[
                alt.Tooltip("year:O",     title="Year"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q",    title="Papers"),
            ],
        )
        .properties(height=380, title="Category Breakdown per Year (Top 10)")
    )

    st.altair_chart(themed(stacked), use_container_width=True)

    st.markdown("---")

    # ── Chart 3: Top 15 categories horizontal bar ──────────────────────────────
    st.markdown("#### 🏆 Top 15 Categories — All Time")

    top15 = (
        df_trend["category"]
        .value_counts()
        .head(15)
        .reset_index()
    )
    top15.columns = ["category", "count"]

    horiz = (
        alt.Chart(top15)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            y=alt.Y("category:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=220, labelFontSize=12)),
            x=alt.X("count:Q", title="Papers"),
            color=alt.Color(
                "count:Q",
                scale=alt.Scale(scheme="purples"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q",    title="Papers"),
            ],
        )
        .properties(height=420, title="Top 15 Categories by Total Papers")
    )

    st.altair_chart(themed(horiz), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SEARCH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Search":

    st.markdown(
        "<div class='page-title'>🔍 Description-Based Search</div>"
        "<div class='page-subtitle'>"
        "Describe any research topic in plain English — we'll find the most relevant papers."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    query = st.text_area(
        "📝 Describe what you're looking for",
        placeholder="e.g.  deep learning methods for medical image segmentation using CNNs",
        height=110,
    )

    col1, col2 = st.columns([0.75, 0.25])
    with col2:
        top_n = st.slider("Top N Results", 3, 15, 5, key="search_slider")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔎 Search Papers", type="primary"):
        if not query.strip():
            st.warning("⚠️ Please enter a description before searching.")
        else:
            with st.spinner("Scanning 136K+ papers..."):
                results = search_by_description(
                    query, vectorizer, tfidf_matrix, metadata, top_n=top_n
                )

            if results.empty:
                st.error("No results found. Try rephrasing your description.")
            else:
                st.markdown(
                    f"<div style='color:#4ecca3;font-weight:600;"
                    f"margin-bottom:12px'>✅ Top {len(results)} papers matching "
                    f"your description</div>",
                    unsafe_allow_html=True,
                )
                for rank, (_, row) in enumerate(results.iterrows(), start=1):
                    show_paper_card(row, rank=rank)
    else:
        st.markdown(
            "<div style='color:#666688;font-style:italic'>"
            "👆 Describe a topic above and press Search.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CREDIT FOOTER
# ══════════════════════════════════════════════════════════════════════════════

credit_footer()