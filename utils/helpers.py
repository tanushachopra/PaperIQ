# utils/helpers.py  —  v2: polished UI helpers

import streamlit as st
import pandas as pd


def inject_global_css():
    """
    Inject global CSS for a polished, consistent look.
    Call this ONCE at the top of app.py, right after set_page_config.
    """
    st.markdown(
        """
        <style>
        /* ── Import font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ── Global ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
            border-right: 1px solid #2a2a3e;
        }
        section[data-testid="stSidebar"] * {
            color: #e0e0f0 !important;
        }

        /* ── Main background ── */
        .stApp {
            background: #0d0d1a;
            color: #e0e0f0;
        }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1e1e35 0%, #252540 100%);
            border: 1px solid #3a3a5c;
            border-radius: 12px;
            padding: 18px 22px !important;
            transition: transform 0.2s;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #6c63ff;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            color: #9090b0 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #c8b8ff !important;
        }

        /* ── Feature info/success/warning boxes ── */
        [data-testid="stAlert"] {
            border-radius: 12px !important;
            border: none !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.2s !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6c63ff, #9c5fff) !important;
            border: none !important;
            color: white !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(108,99,255,0.4) !important;
        }

        /* ── Text input / text area ── */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: #1e1e35 !important;
            border: 1px solid #3a3a5c !important;
            border-radius: 8px !important;
            color: #e0e0f0 !important;
        }

        /* ── Selectbox ── */
        .stSelectbox > div > div {
            background: #1e1e35 !important;
            border: 1px solid #3a3a5c !important;
            border-radius: 8px !important;
            color: #e0e0f0 !important;
        }

        /* ── Divider ── */
        hr {
            border-color: #2a2a3e !important;
        }

        /* ── Expander ── */
        details {
            background: #1a1a2e !important;
            border: 1px solid #2a2a3e !important;
            border-radius: 10px !important;
        }

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {
            border-radius: 10px !important;
            overflow: hidden;
        }

        /* ── Page title style ── */
        .page-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #c8b8ff;
            margin-bottom: 0.2rem;
        }
        .page-subtitle {
            font-size: 1rem;
            color: #8888aa;
            margin-bottom: 1.5rem;
        }

        /* ── Paper card ── */
        .paper-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #1e1e38 100%);
            border: 1px solid #2a2a4a;
            border-left: 4px solid #6c63ff;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 14px;
            transition: all 0.2s;
        }
        .paper-card:hover {
            border-left-color: #9c5fff;
            transform: translateX(3px);
            box-shadow: 0 4px 20px rgba(108,99,255,0.15);
        }
        .paper-card-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #d0c4ff;
            margin-bottom: 6px;
        }
        .paper-card-meta {
            font-size: 0.82rem;
            color: #7878a0;
            margin-bottom: 8px;
        }
        .score-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .score-high   { background: #1a3a2a; color: #4ecca3; border: 1px solid #4ecca3; }
        .score-medium { background: #3a2e1a; color: #f0a500; border: 1px solid #f0a500; }
        .score-low    { background: #3a1a1a; color: #ff6b6b; border: 1px solid #ff6b6b; }

        /* ── Credit footer ── */
        .credit-footer {
            position: fixed;
            bottom: 12px;
            left: 16px;
            font-size: 0.72rem;
            color: #555577;
            z-index: 9999;
            pointer-events: none;
            letter-spacing: 0.05em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_paper_card(row: pd.Series, rank: int = None):
    """Render a paper as a styled HTML card."""
    score_html = ""
    if "similarity_score" in row:
        score = float(row["similarity_score"])
        css_class = "score-high" if score > 0.5 else "score-medium" if score > 0.25 else "score-low"
        score_html = f'<span class="score-badge {css_class}">Score: {score:.4f}</span>'

    rank_str   = f"<span style='color:#6c63ff;font-weight:700'>#{rank}</span> " if rank else ""
    category   = row.get("category", "")
    year       = int(row["year"]) if pd.notna(row.get("year")) else ""
    title      = row.get("title", "Untitled")
    summary    = row.get("summary", "")

    card_html = f"""
    <div class="paper-card">
        <div class="paper-card-title">{rank_str}{title}</div>
        <div class="paper-card-meta">
            🏷️ {category} &nbsp;&nbsp; 📅 {year} &nbsp;&nbsp; {score_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    if summary:
        with st.expander("📄 View Summary"):
            st.markdown(
                f"<p style='color:#b0b0cc;font-size:0.92rem;line-height:1.7'>{summary}</p>",
                unsafe_allow_html=True,
            )


def credit_footer():
    st.markdown(
        '<div class="credit-footer">✦ Made by Tanusha Chopra</div>',
        unsafe_allow_html=True,
    )