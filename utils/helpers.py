# utils/helpers.py  —  v2: polished UI helpers

# utils/helpers.py — v3: Premium AI Research Platform UI

import streamlit as st
import pandas as pd


def inject_global_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background: #080810;
        color: #e0e0f0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0c1a 0%, #10101e 100%);
        border-right: 1px solid rgba(108,99,255,0.15);
    }
    section[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg,
            rgba(108,99,255,0.08) 0%,
            rgba(30,30,60,0.6) 100%);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 16px;
        padding: 20px 24px !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(108,99,255,0.5);
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(108,99,255,0.2);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        color: #6c63ff !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.8rem !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6c63ff 0%, #9c5fff 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(108,99,255,0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(108,99,255,0.5) !important;
    }

    /* ── Text inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(108,99,255,0.25) !important;
        border-radius: 12px !important;
        color: #e0e0f0 !important;
        font-size: 1rem !important;
        transition: all 0.2s !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(108,99,255,0.6) !important;
        box-shadow: 0 0 0 3px rgba(108,99,255,0.1) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(108,99,255,0.25) !important;
        border-radius: 10px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid rgba(108,99,255,0.15) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #9090b0 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 8px 18px !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,
            rgba(108,99,255,0.3), rgba(156,95,255,0.3)) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ── Expander ── */
    details {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(108,99,255,0.15) !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }
    details summary {
        padding: 14px 18px !important;
        font-weight: 500 !important;
        color: #c8b8ff !important;
        cursor: pointer !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(108,99,255,0.15) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #080810; }
    ::-webkit-scrollbar-thumb {
        background: rgba(108,99,255,0.3);
        border-radius: 3px;
    }

    /* ── Custom components ── */
    .page-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #c8b8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .page-subtitle {
        font-size: 1rem;
        color: #6060a0;
        margin-bottom: 1.5rem;
        font-weight: 400;
        line-height: 1.6;
    }

    /* ── Glass card ── */
    .glass-card {
        background: linear-gradient(135deg,
            rgba(255,255,255,0.04) 0%,
            rgba(108,99,255,0.05) 100%);
        border: 1px solid rgba(108,99,255,0.18);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 14px;
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }
    .glass-card:hover {
        border-color: rgba(108,99,255,0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108,99,255,0.12);
    }

    /* ── Paper card ── */
    .paper-card {
        background: linear-gradient(135deg,
            rgba(255,255,255,0.03) 0%,
            rgba(20,20,45,0.8) 100%);
        border: 1px solid rgba(108,99,255,0.15);
        border-left: 3px solid #6c63ff;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 12px;
        transition: all 0.25s ease;
    }
    .paper-card:hover {
        border-left-color: #9c5fff;
        transform: translateX(4px);
        box-shadow: 0 4px 24px rgba(108,99,255,0.15);
        background: linear-gradient(135deg,
            rgba(108,99,255,0.06) 0%,
            rgba(20,20,45,0.9) 100%);
    }
    .paper-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e8e0ff;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .paper-meta {
        font-size: 0.8rem;
        color: #5050a0;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }

    /* ── Score badge ── */
    .score-badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .score-high   {
        background: rgba(78,204,163,0.12);
        color: #4ecca3;
        border: 1px solid rgba(78,204,163,0.3);
    }
    .score-medium {
        background: rgba(240,165,0,0.12);
        color: #f0a500;
        border: 1px solid rgba(240,165,0,0.3);
    }
    .score-low    {
        background: rgba(255,107,107,0.12);
        color: #ff6b6b;
        border: 1px solid rgba(255,107,107,0.3);
    }

    /* ── Insight card ── */
    .insight-field {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(108,99,255,0.12);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: all 0.2s;
    }
    .insight-field:hover {
        border-color: rgba(108,99,255,0.25);
        background: rgba(108,99,255,0.05);
    }
    .insight-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }
    .insight-value {
        font-size: 0.9rem;
        color: #b0b0d0;
        line-height: 1.6;
    }

    /* ── Cluster card ── */
    .cluster-card {
        background: linear-gradient(135deg,
            rgba(108,99,255,0.08) 0%,
            rgba(20,20,40,0.6) 100%);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
    }
    .cluster-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        background: linear-gradient(90deg, #6c63ff, #9c5fff);
    }

    /* ── Gap analysis sections ── */
    .gap-section {
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 14px;
        border-left: 4px solid;
    }
    .gap-unexplored {
        background: rgba(78,204,163,0.06);
        border-color: #4ecca3;
    }
    .gap-contradiction {
        background: rgba(255,107,107,0.06);
        border-color: #ff6b6b;
    }
    .gap-opportunity {
        background: rgba(108,99,255,0.06);
        border-color: #6c63ff;
    }
    .gap-recommendation {
        background: rgba(240,165,0,0.06);
        border-color: #f0a500;
    }
    .gap-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
    }
    .gap-content {
        font-size: 0.92rem;
        color: #b0b0cc;
        line-height: 1.75;
    }

    /* ── Hero section ── */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(108,99,255,0.12);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #9c8fff;
        letter-spacing: 0.05em;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg,
            #ffffff 0%, #c8b8ff 50%, #9c5fff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 12px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #5858a0;
        line-height: 1.6;
        max-width: 580px;
        margin-bottom: 32px;
        font-weight: 400;
    }

    /* ── Status messages ── */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.88rem;
        color: #9090c0;
        border: 1px solid rgba(108,99,255,0.1);
    }

    /* ── Credit footer ── */
    .credit-footer {
        position: fixed;
        bottom: 12px;
        left: 16px;
        font-size: 0.7rem;
        color: #333360;
        z-index: 9999;
        pointer-events: none;
        letter-spacing: 0.06em;
    }

    /* ── Rank number ── */
    .rank-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: rgba(108,99,255,0.2);
        color: #9c8fff;
        font-weight: 800;
        font-size: 0.85rem;
        margin-right: 8px;
        flex-shrink: 0;
    }
    </style>
    """, unsafe_allow_html=True)


def show_paper_card(row, rank: int = None):
    """Render a premium paper card."""
    score_html = ""
    if "similarity_score" in row:
        score = float(row["similarity_score"])
        cls   = ("score-high"   if score > 0.5
                 else "score-medium" if score > 0.25
                 else "score-low")
        score_html = (
            f'<span class="score-badge {cls}">⚡ {score:.4f}</span>'
        )

    rank_html = (
        f'<span class="rank-number">#{rank}</span>' if rank else ""
    )
    title    = row.get("title",    "Untitled")
    category = row.get("category", "")
    year     = int(row["year"]) if "year" in row and pd.notna(row.get("year")) else ""

    st.markdown(f"""
    <div class="paper-card">
        <div class="paper-title">{rank_html}{title}</div>
        <div class="paper-meta">
            {"<span>🏷️ " + str(category) + "</span>" if category else ""}
            {"<span>📅 " + str(year) + "</span>" if year else ""}
            {score_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    summary = row.get("summary", "")
    if summary:
        with st.expander("📄 View Abstract"):
            st.markdown(
                f"<p style='color:#9090b8;font-size:0.9rem;"
                f"line-height:1.7'>{summary}</p>",
                unsafe_allow_html=True,
            )


def credit_footer():
    st.markdown(
        '<div class="credit-footer">✦ Made by Tanusha Chopra</div>',
        unsafe_allow_html=True,
    )