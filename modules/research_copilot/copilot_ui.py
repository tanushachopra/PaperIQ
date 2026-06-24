# modules/research_copilot/copilot_ui.py — Premium UI v2

import os
import streamlit as st
import streamlit.components.v1 as components
import tempfile
from pyvis.network import Network

from modules.research_copilot.arxiv_fetcher import fetch_papers
from modules.research_copilot.embedder      import shortlist_by_embedding
from modules.research_copilot.reranker      import rerank
from modules.research_copilot.extractor     import extract_all_papers
from modules.research_copilot.synthesizer   import (
    build_landscape, find_research_gaps, build_relationship_graph
)

# ── Insight field colors ────────────────────────────────────────────────────────
INSIGHT_COLORS = {
    "problem":       ("#ff6b6b", "🎯", "Problem Statement"),
    "methodology":   ("#6c63ff", "⚙️", "Methodology"),
    "contributions": ("#4ecca3", "✨", "Key Contributions"),
    "results":       ("#f0a500", "📊", "Results"),
    "limitations":   ("#ff9f7f", "⚠️", "Limitations"),
    "future_work":   ("#7ec8e3", "🔮", "Future Work"),
}


def _metric_card(label: str, value: str, icon: str, color: str):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,
            rgba({color},0.08) 0%, rgba(20,20,45,0.7) 100%);
        border: 1px solid rgba({color},0.25);
        border-radius: 16px;
        padding: 20px 22px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s;
    ">
        <div style="font-size:1.8rem;margin-bottom:6px">{icon}</div>
        <div style="font-size:1.9rem;font-weight:800;
                    color:rgba({color},1);line-height:1">{value}</div>
        <div style="font-size:0.72rem;color:#5050a0;
                    text-transform:uppercase;letter-spacing:0.1em;
                    margin-top:6px;font-weight:600">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def _section_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-size:1.3rem;font-weight:700;
                    color:#e8e0ff;margin-bottom:4px">
            {icon} {title}
        </div>
        {"<div style='font-size:0.85rem;color:#5050a0'>" + subtitle + "</div>"
         if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def render_copilot_page():
    """Main Research Copilot UI — premium redesign."""

    # ── Hero Section ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:40px 0 32px 0">
        <div class="hero-badge">✦ AI-Powered Research Intelligence</div>
        <div class="hero-title">Research Copilot</div>
        <div class="hero-subtitle" style="margin:0 auto">
            Enter any research topic — fetch live papers, rank by semantic
            relevance, extract structured insights, and map the research landscape.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search Box ─────────────────────────────────────────────────────────────
    search_col, btn_col = st.columns([0.82, 0.18])

    with search_col:
        topic = st.text_input(
            label="",
            placeholder="🔬  e.g.  Retrieval Augmented Generation, "
                        "Diffusion Models, Federated Learning...",
            label_visibility="collapsed",
            key="copilot_topic",
        )

    with btn_col:
        search_clicked = st.button(
            "Explore →",
            type="primary",
            use_container_width=True,
            key="copilot_search_btn",
        )

    # ── Settings Row ───────────────────────────────────────────────────────────
    settings_col1, settings_col2, settings_col3 = st.columns([1, 1, 2])
    with settings_col1:
        max_results = st.slider(
            "Papers to fetch", 10, 30, 15,
            key="copilot_max_results"
        )
    with settings_col2:
        top_n_display = st.slider(
            "Top papers", 5, 15, 10,
            key="copilot_top_n"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Execution ─────────────────────────────────────────────────────
    if search_clicked:
        if not topic.strip():
            st.warning("⚠️  Please enter a research topic to explore.")
            return

        # Premium loading experience
        st.markdown("""
        <div style="
            background: rgba(108,99,255,0.05);
            border: 1px solid rgba(108,99,255,0.15);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 20px;
        ">
            <div style="font-size:0.78rem;font-weight:700;
                        color:#6c63ff;letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:14px">
                ⚡ Running Research Intelligence Pipeline
            </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text  = st.empty()

        steps = [
            (10,  "📡",  "Fetching papers from arXiv..."),
            (30,  "🧠",  "Computing semantic embeddings..."),
            (50,  "⚡",  "Cross-encoder reranking for precision..."),
            (65,  "📝",  "Extracting structured insights with LLM..."),
            (80,  "🗺️", "Building research landscape clusters..."),
            (95,  "🔭",  "Identifying research gaps..."),
            (100, "✅",  "Analysis complete!"),
        ]

        st.markdown("</div>", unsafe_allow_html=True)

        try:
            # Step 1
            progress_bar.progress(10)
            status_text.markdown(
                "<p style='color:#6c63ff;font-size:0.88rem'>"
                "📡 Fetching papers from arXiv...</p>",
                unsafe_allow_html=True
            )
            papers_df = fetch_papers(topic, max_results=max_results)

            # Step 2
            progress_bar.progress(30)
            status_text.markdown(
                "<p style='color:#6c63ff;font-size:0.88rem'>"
                "🧠 Computing semantic embeddings...</p>",
                unsafe_allow_html=True
            )
            shortlisted = shortlist_by_embedding(topic, papers_df, top_n=20)

            # Step 3
            progress_bar.progress(50)
            status_text.markdown(
                "<p style='color:#6c63ff;font-size:0.88rem'>"
                "⚡ Cross-encoder reranking for precision...</p>",
                unsafe_allow_html=True
            )
            top_papers = rerank(topic, shortlisted, top_n=top_n_display)

            # Step 4
            progress_bar.progress(65)
            status_text.markdown(
                "<p style='color:#6c63ff;font-size:0.88rem'>"
                "📝 Extracting structured insights with LLM...</p>",
                unsafe_allow_html=True
            )
            extracted = extract_all_papers(top_papers)

            # Step 5
            progress_bar.progress(80)
            status_text.markdown(
                "<p style='color:#6c63ff;font-size:0.88rem'>"
                "🗺️ Building research landscape clusters...</p>",
                unsafe_allow_html=True
            )
            landscape = build_landscape(top_papers, extracted)

            # Step 6
            progress_bar.progress(95)
            status_text.markdown(
                "<p style='color:#6c63ff;font-size:0.88rem'>"
                "🔭 Identifying research gaps...</p>",
                unsafe_allow_html=True
            )
            gaps = find_research_gaps(extracted)

            # Done
            progress_bar.progress(100)
            status_text.markdown(
                "<p style='color:#4ecca3;font-size:0.88rem;font-weight:600'>"
                "✅ Analysis complete!</p>",
                unsafe_allow_html=True
            )

            st.session_state.copilot_results = {
                "papers":    top_papers,
                "extracted": extracted,
                "landscape": landscape,
                "gaps":      gaps,
                "topic":     topic,
            }

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            return

    # ── Results ────────────────────────────────────────────────────────────────
    if "copilot_results" not in st.session_state:
        # Empty state
        st.markdown("""
        <div style="
            text-align:center;
            padding:60px 20px;
            color:#2a2a60;
        ">
            <div style="font-size:3rem;margin-bottom:12px">🔬</div>
            <div style="font-size:1rem;font-weight:500">
                Enter a topic above to begin your research exploration
            </div>
            <div style="font-size:0.85rem;margin-top:8px">
                Try: "Large Language Models", "Computer Vision",
                "Federated Learning"
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    r = st.session_state.copilot_results

    # ── Results Header ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,
            rgba(108,99,255,0.08) 0%, rgba(20,20,40,0.5) 100%);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 16px;
        padding: 20px 26px;
        margin: 24px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <div style="font-size:1.6rem">🧠</div>
        <div>
            <div style="font-size:0.72rem;font-weight:700;
                        color:#6c63ff;text-transform:uppercase;
                        letter-spacing:0.1em">Research Analysis</div>
            <div style="font-size:1.3rem;font-weight:700;color:#e8e0ff">
                {r['topic']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ───────────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        _metric_card("Papers Retrieved",  str(len(r["papers"])),
                     "📄", "108,99,255")
    with mc2:
        _metric_card("Top Selected",
                     str(len(r["papers"])), "⚡", "78,204,163")
    with mc3:
        n_clusters = len(r["landscape"].get("clusters", []))
        _metric_card("Research Clusters", str(n_clusters),
                     "🗺️", "240,165,0")
    with mc4:
        has_gaps = (r["gaps"] and
                    len(r["gaps"]) > 50 and
                    "not generate" not in r["gaps"].lower())
        _metric_card("Gap Analysis",
                     "Ready" if has_gaps else "Limited",
                     "🔭", "156,95,255")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄  Top Papers",
        "🧠  Deep Insights",
        "🗺️  Research Map",
        "🔭  Research Gaps",
    ])

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 1 — TOP PAPERS
    # ────────────────────────────────────────────────────────────────────────────
    with tab1:
        _section_header(
            "📄", "Top Ranked Papers",
            f"Ranked by semantic relevance to '{r['topic']}' "
            f"using bi-encoder + cross-encoder pipeline"
        )

        for i, (_, row) in enumerate(r["papers"].iterrows(), start=1):
            score = float(row.get("rerank_score", 0))
            cls   = ("score-high"   if score > 0.5
                     else "score-medium" if score > 0.2
                     else "score-low")

            st.markdown(f"""
            <div class="paper-card">
                <div style="display:flex;align-items:flex-start;gap:12px">
                    <div class="rank-number">#{i}</div>
                    <div style="flex:1">
                        <div class="paper-title">{row['title']}</div>
                        <div class="paper-meta">
                            <span>👤 {row.get('authors','Unknown')}</span>
                            <span>📅 {row.get('date','')}</span>
                            <span class="score-badge {cls}">
                                ⚡ {score:.4f}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"Read abstract — {row['title'][:55]}..."):
                st.markdown(
                    f"<p style='color:#9090b8;font-size:0.9rem;"
                    f"line-height:1.7'>{row.get('abstract','')}</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"[🔗 View on arXiv →]({row.get('url','')})",
                    unsafe_allow_html=False,
                )

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 2 — DEEP INSIGHTS
    # ────────────────────────────────────────────────────────────────────────────
    with tab2:
        _section_header(
            "🧠", "Structured Paper Insights",
            "LLM-extracted problem, methodology, contributions, "
            "results, limitations and future work per paper"
        )

        for insight in r["extracted"]:
            title = insight.get("title", "Unknown Paper")

            with st.expander(f"📄  {title[:75]}..."):
                # top row: 3 fields
                row1 = st.columns(3)
                row2 = st.columns(3)

                fields_ordered = [
                    "problem", "methodology", "contributions",
                    "results", "limitations", "future_work"
                ]

                for col_idx, field in enumerate(fields_ordered):
                    color, icon, label = INSIGHT_COLORS[field]
                    value = insight.get(field, "Not specified")

                    target_col = (row1 if col_idx < 3 else row2)[col_idx % 3]
                    with target_col:
                        st.markdown(f"""
                        <div class="insight-field">
                            <div class="insight-label"
                                 style="color:{color}">
                                {icon} {label}
                            </div>
                            <div class="insight-value">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # arXiv link
                url = insight.get("url", "")
                if url:
                    st.markdown(
                        f"<div style='margin-top:8px'>"
                        f"[🔗 View on arXiv →]({url})</div>",
                        unsafe_allow_html=False,
                    )

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 3 — RESEARCH MAP
    # ────────────────────────────────────────────────────────────────────────────
    with tab3:
        _section_header(
            "🗺️", "Research Landscape Map",
            "Papers clustered by semantic similarity — "
            "explore thematic relationships"
        )

        clusters   = r["landscape"].get("clusters", [])
        cluster_map = r["landscape"].get("paper_cluster_map", {})

        # cluster count per group
        cluster_counts = {}
        for title, cid in cluster_map.items():
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        # cluster summary cards
        if clusters:
            cols = st.columns(min(len(clusters), 2))
            for idx, cluster in enumerate(clusters):
                cid   = cluster.get("id", idx)
                count = cluster_counts.get(cid, 0)
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div class="cluster-card">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:flex-start;margin-bottom:8px">
                            <div style="font-size:0.95rem;font-weight:700;
                                        color:#c8b8ff">
                                {cluster.get('name','Cluster')}
                            </div>
                            <div style="
                                background:rgba(108,99,255,0.2);
                                color:#9c8fff;
                                font-size:0.72rem;
                                font-weight:700;
                                padding:2px 10px;
                                border-radius:12px;
                                border:1px solid rgba(108,99,255,0.3)">
                                {count} paper{"s" if count != 1 else ""}
                            </div>
                        </div>
                        <div style="font-size:0.85rem;color:#6060a0;
                                    line-height:1.5">
                            {cluster.get('description','')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # interactive graph
        try:
            G   = build_relationship_graph(r["landscape"], r["papers"])
            net = Network(
                height="520px", width="100%",
                bgcolor="#080810", font_color="#c8b8ff",
            )
            net.from_nx(G)
            net.set_options("""{
                "nodes": {
                    "font": {"size": 13, "color": "#c8b8ff"},
                    "borderWidth": 2
                },
                "edges": {
                    "color": {"color": "rgba(108,99,255,0.4)"},
                    "width": 1.5,
                    "smooth": {"type": "continuous"}
                },
                "physics": {
                    "stabilization": {"iterations": 150},
                    "barnesHut": {
                        "gravitationalConstant": -3000,
                        "springLength": 120
                    }
                }
            }""")

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".html",
                mode="w", encoding="utf-8"
            ) as tmp:
                net.save_graph(tmp.name)
                html_content = open(tmp.name, encoding="utf-8").read()
            os.unlink(tmp.name)

            components.html(html_content, height=540)

        except Exception as e:
            st.error(f"Graph rendering error: {e}")

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 4 — RESEARCH GAPS
    # ────────────────────────────────────────────────────────────────────────────
    with tab4:
        _section_header(
            "🔭", "Research Gap Analysis",
            "AI-synthesized view of underexplored areas, "
            "contradictions, and future opportunities"
        )

        gaps_text = r.get("gaps", "")

        if (not gaps_text or
            len(gaps_text) < 80 or
            "not generate" in gaps_text.lower()):
            st.markdown("""
            <div style="
                text-align:center;padding:40px;
                color:#303060;border:1px dashed rgba(108,99,255,0.2);
                border-radius:16px;
            ">
                <div style="font-size:2rem;margin-bottom:8px">🔭</div>
                <div>Gap analysis unavailable for this search.</div>
                <div style="font-size:0.85rem;margin-top:6px">
                    Try a more specific research topic.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # parse sections from LLM output
            sections = {
                "UNDEREXPLORED":    ("gap-unexplored",    "🌱",
                                     "Underexplored Areas",    "#4ecca3"),
                "CONTRADICTION":    ("gap-contradiction", "⚡",
                                     "Contradictions & Tensions", "#ff6b6b"),
                "FUTURE":           ("gap-opportunity",   "🚀",
                                     "Future Directions",      "#6c63ff"),
                "RECOMMENDATION":   ("gap-recommendation","💡",
                                     "For New Researchers",    "#f0a500"),
                "PROMISING":        ("gap-opportunity",   "🚀",
                                     "Promising Directions",   "#6c63ff"),
            }

            lines      = gaps_text.split("\n")
            current    = []
            cur_section = None
            rendered   = False

            for line in lines:
                line_up = line.upper()

                matched = None
                for key, (css, icon, label, color) in sections.items():
                    if key in line_up and (
                        line.strip().startswith("#") or
                        any(c.isdigit() for c in line[:3]) or
                        line.strip().isupper() or
                        "." in line[:5]
                    ):
                        matched = (css, icon, label, color)
                        break

                if matched:
                    # flush previous section
                    if current and cur_section:
                        css, icon, label, color = cur_section
                        content = " ".join(
                            l for l in current if l.strip()
                        ).strip()
                        if content:
                            st.markdown(f"""
                            <div class="gap-section {css}">
                                <div class="gap-title"
                                     style="color:{color}">
                                    {icon} {label}
                                </div>
                                <div class="gap-content">{content}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            rendered = True
                    current     = []
                    cur_section = matched
                else:
                    if line.strip():
                        current.append(line.strip())

            # flush last section
            if current and cur_section:
                css, icon, label, color = cur_section
                content = " ".join(
                    l for l in current if l.strip()
                ).strip()
                if content:
                    st.markdown(f"""
                    <div class="gap-section {css}">
                        <div class="gap-title" style="color:{color}">
                            {icon} {label}
                        </div>
                        <div class="gap-content">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    rendered = True

            # if parsing found nothing structured, show raw nicely
            if not rendered:
                paragraphs = [
                    p.strip()
                    for p in gaps_text.split("\n\n")
                    if p.strip()
                ]
                colors_cycle = [
                    ("gap-unexplored",    "#4ecca3", "🌱"),
                    ("gap-contradiction", "#ff6b6b", "⚡"),
                    ("gap-opportunity",   "#6c63ff", "🚀"),
                    ("gap-recommendation","#f0a500", "💡"),
                ]
                for i, para in enumerate(paragraphs):
                    css, color, icon = colors_cycle[i % len(colors_cycle)]
                    st.markdown(f"""
                    <div class="gap-section {css}">
                        <div class="gap-content">{para}</div>
                    </div>
                    """, unsafe_allow_html=True)