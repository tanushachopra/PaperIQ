# modules/research_copilot/synthesizer.py
import os
import json
import re
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from dotenv import load_dotenv
load_dotenv()


def get_groq_client():
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[Synthesizer] ERROR: GROQ_API_KEY not found")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"[Synthesizer] ERROR creating Groq client: {e}")
        return None


def clean_json_response(raw: str) -> str:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        raw = match.group(0)
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    return raw.strip()


def build_landscape(papers_df, extracted_insights: list) -> dict:
    """Cluster papers and generate landscape using LLM."""
    abstracts = papers_df["abstract"].tolist()
    titles    = papers_df["title"].tolist()

    # ── Clustering ──────────────────────────────────────────────────────────────
    try:
        from modules.research_copilot.embedder import embed_texts
        embeddings = embed_texts(abstracts)
        n_clusters = min(4, max(2, len(papers_df) // 3))
        kmeans     = KMeans(n_clusters=n_clusters,
                            random_state=42, n_init=10)
        labels     = kmeans.fit_predict(embeddings)
        print(f"[Synthesizer] Clustering done: {n_clusters} clusters")
    except Exception as e:
        print(f"[Synthesizer] Clustering error: {e}")
        labels = [i % 4 for i in range(len(titles))]

    clusters_dict = {}
    for i, label in enumerate(labels):
        clusters_dict.setdefault(int(label), []).append(titles[i])

    # ── LLM cluster naming ──────────────────────────────────────────────────────
    client   = get_groq_client()
    landscape = {"clusters": []}

    if client:
        prompt = (
            "You are a research analyst. "
            "Name each cluster of papers and describe the research theme.\n"
            "Return ONLY a JSON object. No markdown. No explanation.\n\n"
            "Format:\n"
            '{"clusters": [{"id": 0, "name": "Theme Name", '
            '"description": "One sentence description"}]}\n\n'
            f"Clusters:\n{json.dumps(clusters_dict, indent=2)}\n\n"
            "JSON output:"
        )

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "Output only valid JSON. No markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500,
            )

            raw     = response.choices[0].message.content
            cleaned = clean_json_response(raw)

            try:
                parsed = json.loads(cleaned)
                if "clusters" in parsed and parsed["clusters"]:
                    landscape = parsed
                    print("[Synthesizer] Landscape generated successfully")
            except json.JSONDecodeError as e:
                print(f"[Synthesizer] JSON parse error: {e}")

        except Exception as e:
            print(f"[Synthesizer] LLM error: {e}")

    # fallback if LLM failed
    if not landscape["clusters"]:
        landscape["clusters"] = [
            {
                "id":          k,
                "name":        f"Research Theme {k + 1}",
                "description": f"A cluster of {len(v)} related papers"
            }
            for k, v in clusters_dict.items()
        ]

    landscape["paper_cluster_map"] = {
        titles[i]: int(labels[i]) for i in range(len(titles))
    }

    return landscape


def find_research_gaps(extracted_insights: list) -> str:
    """Generate research gap analysis from extracted paper insights."""

    if not extracted_insights:
        return "No papers available for gap analysis."

    client = get_groq_client()
    if not client:
        return "Groq API unavailable. Check your API key in .env file."

    # ── Build summaries skipping empty fields ───────────────────────────────────
    skip = {"not specified", "could not extract", "n/a", "", None}
    summaries = []

    for i, p in enumerate(extracted_insights):
        lines = [f"Paper {i+1}: {p.get('title', 'Unknown Title')}"]
        for field in ["problem", "methodology", "results",
                      "limitations", "future_work"]:
            val = str(p.get(field, "")).strip()
            if val.lower() not in skip and len(val) > 5:
                lines.append(f"  {field.replace('_',' ').title()}: {val[:300]}")
        if len(lines) > 1:
            summaries.append("\n".join(lines))

    print(f"[Synthesizer] Gap analysis: {len(summaries)} papers with content")

    # if ALL papers failed extraction, use raw abstracts instead
    if not summaries:
        print("[Synthesizer] Falling back to raw abstracts for gap analysis")
        for i, (_, row) in enumerate(
            zip(range(len(extracted_insights)), [{}]*len(extracted_insights))
        ):
            pass

        # use titles only as last resort
        titles_list = [p.get("title", "") for p in extracted_insights if p.get("title")]
        if titles_list:
            topic_hint = "\n".join(f"- {t}" for t in titles_list[:10])
            summaries_text = (
                f"These are the titles of recent papers on this topic:\n{topic_hint}"
            )
        else:
            return "Insufficient data to generate gap analysis."
    else:
        summaries_text = "\n\n".join(summaries[:10])

    # ── Gap analysis prompt ────────────────────────────────────────────────────
    prompt = (
        "You are a senior research advisor. "
        "Analyze the following research papers and identify gaps.\n\n"
        "Write a detailed analysis with these 4 sections:\n\n"
        "1. UNDEREXPLORED AREAS\n"
        "What important problems or approaches have not been studied enough?\n\n"
        "2. CONTRADICTIONS AND TENSIONS\n"
        "Where do these papers disagree or use conflicting approaches?\n\n"
        "3. PROMISING FUTURE DIRECTIONS\n"
        "What are the 3 most promising areas for new research?\n\n"
        "4. RECOMMENDATION FOR NEW RESEARCHERS\n"
        "What should someone new to this field focus on first?\n\n"
        f"Papers:\n{summaries_text}\n\n"
        "Write your analysis:"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior research advisor who gives "
                        "specific, actionable, well-structured research analyses. "
                        "Always write in clear paragraphs."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1200,
        )

        result = response.choices[0].message.content
        print(f"[Synthesizer] Gap analysis length: {len(result)} chars")

        if result and len(result.strip()) > 100:
            return result.strip()
        else:
            return "Gap analysis returned insufficient content. Please try again."

    except Exception as e:
        print(f"[Synthesizer] Gap finder exception: {type(e).__name__}: {e}")
        return f"Error generating gap analysis: {str(e)[:150]}"


def build_relationship_graph(landscape: dict, papers_df) -> nx.Graph:
    """Build NetworkX graph from cluster data."""
    G           = nx.Graph()
    titles      = papers_df["title"].tolist()
    cluster_map = landscape.get("paper_cluster_map", {})
    clusters    = landscape.get("clusters", [])

    for cluster in clusters:
        G.add_node(
            cluster.get("name", "Unknown"),
            node_type="cluster",
            size=30,
            color="#6c63ff",
        )

    for title in titles:
        cluster_id = cluster_map.get(title, 0)
        if cluster_id < len(clusters):
            cluster_name = clusters[cluster_id].get("name", "General")
        else:
            cluster_name = "General"
            if "General" not in G.nodes:
                G.add_node("General", node_type="cluster",
                           size=30, color="#6c63ff")

        short = title[:45] + "..." if len(title) > 45 else title
        G.add_node(short, node_type="paper", size=10, color="#4ecca3")
        G.add_edge(cluster_name, short)

    return G