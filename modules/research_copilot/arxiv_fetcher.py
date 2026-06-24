import arxiv
import pandas as pd

def fetch_papers(topic: str, max_results: int = 50) -> pd.DataFrame:

    client = arxiv.Client()

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    records = []

    for paper in client.results(search):
        records.append({
            "title": paper.title,
            "abstract": paper.summary.replace("\n", " "),
            "authors": ", ".join(a.name for a in paper.authors[:3]),
            "date": paper.published.strftime("%Y-%m-%d"),
            "arxiv_id": paper.entry_id.split("/")[-1],
            "url": paper.entry_id,
        })

    return pd.DataFrame(records)