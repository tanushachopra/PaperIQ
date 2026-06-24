# modules/research_copilot/extractor.py
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

# ── Groq client (lazy import to avoid cascade errors) ──────────────────────────
def get_groq_client():
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[Extractor] ERROR: GROQ_API_KEY not found in environment")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"[Extractor] ERROR creating Groq client: {e}")
        return None


def clean_and_parse_json(raw: str) -> dict | None:
    """
    Try every possible way to extract valid JSON from LLM response.
    Returns parsed dict or None if all attempts fail.
    """
    if not raw:
        return None

    attempts = []

    # Attempt 1: direct parse
    attempts.append(raw.strip())

    # Attempt 2: strip markdown fences
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*",     "", cleaned)
    attempts.append(cleaned.strip())

    # Attempt 3: extract just the {...} block
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        attempts.append(match.group(0))

    # Attempt 4: fix common issues then extract
    fixed = re.sub(r"```json\s*", "", raw)
    fixed = re.sub(r"```\s*",     "", fixed)
    fixed = re.sub(r",\s*}",      "}", fixed)
    fixed = re.sub(r",\s*]",      "]", fixed)
    match2 = re.search(r"\{[\s\S]*\}", fixed)
    if match2:
        attempts.append(match2.group(0))

    for attempt in attempts:
        try:
            result = json.loads(attempt)
            if isinstance(result, dict):
                return result
        except:
            continue

    return None


def extract_paper_insights(abstract: str) -> dict:
    """
    Extract structured insights from paper abstract using Groq.
    """
    fallback = {
        "problem":       "Not specified",
        "methodology":   "Not specified",
        "contributions": "Not specified",
        "results":       "Not specified",
        "limitations":   "Not specified",
        "future_work":   "Not specified",
    }

    if not abstract or len(abstract.strip()) < 30:
        return fallback

    client = get_groq_client()
    if not client:
        return fallback

    # Simple, unambiguous prompt
    prompt = (
        "Read this research paper abstract and extract 6 pieces of information.\n"
        "Return ONLY a JSON object with exactly these keys.\n"
        "Do not include any text before or after the JSON.\n"
        "Do not use markdown formatting.\n\n"
        "Required JSON format:\n"
        '{"problem": "the main problem addressed",'
        '"methodology": "the method or approach used",'
        '"contributions": "key contributions of this work",'
        '"results": "main results or findings",'
        '"limitations": "limitations of this work",'
        '"future_work": "future directions mentioned"}\n\n'
        f"Abstract:\n{abstract[:1500]}\n\n"
        "JSON output:"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research analyst. "
                        "You output only valid JSON objects. "
                        "Never use markdown. Never add explanation."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=500,
        )

        raw = response.choices[0].message.content
        print(f"[Extractor] Raw response preview: {repr(raw[:100])}")

        parsed = clean_and_parse_json(raw)

        if parsed:
            # fill any missing keys with fallback
            for key in fallback:
                if key not in parsed or not str(parsed[key]).strip():
                    parsed[key] = "Not specified"
            return parsed
        else:
            print(f"[Extractor] Could not parse JSON from: {repr(raw[:200])}")
            # last resort: regex extraction per field
            result = {}
            for key in fallback:
                pattern = rf'["\']?{key}["\']?\s*:\s*["\']([^"\'{{}}]+)["\']'
                m = re.search(pattern, raw, re.IGNORECASE)
                result[key] = m.group(1).strip() if m else "Not specified"
            return result

    except Exception as e:
        print(f"[Extractor] Exception: {type(e).__name__}: {e}")
        return fallback


def extract_all_papers(papers_df) -> list:
    """Extract insights for all papers."""
    results = []
    total   = len(papers_df)

    for idx, (_, row) in enumerate(papers_df.iterrows()):
        title = row.get("title", "Unknown")
        print(f"[Extractor] Paper {idx+1}/{total}: {title[:50]}")

        insights          = extract_paper_insights(row.get("abstract", ""))
        insights["title"] = title
        insights["url"]   = row.get("url",  "")
        insights["date"]  = row.get("date", "")
        results.append(insights)

    successful = sum(
        1 for r in results
        if r.get("problem", "Not specified") != "Not specified"
    )
    print(f"[Extractor] Complete: {successful}/{total} extracted successfully")
    return results