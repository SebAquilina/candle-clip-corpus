"""Resolve YouTube video IDs from TITLES via search (only used when the user wants Claude
to source videos from a title list, e.g. a content plan). Importable: `keywords()` and
`search()`. Also runnable as a script over a JSON list of titles.

Config: REVAMP_COOKIES (cookie file), REVAMP_NICHE_HINT (topic words appended to each
query, default "candle making").
"""
from __future__ import annotations
import os, re, json, sys, glob

_NICHE = os.environ.get("REVAMP_NICHE_HINT", "candle making")
_COOKIES = os.environ.get("REVAMP_COOKIES", "/tmp/revamp_cookies.txt")
# clickbait framing to strip so the search hits the actual topic, not the angle
_STRIP = re.compile(
    r"\b(i|tested|tried|melted|burned|burning|almost|all|of|them|are|is|lies|here'?s|what|"
    r"went|wrong|stop|buying|can|you|make|made|for|hours|straight|every|my|house|together|"
    r"the|a|an|in|with|vs)\b", re.I)


def keywords(title: str) -> str:
    """Turn a (clickbait) title into a short topical search query."""
    t = re.split(r"[—:?\-]", title or "")[0]          # topic is usually before the dash/colon/?
    t = _STRIP.sub(" ", t)
    t = re.sub(r"[^A-Za-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return (f"{t} {_NICHE}").strip()[:80]


def search(q: str, n: int = 2) -> list[str]:
    """Top-n YouTube video IDs for a query (metadata only, no download)."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "nocheckcertificate": True,
            "extractor_args": {"youtube": {"player_client": ["mweb", "web"]}}}
    if os.path.exists(_COOKIES):
        opts["cookiefile"] = _COOKIES
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(f"ytsearch{n}:{q}", download=False)
    return [e.get("id") for e in (info.get("entries") or []) if e.get("id")]


def _existing_ids():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    ids = set()
    for f in (glob.glob(base + "/shared_db/records/*.json")
              + glob.glob(base + "/shared_db_v2/records/*.json")
              + glob.glob(base + "/shared_db_v2/rejected/*.json")):
        try:
            ids.add(json.load(open(f)).get("video_id"))
        except Exception:
            pass
    return ids


def resolve_titles(titles, per_title=2, exclude=None):
    """titles -> deduped new video ids (excluding the corpus + anything in `exclude`)."""
    seen = set(exclude or set()) | _existing_ids()
    out = []
    for t in titles:
        q = keywords(t)
        try:
            ids = search(q, per_title)
        except Exception as e:
            print(f"  [searchfail] {t[:40]}: {str(e)[:50]}"); continue
        new = [v for v in ids if v not in seen]
        for v in new:
            seen.add(v); out.append(v)
        print(f"  '{q}' -> {ids} (new: {len(new)})")
    return out


if __name__ == "__main__":
    # batch mode: read a JSON list of titles (arg1 or /tmp/workbook_titles.json), write ids
    src = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else "/tmp/workbook_titles.json"
    titles = json.load(open(src))
    out = resolve_titles(titles, per_title=2)
    json.dump(out, open("/tmp/search_ids.json", "w"))
    print(f"\n{len(out)} new deduped ids from {len(titles)} titles")
