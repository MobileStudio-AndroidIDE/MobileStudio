"""Web search — the web as an EXTERNAL EVIDENCE source.

The pipeline is unchanged: UNDERSTAND → RETRIEVE RELEVANT EVIDENCE →
REASON → COMPOSE → VERIFY → OUTPUT.  The concept store is the first
evidence source (local, fast, curated); when it has no answer, the web
retrieves external evidence — the same reasoning consumes it, and the
response cites the source (honest attribution, never a silent lift).

Sources (both public, keyless, stdlib-only):
  - Wikipedia's summary API — structured knowledge for topic questions
  - DuckDuckGo lite — the general web for everything else

Network failures degrade gracefully (an empty result list — the
caller falls back to the honest no-knowledge response).  A per-query
TTL cache keeps a conversation (and a test run) from re-fetching.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

_USER_AGENT = "RENA-BR/0.5 (autonomous agent; educational)"
_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_TIMEOUT = 6.0
_CACHE_TTL = 900.0  # 15 minutes
_CACHE: dict[str, tuple[float, list]] = {}
_MAX_RESULTS = 5


@dataclass
class WebResult:
    """One retrieved external evidence item."""
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""  # wikipedia / web

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url,
                "snippet": self.snippet, "source": self.source}


def _fetch(url: str, data: bytes | None = None,
           headers: dict | None = None) -> str:
    merged = {"User-Agent": _USER_AGENT}
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, headers=merged, data=data)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        raw = resp.read(512 * 1024)  # hard size cap
    return raw.decode("utf-8", "ignore")


def _cached(key: str) -> list | None:
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    return None


def _store(key: str, results: list) -> list:
    _CACHE[key] = (time.time(), results)
    return results


def wikipedia_summary(topic: str) -> WebResult | None:
    """The Wikipedia summary of a topic — structured knowledge with
    attribution.  None when the topic has no article (or offline)."""
    key = "wiki:" + topic.lower()
    cached = _cached(key)
    if cached is not None:
        return cached[0] if cached else None
    title = urllib.parse.quote(topic.replace(" ", "_"), safe="")
    url = (f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
    try:
        data = json.loads(_fetch(url))
    except Exception:
        _store(key, [])
        return None
    extract = str(data.get("extract") or "").strip()
    if not extract or data.get("type") == "disambiguation":
        _store(key, [])
        return None
    result = WebResult(
        title=str(data.get("title") or topic),
        url=str(data.get("content_urls", {})
                .get("desktop", {}).get("page") or url),
        snippet=extract,
        source="wikipedia",
    )
    return _store(key, [result])[0]


def wikipedia_extract(topic: str, max_chars: int = 8000) -> WebResult | None:
    """The article's OWN text — for a turn that asked for depth.

    The summary API carries one paragraph; the extract API carries the
    article.  Same source, same attribution, same evidence path: only
    how much of it the turn may read differs, and the read is bounded
    so a deep answer draws a pool rather than a corpus."""
    key = f"wikix:{topic.lower()}:{max_chars}"
    cached = _cached(key)
    if cached is not None:
        return cached[0] if cached else None
    title = urllib.parse.quote(topic.replace(" ", "_"), safe="")
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           "&prop=extracts&explaintext=1&redirects=1&titles=" + title)
    try:
        data = json.loads(_fetch(url))
    except Exception:
        _store(key, [])
        return None
    pages = (data.get("query") or {}).get("pages") or {}
    extract = ""
    page_title = topic
    for page in pages.values():
        candidate = str(page.get("extract") or "").strip()
        if candidate:
            extract = candidate
            page_title = str(page.get("title") or topic)
            break
    if not extract:
        _store(key, [])
        return None
    if len(extract) > max_chars:
        cut = extract[:max_chars]
        dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if dot > max_chars // 2:
            cut = cut[:dot + 1]
        extract = cut
    result = WebResult(
        title=page_title,
        url=f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
        snippet=extract,
        source="wikipedia",
    )
    return _store(key, [result])[0]


def wikipedia_search(query: str, limit: int = 5) -> list[str]:
    """The article titles Wikipedia's own search ranks for a query.

    A request states what it wants as a sentence ("how stars are
    born"); an article is addressed by a noun phrase.  The search API
    is the bridge between the two — a general retrieval step, never a
    mapping from a particular question to a particular article."""
    key = "wikis:" + query.lower()
    cached = _cached(key)
    if cached is not None:
        return [getattr(r, "title", "") for r in cached]
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           "&list=search&srlimit=" + str(max(1, int(limit))) +
           "&srsearch=" + urllib.parse.quote(query))
    try:
        data = json.loads(_fetch(url))
    except Exception:
        _store(key, [])
        return []
    hits = ((data.get("query") or {}).get("search")) or []
    titles = [str(h.get("title") or "").strip() for h in hits]
    titles = [t for t in titles if t]
    _store(key, [WebResult(title=t, source="wikipedia") for t in titles])
    return titles


def _disambiguation_title(title: str) -> bool:
    """A title whose qualifier names a different SENSE of the name — a
    television series, a film, a band, a year, an explicit
    disambiguation page — never the topic the request named.

    A qualifier that only disambiguates the primary sense ("Android
    (operating system)") IS the topic: skipping every parenthetical
    title used to throw the topic away and leave its sub-articles
    ("Android version history") as the answer."""
    t = str(title or "")
    if t.strip().lower().endswith("(disambiguation)"):
        return True
    return bool(_SECONDARY_SENSE.search(t))


_SECONDARY_SENSE = re.compile(
    r"\((?:[^)]*\b(?:tv|television)\s+series\b[^)]*"
    r"|[^)]*\b(?:film|movie|miniseries|sitcom|band|song|album|"
    r"soundtrack|musical|video\s+game|novel|book|magazine|"
    r"newspaper|season|episode)\b[^)]*"
    r"|[^)]*\b(?:1[89]|20)\d{2}\b[^)]*"
    r"|disambiguation)\)",
    re.IGNORECASE)


def wikipedia_sections(topic: str, max_chars: int = 24000) -> list[dict]:
    """The article's own text SPLIT AT ITS OWN HEADINGS.

    A long request asks about a subject in stages; the source states
    the same subject in sections.  Reading the article's structure (its
    headings are its own organisation of the knowledge) lets the
    response's sections follow the source's coverage instead of
    guessing at it.  Returns [{"heading": str, "text": str}] with the
    lead under the empty heading, or [] when the article does not
    exist."""
    key = f"wikisec:{topic.lower()}:{max_chars}"
    cached = _cached(key)
    if cached is not None:
        return [dict(s) for s in cached]
    title = urllib.parse.quote(topic.replace(" ", "_"), safe="")
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           "&prop=extracts&explaintext=1&exsectionformat=wiki"
           "&redirects=1&titles=" + title)
    try:
        data = json.loads(_fetch(url))
    except Exception:
        _store(key, [])
        return []
    pages = ((data.get("query") or {}).get("pages")) or {}
    extract = ""
    page_title = topic
    for page in pages.values():
        candidate = str(page.get("extract") or "").strip()
        if candidate:
            extract = candidate
            page_title = str(page.get("title") or topic)
            break
    if not extract or data.get("type") == "disambiguation":
        _store(key, [])
        return []
    sections: list[dict] = [{"heading": "", "text": ""}]
    for line in extract.split("\n"):
        match = re.match(r"^(={2,6})\s*(.+?)\s*\1\s*$", line.strip())
        if match:
            sections.append({"heading": match.group(2).strip(), "text": ""})
        else:
            if line.strip():
                sections[-1]["text"] += ((" " if sections[-1]["text"]
                                          else "") + line.strip())
    out = [s for s in sections if s["text"].strip()]
    if out and out[0]["heading"]:
        out.insert(0, {"heading": "", "text": ""})
    if out:
        out[0]["page_title"] = page_title
    remaining = max_chars
    kept: list[dict] = []
    for section in out:
        if remaining <= 0:
            break
        text = section["text"]
        if len(text) > remaining:
            cut = text[:remaining]
            dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
            text = cut[:dot + 1] if dot > remaining // 3 else cut
        remaining -= len(text)
        kept.append({"heading": section["heading"], "text": text})
    if kept and "page_title" in out[0]:
        kept[0]["page_title"] = out[0]["page_title"]
    if kept:
        page = str(kept[0].get("page_title") or topic).replace(" ", "_")
        url = ("https://en.wikipedia.org/wiki/" +
               urllib.parse.quote(page, safe="_()"))
        for section in kept:
            section["url"] = url
            section["source"] = "wikipedia"
    _store(key, kept)
    return [dict(s) for s in kept]


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[#\w]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_ddg(html: str, limit: int) -> list[WebResult]:
    """The lite layout: anchor rows (href before its class attr) and
    then the snippet cells, in result order."""
    results: list[WebResult] = []
    links = re.findall(
        r"<a[^>]*href=['\"]([^'\"]+)['\"][^>]*"
        r"class=['\"]result-link['\"][^>]*>(.*?)</a>",
        html, re.DOTALL)
    snippets = re.findall(
        r"class=['\"]result-snippet['\"][^>]*>(.*?)</td>", html, re.DOTALL)
    for i, (href, title) in enumerate(links[:limit]):
        snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
        results.append(WebResult(
            title=_strip_html(title),
            url=_unwrap_redirect(href),
            snippet=snippet,
            source="web",
        ))
    return results


def search_web(query: str, limit: int = _MAX_RESULTS) -> list[WebResult]:
    """The general web search (DuckDuckGo lite) — titles, snippets,
    urls.  Empty on failure (offline, blocked, rate-limited).

    The lite endpoint bot-checks a plain GET with the agent's own
    user-agent; the FORM POST with a browser user-agent is served the
    real result page (verified against this environment).  The GET is
    kept as the fallback for environments where the reverse holds."""
    key = "web:" + query.lower()
    cached = _cached(key)
    if cached is not None:
        return cached
    headers = {"User-Agent": _BROWSER_UA,
               "Accept-Language": "en-US,en;q=0.9"}
    results: list[WebResult] = []
    try:
        body = urllib.parse.urlencode({"q": query}).encode("utf-8")
        html = _fetch("https://lite.duckduckgo.com/lite/", data=body,
                      headers=headers)
        results = _parse_ddg(html, limit)
    except Exception:
        results = []
    if not results:
        try:
            html = _fetch(
                "https://lite.duckduckgo.com/lite/?q=" +
                urllib.parse.quote(query), headers=headers)
            results = _parse_ddg(html, limit)
        except Exception:
            results = []
    return _store(key, results)


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo wraps result links (//duckduckgo.com/l/?uddg=...):
    unwrap the real target url and decode the entities."""
    import html as _html
    href = _html.unescape(href)
    if "uddg=" in href:
        query_part = href.split("?", 1)[-1]
        for part in query_part.split("&"):
            if part.startswith("uddg="):
                return urllib.parse.unquote(part[5:])
    return href


def _content_words(text: str) -> list[str]:
    """The content words of a title or a question — function words
    carry no subject matter and never anchor a match."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", (text or "").lower())
    return [w for w in words
            if len(w) > 2 and w not in _TITLE_STOP]


_TITLE_STOP = {"the", "and", "for", "that", "this", "with", "from",
               "are", "was", "were", "what", "which", "when", "where",
               "how", "why", "you", "your", "its", "his", "her",
               "their", "about", "into", "does", "can", "should"}

_HISTORY_WORDS = {"history", "historical", "timeline", "origin",
                  "origins", "version", "versions", "past", "evolution"}

_PROCEDURAL_RE = re.compile(
    r"(?:\bhow\b|\bways?\s+to\b|\bsteps?\s+to\b|\bmethod\s+to\b|"
    r"what(?:'s|\s+is)\s+the\s+best\s+way\s+to\b|"
    r"어떻게|방법|만드는\s*법|만들기|사용법)", re.IGNORECASE)

# A source that ANSWERS a procedure announces one: a guide, a tutorial,
# steps.  When such sources exist, only they are evidence for a
# procedural ask — a page that merely shares the topic's name does not
# answer "how do I …".
_PROCEDURAL_CUE_RE = re.compile(
    r"(?:how\s+(?:to|can|do|should)\b|step[- ]by[- ]step|\bguide\b|"
    r"\btutorial\b|\bsteps\b|방법|가이드|튜토리얼|만드는\s*법|만들기|"
    r"시작하기|사용법)", re.IGNORECASE)


def _is_procedural(question: str) -> bool:
    """The turn asks for a PROCEDURE: the answer it wants is a way of
    doing something, not a definition of a thing."""
    return bool(_PROCEDURAL_RE.search(question or ""))


def _refine_web_query(topic: str, question: str) -> str:
    """The general web's own query: the request's words, not only the
    topic's name.  A procedure's question is searched as a phrase
    ("how can I make an android app" → "how to make an android app");
    a definition keeps the topic as its encyclopedic address."""
    q = " ".join(str(question or "").split()).strip()
    if q and _is_procedural(q):
        q = re.sub(r"^\s*how\s+(?:can|do|should|would)\s+(?:i|you|we)\b",
                   "how to", q, count=1, flags=re.IGNORECASE)
        q = q.rstrip(" ?.!")
        if q:
            return q
    return topic or q


def _title_score(title: str, topic: str, question: str) -> float:
    """How likely a title IS the topic the request named.

    A title that only qualifies the topic's sense in parentheses
    ("Android (operating system)") is the topic itself; a title that
    EXTENDS it with words the request never asked for ("Android
    version history", "List of ...") is a sub-page — and when those
    words are about the past and the question is not, it answers a
    different question."""
    title_words = set(_content_words(title))
    topic_words = set(_content_words(topic))
    question_words = set(_content_words(question))
    if not topic_words:
        return 0.0
    cover = len(topic_words & title_words) / len(topic_words)
    score = 4.0 * cover
    if re.search(r"\([^)]*\)", title or ""):
        score += 1.5
    extra = title_words - topic_words - question_words
    score -= 0.6 * len(extra)
    if (extra & _HISTORY_WORDS) and not (question_words & _HISTORY_WORDS):
        score -= 2.0
    if (question_words & _HISTORY_WORDS) and (title_words & _HISTORY_WORDS):
        # the question ASKED for the past: the past's page is the answer
        score += 1.8
    if str(title or "").strip().lower().startswith("list of"):
        score -= 1.5
    return score


def _select_title(titles: list[str], topic: str,
                  question: str = "") -> list[str]:
    """The index's hits ordered by how well each answers for the topic
    — best first, so the caller fetches the best article the index
    has, never merely the first hit or its leftovers."""
    scored = []
    for title in titles or []:
        t = str(title or "").strip()
        if not t or _disambiguation_title(t):
            continue
        scored.append((_title_score(t, topic, question), t))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [t for _, t in scored]


def retrieve(topic: str, deep: bool = False,
             question: str = "") -> list[WebResult]:
    """The external evidence for a request: the Wikipedia article that
    best addresses it (structured knowledge), the general web behind
    it — several sources when the ask needs them.

    `question` is the request's own words.  It selects between the
    index's candidate articles (the topic's own sense outranks its
    sub-pages, and a page about the past only wins when the question
    asked for the past) and it frames the general web's query.  A
    procedural ask ("how do I ...") puts the general web first: the
    sources that answer it are guides, not encyclopedic articles.

    `deep` is the TURN'S OWN ASK — a request that wants depth reads the
    article rather than its summary; a short turn keeps the paragraph.
    """
    results: list[WebResult] = []
    wiki = wikipedia_extract(topic) if deep else None
    if wiki is None:
        wiki = wikipedia_summary(topic)
    if wiki is None:
        candidates = wikipedia_search(topic, limit=5)
        if _is_procedural(question):
            refined = _refine_web_query(topic, question)
            if refined and refined.lower() != topic.lower():
                for title in wikipedia_search(refined, limit=4):
                    if title not in candidates:
                        candidates.append(title)
        for title in _select_title(candidates, topic, question):
            wiki = wikipedia_extract(title) if deep \
                else wikipedia_summary(title)
            if wiki:
                break
    web = search_web(_refine_web_query(topic, question),
                     limit=_MAX_RESULTS - (1 if wiki else 0))
    procedural = _is_procedural(question)
    if procedural and web:
        cued = [item for item in web
                if _PROCEDURAL_CUE_RE.search(item.title or "")
                or _PROCEDURAL_CUE_RE.search((item.snippet or "")[:160])]
        if cued:
            web = cued
    if wiki and not procedural:
        results.append(wiki)
    results.extend(web)
    if wiki and procedural:
        results.append(wiki)
    seen: set[str] = set()
    unique: list[WebResult] = []
    for item in results:
        key = (item.url or item.title or "").lower()
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:_MAX_RESULTS]


def clear_cache() -> None:
    _CACHE.clear()
