"""Compose clean answer bodies from raw retrieved web snippets.

Standalone, stdlib-only, deterministic.  This module is deliberately
NOT part of the brain package's import graph: a renderer may call it
from anywhere without creating cycles.  Every public function tolerates
None / wrong types / empty text by returning a neutral empty result
instead of raising.

Pipeline (``compose_body``)::

    clean (strip_source_apparatus -> repair_punctuation)
      -> split_sentences
      -> drop meta (is_meta_sentence, anchored by question terms)
      -> rank_sentences (stable question overlap)
      -> dedup_sentences (token-set Jaccard, keep the informative one)
      -> condense (clause-boundary length bound)
      -> order (language "ko": Korean first, English support after)
      -> fill until max_chars

Meta-sentence rule (important): a sentence counts as request/process
meta -- not answer content -- only when BOTH hold:

  1. a meta pattern fires (explicit talk of a request / scope / items /
     "the following" / "what follows" / summarizing), and
  2. no content term from the question appears in the sentence
     (see ``is_meta_sentence``).

A bare word such as "explain/설명" is not a meta pattern: topical
sentences may describe things without being about the request.
"""

from __future__ import annotations

import html as _html
import re

__all__ = [
    "repair_punctuation",
    "strip_source_apparatus",
    "split_sentences",
    "normalized_tokens",
    "is_meta_sentence",
    "dedup_sentences",
    "rank_sentences",
    "condense",
    "compose_body",
]

# ---------------------------------------------------------------------------
# constants

_CLOSERS = ")\"]'”’」』"
_STARTER_QUOTES = "\"'“‘「『("
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’",
                "「": "」", "『": "』"}
_QUOTE_CHARS = "".join(_QUOTE_PAIRS) + "”’」』"

_DEDUP_THRESHOLD = 0.75
_CONDENSE_WORDS = 42
_SUPPORT_CAP = 6

_CONTRACT_STEMS = (
    "don", "doesn", "isn", "aren", "wasn", "weren", "didn", "hasn",
    "haven", "hadn", "can", "couldn", "wouldn", "shouldn", "won",
    "ain", "let", "i", "you", "we", "they", "he", "she", "it", "that",
    "there", "what", "who", "here",
)
_CONTRACTIONS = re.compile(
    r"\b(" + "|".join(_CONTRACT_STEMS) + r")\s+(t|m|ve|ll|re|d)\b",
    re.IGNORECASE)
_POSSESSIVE_MID = re.compile(r"(?<=[A-Za-z]{2})[ \t]+s(?=[ \t]+[A-Za-z])")
_POSSESSIVE_END = re.compile(
    r"(?<=[A-Za-z]{2})[ \t]+s(?=[ \t]*[.,;:!?)\]\"']?[ \t]*$)")
# broken spacing inside a number: "5. 513" -> "5.513".  The digit run
# before the dot is limited to 1-3 digits so a sentence break such as
# "In 2020. 500 people ..." is not glued together.
_DECIMAL = re.compile(r"(?<![0-9])(\d{1,3})\.\s+(?=\d)")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?%])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[¿¡])[ \t]+")
_SPACE_BEFORE_CLOSE = re.compile(r"[ \t]+([)\]])")
_TRAIL_ELLIPSIS = re.compile(r"(?:\.{2,}|…)(?:[\"'”’»)\]]|\s)*$")

_CITATION = re.compile(
    r"\[\s*(?:\d{1,3}(?:\s*[-,–]\s*\d{1,3})*"
    r"|citation needed|edit|note\s*\d*|source|ref\s*\d*|n\.?\s*\d*"
    r"|출처|편집|각주|위키\s*편집)\s*\]",
    re.IGNORECASE)

_NAV_LINE = frozenset((
    "menu", "home", "main menu", "navigation", "search", "skip to content",
    "skip to main content", "site navigation", "breadcrumb", "advertisement",
    "ad", "sign in", "log in", "subscribe", "share", "tweet", "close",
    "toggle navigation", "open menu", "cookie policy", "privacy policy",
))

_ABBREV = frozenset((
    "dr", "mr", "mrs", "ms", "prof", "st", "sr", "jr", "vs", "etc",
    "fig", "no", "al", "dept", "est", "inc", "ltd", "gov", "sgt",
    "capt", "lt", "col", "gen", "rev", "hon", "pres", "approx",
))

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_HANGUL_RUN = re.compile(r"[\uac00-\ud7a3]+")
_HANGUL_CHAR = re.compile(r"[\uac00-\ud7a3]")

_META_KO = re.compile(
    r"(?:요청|범위|항목|정리해|정리해서|정리하여|다뤘|다루었|이어서\s*보면"
    r"|위\s*(?:내용|항목)|앞서|다음과\s*같이|아래에서|설명드립니다"
    r"|설명해\s*드립니다|답변|질문에서|질문의)")
_META_EN = re.compile(
    r"(?:\bthe\s+request\b|\bas\s+requested\b|\bthe\s+question\b"
    r"|\brequested\s+(?:scope|items?|parts?)\b"
    r"|\bscope\s+of\s+the\s+request\b"
    r"|\b(?:items?|parts?)\s+above\b"
    r"|\bwhat\s+follows\b|\bthe\s+following\b"
    r"|\bthis\s+(?:answer|reply|response|section)\b"
    r"|\bsummariz\w*|\boutline[sd]?\b"
    r"|\bwe\s+will\s+cover\b|\bcovered\s+above\b|\bas\s+outlined\b)",
    re.IGNORECASE)

_EN_STOP = frozenset((
    "the", "and", "for", "are", "but", "not", "you", "all", "any",
    "can", "had", "her", "was", "one", "our", "out", "who", "how",
    "its", "let", "may", "she", "too", "use", "via", "what", "when",
    "where", "which", "while", "why", "with", "would", "about",
    "after", "again", "against", "also", "among", "because", "been",
    "before", "being", "below", "between", "both", "cannot", "could",
    "does", "doing", "done", "down", "during", "each", "few", "from",
    "further", "have", "having", "here", "hers", "herself", "him",
    "himself", "his", "into", "itself", "just", "more", "most",
    "much", "must", "myself", "nor", "off", "once", "only", "other",
    "ought", "ours", "ourselves", "over", "own", "same", "should",
    "some", "such", "than", "that", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "under", "until", "very", "were", "will", "your",
    "yours", "yourself", "yourselves", "ask", "asked", "asks",
    "answer", "answers", "tell", "tells", "told", "explain",
    "explains", "explained", "describe", "describes", "described",
    "summarize", "summary", "overview", "information", "info",
    "detail", "details", "please", "help", "helps", "give", "gives",
    "given", "show", "shows", "shown", "find", "finds", "found",
    "know", "knows", "want", "wants", "need", "needs", "make",
    "makes", "made", "take", "takes", "took", "get", "gets", "got",
    "see", "sees", "seen", "say", "says", "said", "go", "goes",
    "come", "comes", "look", "looks", "way", "thing", "things",
    "part", "parts", "stuff", "work", "works", "working", "question",
    "questions", "request", "requests", "source", "sources",
    "content", "section", "sections", "above", "below", "following",
    "follows", "mention", "mentioned", "discuss", "discussed",
    "regarding", "research", "main", "many",
))

_KO_STOP = frozenset((
    "대한", "위한", "통해", "에서", "으로", "로써", "라고", "하고",
    "하는", "되는", "되어", "있습니다", "있었다", "합니다", "했습니다",
    "입니다", "이며", "그리고", "하지만", "그러나", "또한", "때문",
    "경우", "정도", "부분", "내용", "여러", "모든", "각각", "다시",
    "매우", "아주", "정말", "너무", "그냥", "이제", "여기", "거기",
    "저기", "우리", "저는", "제가", "그것", "이것", "저것", "무엇",
    "어떤", "어떻게", "왜냐", "같은", "다른", "많은", "위해", "관해",
    "대해", "설명", "요청", "질문", "답변", "정리",
))


# ---------------------------------------------------------------------------
# small helpers

def _as_text(value) -> str:
    """Best-effort text for a str / None / iterable-of-terms argument."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(_as_text(v) for v in value)
    return ""


def _is_hangul(ch: str) -> bool:
    return "\uac00" <= ch <= "\ud7a3"


def _stem(word: str) -> str:
    """Very light English stemmer, applied to both sides of a match."""
    if len(word) > 4:
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith(("ing", "ed")):
            return word[:-3]
        if word.endswith("es") and not word.endswith("ses"):
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
    return word


def _strip_stray_quotes(text: str) -> str:
    t = text.strip()
    while t and t[-1] in _QUOTE_CHARS:
        t = t[:-1].strip()
    if t and t[0] in _QUOTE_CHARS:
        q = t[0]
        partner = _QUOTE_PAIRS.get(q, q)
        if q == partner:
            balanced = t.count(q) % 2 == 0
        else:
            balanced = t.count(partner) >= 1
        if not balanced:
            t = t[1:].strip()
    return t


def _ensure_end(text: str) -> str:
    t = text.rstrip()
    if not t:
        return ""
    if t[-1] in ".!?…":
        return t
    if t[-1] in "\"'”’)]":
        i = len(t) - 1
        while i >= 0 and t[i] in "\"'”’)]":
            i -= 1
        if i >= 0 and t[i] in ".!?…":
            return t
    return t + "."


def _query_terms(question) -> tuple[set[str], set[str]]:
    """(stemmed English terms, Korean terms) that a question is about."""
    text = _as_text(question)
    en: set[str] = set()
    ko: set[str] = set()
    if not text:
        return en, ko
    for w in _WORD.findall(text.lower()):
        if w.isdigit() or len(w) <= 2 or w in _EN_STOP:
            continue
        en.add(_stem(w))
    for run in _HANGUL_RUN.findall(text):
        if len(run) >= 2 and run not in _KO_STOP:
            ko.add(run)
    return en, ko


def _overlap(sentence: str, en_terms: set[str],
             ko_terms: set[str]) -> int:
    """How many question content terms appear in the sentence."""
    if not en_terms and not ko_terms:
        return 0
    score = 0
    if en_terms:
        present = set()
        for w in _WORD.findall(sentence.lower()):
            if len(w) < 2 or w.isdigit():
                continue
            present.add(_stem(w))
        score += len(present & en_terms)
    if ko_terms:
        blob = " ".join(_HANGUL_RUN.findall(sentence))
        score += sum(1 for t in ko_terms if t in blob)
    return score


def _rank_pairs(pairs, question):
    en_terms, ko_terms = _query_terms(question)
    indexed = list(enumerate(pairs))
    indexed.sort(key=lambda t: (-_overlap(t[1][0], en_terms, ko_terms), t[0]))
    return [p for _, p in indexed]


def _dedup_pairs(pairs, threshold=_DEDUP_THRESHOLD):
    kept = []
    kept_toks = []
    removed = 0
    try:
        th = float(threshold)
    except (TypeError, ValueError):
        th = _DEDUP_THRESHOLD
    th = min(max(th, 0.0), 1.0)
    for text, src in pairs:
        s = text.strip()
        if not s:
            continue
        toks = normalized_tokens(s)
        match = -1
        if toks:
            for idx, other in enumerate(kept_toks):
                if not other:
                    continue
                inter = len(toks & other)
                if not inter:
                    continue
                union = len(toks | other)
                if union and inter / union >= th:
                    match = idx
                    break
        if match >= 0:
            removed += 1
            # keep the FIRST wording unless the later one is clearly
            # more informative (length differs by more than 1.6x).
            if len(s) > 1.6 * len(kept[match][0]):
                kept[match] = (s, src)
                kept_toks[match] = toks
            continue
        kept.append((s, src))
        kept_toks.append(toks)
    return kept, removed


# ---------------------------------------------------------------------------
# public API

def repair_punctuation(text: str) -> str:
    """Fix common web-extraction damage without ever inventing words.

    - "word s" -> "word's" (lowercase s only, after a 2+ letter word,
      mid-sentence or sentence-final; initials like "U.S." stay intact)
    - common split contractions: "don t" -> "don't", "I m" -> "I'm"
    - broken number spacing: "5. 513" -> "5.513" (digit.digit only,
      max 3 digits before the dot so "In 2020. 500 ..." is untouched)
    - collapse spaces around punctuation and brackets, normalize all
      whitespace, drop a trailing "..."/"…" and stray end quotes
    """
    if not isinstance(text, str) or not text:
        return ""
    s = text.replace("\u00a0", " ")
    s = _CONTRACTIONS.sub(r"\1'\2", s)
    s = _POSSESSIVE_MID.sub("'s", s)
    s = _POSSESSIVE_END.sub("'s", s)
    s = _DECIMAL.sub(r"\1.", s)
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = _DECIMAL.sub(r"\1.", s)
    s = _SPACE_AFTER_OPEN.sub(r"\1", s)
    s = _SPACE_BEFORE_CLOSE.sub(r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _TRAIL_ELLIPSIS.sub("", s).strip()
    s = _strip_stray_quotes(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_source_apparatus(text: str) -> str:
    """Remove citation markers, nav chrome, and decode HTML entities.

    Citation junk: ``[1]``, ``[1,2]``, ``[citation needed]``, ``[note 2]``,
    ``[출처]``, ``[편집]`` etc.  Entities are decoded with stdlib
    ``html.unescape``.

    Site-nav lines (conservative, documented):
    - a line that is exactly a known nav phrase ("Menu", "Home", "Search",
      "Skip to content", ...), dropped anywhere it appears as a whole line;
    - a short ALL-CAPS phrase (2-5 words, <= 40 chars, no terminal
      punctuation, e.g. "WORLD NEWS") dropped only while it is still in
      the LEADING region, before the first real content line.
    Mixed-case content ("Go Full Throttle Set") and single-word caps
    ("NASA") are never dropped by the ALL-CAPS rule.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = _html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = _CITATION.sub(" ", text)
    kept = []
    leading = True
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.lower().strip(" :·|") in _NAV_LINE:
            continue
        if leading and _is_caps_nav(s):
            continue
        leading = False
        kept.append(s)
    return " ".join(kept).strip()


def _is_caps_nav(line: str) -> bool:
    if line.endswith((".", "!", "?", "…")):
        return False
    words = line.split()
    if not 2 <= len(words) <= 5 or len(line) > 40:
        return False
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def split_sentences(text: str) -> list[str]:
    """Robust EN+KO sentence splitter.

    Handles ". ", "다. ", "다!", ellipses ("..."/"…"), and the missing
    space after a period when the next word starts a sentence
    ("works.It converts" -> two sentences).  Does not split decimals
    ("5.513"), version-like numbers with broken spacing ("5. 513"),
    initials ("U.S."), dotted abbreviations ("Ph.D."), or a known
    abbreviation list ("Dr.", "e.g.", "etc.").
    """
    if not isinstance(text, str):
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out = []
    start = 0
    n = len(text)
    for m in re.finditer(r"[.!?…]+", text):
        mark = m.group()
        end = m.end()
        prev = text[m.start() - 1] if m.start() > 0 else ""
        if mark == "." and prev.isdigit():
            j = m.start() - 1
            while j >= 0 and text[j].isdigit():
                j -= 1
            digits_before = m.start() - 1 - j
            k0 = end
            while k0 < n and text[k0].isspace():
                k0 += 1
            if k0 < n and text[k0].isdigit() and digits_before <= 3:
                continue
        if mark == "." and prev.isalpha():
            j = m.start() - 1
            while j >= 0 and text[j].isalpha():
                j -= 1
            word = text[j + 1:m.start()]
            if len(word) == 1 or word.lower() in _ABBREV:
                continue
        cut = end
        while cut < n and text[cut] in _CLOSERS:
            cut += 1
        k = cut
        while k < n and text[k].isspace():
            k += 1
        spaced = k > cut
        if k >= n or spaced:
            boundary = True
        else:
            nxt = text[k]
            boundary = (nxt.isupper() or _is_hangul(nxt)
                        or nxt.isdigit() or nxt in _STARTER_QUOTES)
            if boundary and not spaced and prev.isupper():
                boundary = False
            if (boundary and nxt.isupper() and k + 1 < n
                    and text[k + 1] == "."):
                boundary = False  # "Ph.D." style internal dot
        if boundary:
            seg = text[start:cut].strip()
            if seg:
                out.append(seg)
            start = k
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def normalized_tokens(sentence: str) -> set[str]:
    """Lowercase word set for EN plus 2-char Hangul shingles for KO.

    English/ASCII words of 2+ characters are kept as-is (lowercased,
    optional internal apostrophe).  Korean runs contribute every
    2-character shingle ("회로가" -> {"회로", "로가"}); a single Hangul
    character is kept whole.  Used for token-set Jaccard similarity.
    """
    toks: set[str] = set()
    if not isinstance(sentence, str) or not sentence:
        return toks
    for w in _WORD.findall(sentence.lower()):
        if len(w) >= 2:
            toks.add(w)
    for run in _HANGUL_RUN.findall(sentence):
        if len(run) == 1:
            toks.add(run)
        else:
            for i in range(len(run) - 1):
                toks.add(run[i:i + 2])
    return toks


def is_meta_sentence(sentence: str, question_terms=()) -> bool:
    """True only for request/process meta, never for topical content.

    Rule (BOTH conditions required):

    1. a meta pattern fires: explicit talk of a request / scope / items /
       "위 항목", "정리해서 설명드립니다", "the request", "the following",
       "what follows", summarizing/outlining the answer itself; and
    2. NO content term extracted from ``question_terms`` (a question
       string or an iterable of terms; stopwords and 1-2 char words are
       ignored) appears in the sentence -- no topical anchor.

    So "'X'의 요청 범위를 정리해서 설명드립니다" is meta when the question
    is about transistors, but "이 회로는 트랜지스터를 설명합니다" is not,
    even though it contains "설명": the anchor "회로" is present and
    "설명" alone is not a meta pattern.  With no question terms at all,
    any sentence that matches a meta pattern is treated as meta.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return False
    text = sentence.strip()
    if not (_META_KO.search(text) or _META_EN.search(text)):
        return False
    en_terms, ko_terms = _query_terms(question_terms)
    return _overlap(text, en_terms, ko_terms) == 0


def dedup_sentences(sentences, threshold: float = 0.75):
    """Drop near-duplicates by token-set Jaccard; keep the best wording.

    Returns ``(kept, removed_count)``.  The FIRST wording is kept, except
    when a later duplicate is more than 1.6x longer (by character
    length): then the longer, more informative one replaces it at the
    earlier position.  Jaccard ``>= threshold`` is a duplicate.
    """
    pairs = []
    for s in sentences or []:
        if isinstance(s, str) and s.strip():
            pairs.append((s.strip(), None))
    kept, removed = _dedup_pairs(pairs, threshold)
    return [s for s, _ in kept], removed


def rank_sentences(sentences, question) -> list[str]:
    """Stable best-first ordering by question content overlap.

    A question term is an English word longer than 2 characters (lightly
    stemmed, stopwords ignored) or a Hangul word; a sentence scores one
    point per distinct term it contains.  Ties keep the input order.
    """
    if isinstance(sentences, str):
        return [sentences.strip()] if sentences.strip() else []
    pairs = []
    for s in sentences or []:
        if isinstance(s, str) and s.strip():
            pairs.append((s.strip(), None))
    return [s for s, _ in _rank_pairs(pairs, question)]


def condense(sentence: str, max_words: int = 42) -> str:
    """Bound a sentence, cutting at a clause boundary near the limit.

    Clause boundaries are ", ", ";" or "—".  The last boundary whose
    prefix is within ``max_words`` and at least 8 words wins (never cut
    below 8 words); otherwise a hard cut at the clamped limit (>= 8) is
    used.  The result always ends with sentence-ending punctuation.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return ""
    text = sentence.strip()
    try:
        limit = max(8, int(max_words))
    except (TypeError, ValueError):
        limit = _CONDENSE_WORDS
    if len(text.split()) <= limit:
        return _ensure_end(text)
    best = -1
    for m in re.finditer(r",\s|;|—", text):
        pos = m.start()
        if 8 <= len(text[:pos].split()) <= limit:
            best = max(best, pos)
    if best > 0:
        cut = text[:best].rstrip(" ,;:—-")
    else:
        cut = " ".join(text.split()[:limit]).rstrip(" ,;:—-")
    return _ensure_end(cut)


def compose_body(items, question, language: str = "ko",
                 max_chars: int = 1800,
                 max_words: int = _CONDENSE_WORDS) -> dict:
    """Turn raw ``{"text", "source"}`` snippets into an answer body.

    Items are cleaned (``strip_source_apparatus`` then
    ``repair_punctuation``), split, meta sentences are dropped, the rest
    ranked by question overlap, deduplicated, condensed, and ordered:
    for ``language == "ko"`` Korean sentences come first while on-topic
    non-Korean sentences follow as support (capped at 6, never dropped
    silently -- the cap shows up as ``truncated``).  Sentences are added
    greedily until ``max_chars``; leftovers also set ``truncated``.

    Returns ``{"body", "kept", "dropped_meta", "dropped_dupes",
    "truncated"}`` where ``kept`` is a list of ``{"text", "source"}``.
    Never raises: empty/None/weird input yields an empty body.
    """
    if items is None:
        item_list = []
    elif isinstance(items, (dict, str)):
        item_list = [items]
    else:
        try:
            item_list = list(items)
        except TypeError:
            item_list = []

    collected = []
    dropped_meta = 0
    for it in item_list:
        if isinstance(it, dict):
            raw = it.get("text")
            src = it.get("source")
        elif isinstance(it, str):
            raw = it
            src = None
        else:
            continue
        if not isinstance(raw, str):
            continue
        src = "" if src is None else str(src)
        cleaned = repair_punctuation(strip_source_apparatus(raw))
        for s in split_sentences(cleaned):
            if is_meta_sentence(s, question):
                dropped_meta += 1
                continue
            collected.append((s, src))

    ranked = _rank_pairs(collected, question)
    kept_pairs, dropped_dupes = _dedup_pairs(ranked, _DEDUP_THRESHOLD)
    condensed = []
    for s, src in kept_pairs:
        c = condense(s, max_words)
        if c:
            condensed.append((c, src))

    capped = False
    if str(language).lower() == "ko":
        primary = [p for p in condensed if _HANGUL_CHAR.search(p[0])]
        support = [p for p in condensed if not _HANGUL_CHAR.search(p[0])]
        capped = len(support) > _SUPPORT_CAP
        ordered = primary + support[:_SUPPORT_CAP]
    else:
        ordered = condensed

    try:
        budget = int(max_chars)
    except (TypeError, ValueError):
        budget = 1800
    if budget < 0:
        budget = 0

    parts = []
    kept = []
    used = 0
    skipped = 0
    for text, src in ordered:
        need = len(text) + (1 if parts else 0)
        if used + need > budget:
            skipped += 1
            continue
        parts.append(text)
        kept.append({"text": text, "source": src})
        used += need

    return {
        "body": " ".join(parts),
        "kept": kept,
        "dropped_meta": dropped_meta,
        "dropped_dupes": dropped_dupes,
        "truncated": bool(skipped or capped),
    }
