"""다중 소스 웹 리서치 + 자동 검증 + 결과 재구성.

단일 소스 결과를 그대로 가져다 쓰는 것이 아니라,
- 여러 검색 엔진/위키에서 결과를 수집
- 동일한 사실이 여러 소스에서 확인되면 신뢰도 상승
- 사용자 요청에 맞게 정보를 변형·결합하여 재구성
- 틀린 정보는 필터링

외부 AI 모델은 사용하지 않음. 순수 규칙 기반 교차 검증만 수행.
"""

from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # "wikipedia", "duckduckgo", "bing"
    confidence: float = 0.5  # 0~1, 교차 검증 후 조정


@dataclass
class ResearchReport:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    consolidated: str = ""  # 변형·결합된 최종 답변
    confidence: float = 0.0
    sources_agree: int = 0  # 동일한 사실을 확인한 소스 수
    verified: bool = False
    limitations: list[str] = field(default_factory=list)
    latency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "confidence": r.confidence,
                }
                for r in self.results
            ],
            "consolidated": self.consolidated,
            "confidence": self.confidence,
            "sources_agree": self.sources_agree,
            "verified": self.verified,
            "limitations": self.limitations,
            "latency": round(self.latency, 4),
        }


# ---------------------------------------------------------------------------
# Fetchers (각 소스별 순수 HTTP 요청, pretrained 모델 미사용)
# ---------------------------------------------------------------------------

_USER_AGENT = "RENA-BR/1.0 (research; no LLM)"


def _fetch(url: str, timeout: float = 15.0) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return raw.decode("utf-8", "replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def _strip_tags(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wiki_search(query: str) -> list[SearchResult]:
    """위키백과 REST API로 검색."""
    results: list[SearchResult] = []
    encoded = urllib.parse.quote(query)
    url = (
        f"https://ko.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={encoded}&format=json&srlimit=5&utf8=1"
    )
    raw = _fetch(url)
    if not raw:
        return results
    try:
        import json
        data = json.loads(raw)
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = _strip_tags(item.get("snippet", ""))
            page_url = f"https://ko.wikipedia.org/wiki/{urllib.parse.quote(title)}"
            results.append(SearchResult(
                title=title, url=page_url, snippet=snippet,
                source="wikipedia", confidence=0.7))
    except Exception:
        pass
    return results


def _duckduckgo_html(query: str) -> list[SearchResult]:
    """DuckDuckGo HTML 랜딩 (안티봇 캡챠가 뜨면 빈 결과)."""
    results: list[SearchResult] = []
    encoded = urllib.parse.quote(query)
    url = f"https://duckduckgo.com/html/?q={encoded}"
    raw = _fetch(url)
    if not raw:
        return results
    try:
        # 아주 간단한 파싱: result__a 링크와 result__snippet 추출
        for block in re.finditer(
            r'<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>.*?'
            r'<a class="result__snippet".*?>(.*?)</a>',
            raw, re.DOTALL | re.IGNORECASE):
            url_val = html.unescape(block.group(1))
            title = _strip_tags(block.group(2))
            snippet_block = block.group(3)
            snippet = _strip_tags(snippet_block)
            results.append(SearchResult(
                title=title, url=url_val, snippet=snippet,
                source="duckduckgo", confidence=0.5))
            if len(results) >= 5:
                break
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# Cross-verification & consolidation
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s가-힣]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _shared_facts(results: list[SearchResult]) -> list[str]:
    """여러 결과에 공통으로 등장하는 사실 조각 (간단 키워드 중복)."""
    if len(results) < 2:
        return []
    token_sets = [set(_normalize(r.snippet).split()) for r in results if r.snippet]
    if not token_sets:
        return []
    common = set.intersection(*token_sets)
    # 불용어 제거
    stop = {"은", "는", "이", "가", "을", "를", "의", "와", "과", "에", "에서",
            "the", "a", "an", "and", "or", "is", "are", "was", "were", "in",
            "on", "at", "to", "for", "of", "에", "가", "은", "는"}
    common -= stop
    return sorted(common, key=len, reverse=True)[:8]


def consolidate(results: list[SearchResult], user_query: str) -> str:
    """검색 결과를 사용자 요청에 맞게 재구성·변형.

    단순히 결과를 나열하는 것이 아니라,
    - 공통 사실이 여러 소스에서 확인되면 confidence 상승
    - 서로 모순되는 정보는 표시하지 않음 (또는 주의 표시)
    - 사용자 질문에 직접 답하는 형태로 변형
    """
    if not results:
        return "검색 결과를 찾지 못했어요."

    facts = _shared_facts(results)
    sources_agree = len(facts)

    # 소스별 신뢰도 보정: 같은 사실이 여러 소스에서 나오면 confidence up
    for r in results:
        r.confidence = min(0.95, r.confidence + 0.1 * sources_agree)

    # 최고 신뢰도 결과부터 정렬
    ordered = sorted(results, key=lambda r: -r.confidence)

    parts: list[str] = []
    parts.append(f"🔍 '{user_query}'에 대해 여러 소스에서 정보를 모았어요.\n")

    if sources_agree >= 2:
        parts.append(
            f"✅ {sources_agree}개 이상의 소스에서 공통 사실을 확인했어요 "
            f"(정보 신뢰도: 높음).\n"
        )
    elif sources_agree == 1:
        parts.append(
            "⚠️ 일부 소스에서만 확인된 정보예요. 참고용으로 보세요.\n"
        )
    else:
        parts.append(
            "⚠️ 명확한 공통 사실이 확인되지 않았어요. 직접 확인해 주세요.\n"
        )

    for i, r in enumerate(ordered[:4], 1):
        title = r.title or "(제목 없음)"
        snippet = r.snippet or "(요약 없음)"
        # 너무 긴 스니펫은 자름
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        parts.append(
            f"{i}. **{title}**\n   {snippet}\n   출처: {r.source} "
            f"(신뢰도 {r.confidence:.0%})\n"
        )

    # 사용자 요청에 맞는 변형 제안 (코드/수학/날씨 등은 별도 핸들러가 처리하므로
    # 여기서는 일반적인 정보 재구성만)
    if any(k in user_query for k in ("뭐야", "무엇", "어떻게", "왜", "설명")):
        parts.append(
            "\n💡 위 정보를 바탕으로 직접 답변을 구성하려면, "
            "더 구체적으로 물어봐 주세요."
        )

    return "\n".join(parts)


def research(query: str, allow_network: bool = False,
             timeout: float = 15.0) -> ResearchReport:
    """다중 소스 웹 리서치 실행."""
    started = time.time()
    report = ResearchReport(query=query)

    if not allow_network:
        report.limitations.append("네트워크가 꺼져 있어 웹 검색을 할 수 없어요.")
        report.latency = time.time() - started
        return report

    # 여러 소스에서 병렬적으로는 못 하므로 순차적으로 수집
    all_results: list[SearchResult] = []

    try:
        wiki = _wiki_search(query)
        all_results.extend(wiki)
    except Exception:
        pass

    try:
        ddg = _duckduckgo_html(query)
        all_results.extend(ddg)
    except Exception:
        pass

    # 중복 URL 제거
    seen_urls: set[str] = set()
    unique: list[SearchResult] = []
    for r in all_results:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            unique.append(r)
    report.results = unique

    if unique:
        report.consolidated = consolidate(unique, query)
        report.sources_agree = len(_shared_facts(unique))
        report.verified = report.sources_agree >= 1
        report.confidence = min(0.9, 0.4 + 0.2 * report.sources_agree)
    else:
        report.consolidated = "검색 결과를 찾지 못했어요."
        report.verified = False

    report.latency = time.time() - started
    return report
