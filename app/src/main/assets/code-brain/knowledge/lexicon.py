"""분포 통계 어휘 사전 — 신경망 없이 단어 벡터를 만든다.

하는 일은 세 가지뿐이다::

    1. 동시출현 세기   (window 안에서 같이 나온 횟수)
    2. PPMI 가중       (우연보다 얼마나 더 붙어 다니는가)
    3. SVD 압축        (numpy, 외부 의존 없음)

미분도 역전파도 없다.  같은 데이터면 같은 결과가 나오는 결정적 절차다.
코퍼스는 로컬 문서·코드와, 선택적으로 한국어 위키 덤프(bz2 XML)를 쓴다.

사용::

    from knowledge import lexicon
    lex = lexicon.load()                  # 저장된 사전
    lex.neighbors("리눅스", k=8)
    lex.similarity("커밋", "commit")

    # 새로 만들기 (수 분 소요)
    python3 -m knowledge.lexicon --wiki-dump kowiki.xml.bz2 --out lexicon.npz
"""
from __future__ import annotations

import bz2
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np

TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z][A-Za-z_0-9]{1,}")

# 한국어 기능어/영어 불용어 — 뜻이 아니라 문법만 담당하는 말
STOPWORDS = set("""
the a an of to in is are was were and or for with on at by from as be been this
that these those it its we you they he she i not no do does did can could should
would will there here what which who whom whose when where why how all any both
each few more most other some such only own same so than too very just also
if else then return def class import from self none true false and not or pass
값 것 수 등 및 또 즉 이 그 저 때 곳 바 및 위해 통해 대한 위한 같은 다른 이런 그러한
있다 없다 한다 하다 되다 이다 아니다 그리고 하지만 그러나 또한 그래서 때문에
""".split())

WIKI_MARKUP = re.compile(
    r"\{\{[^{}]*\}\}|\[\[(?:[^\]|]*\|)?([^\]]*)\]\]|<[^>]+>|"
    r"\[https?://[^\s\]]+\s*([^\]]*)\]|'{2,}|&[a-z]+;",
    re.IGNORECASE | re.DOTALL)


@dataclass
class DistributionalLexicon:
    words: list[str]
    vectors: np.ndarray
    meta: dict

    # ---- 조회 ----
    def __post_init__(self) -> None:
        self._index = {word: i for i, word in enumerate(self.words)}
        norms = np.linalg.norm(self.vectors, axis=1)
        self._norms = np.where(norms < 1e-9, 1.0, norms)

    def __contains__(self, word: str) -> bool:
        return str(word).lower() in self._index

    def vector(self, word: str) -> Optional[np.ndarray]:
        index = self._index.get(str(word).lower())
        return None if index is None else self.vectors[index]

    def similarity(self, left: str, right: str) -> float:
        a, b = self.vector(left), self.vector(right)
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def neighbors(self, word: str, k: int = 8
                  ) -> list[tuple[str, float]]:
        index = self._index.get(str(word).lower())
        if index is None:
            return []
        vec = self.vectors[index]
        sims = (self.vectors @ vec) / (self._norms * self._norms[index])
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for i in order:
            if i == index:
                continue
            out.append((self.words[i], round(float(sims[i]), 4)))
            if len(out) >= k:
                break
        return out

    def synonyms(self, word: str, k: int = 5, threshold: float = 0.55
                 ) -> list[str]:
        return [w for w, score in self.neighbors(word, k=k)
                if score >= threshold]

    def related_map(self, seeds: Iterable[str], k: int = 5,
                    threshold: float = 0.6) -> dict[str, list[str]]:
        """여러 씨앗 단어에 대해 유사어 목록을 만든다."""
        out: dict[str, list[str]] = {}
        for seed in seeds:
            found = self.synonyms(seed, k=k, threshold=threshold)
            if found:
                out[str(seed)] = found
        return out

    # ---- 저장/로드 ----
    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            path, vectors=self.vectors.astype(np.float32),
            words=np.array(self.words, dtype=object),
            meta=np.array([json.dumps(self.meta, ensure_ascii=False)]))
        return path

    @classmethod
    def load(cls, path: str) -> "DistributionalLexicon":
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"][0]))
        return cls(words=[str(w) for w in data["words"]],
                   vectors=data["vectors"].astype(np.float32), meta=meta)


# ---------------------------------------------------------------------------
# 코퍼스 읽기
# ---------------------------------------------------------------------------

def clean_wiki(text: str) -> str:
    return WIKI_MARKUP.sub(lambda m: m.group(1) or m.group(2) or " ", text)


def tokens_of(text: str) -> list[str]:
    out: list[str] = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        if lowered in STOPWORDS or len(lowered) < 2:
            continue
        out.append(lowered)
    return out


def iter_local_texts(root: str, max_files: int = 2000,
                     max_bytes: int = 400_000) -> Iterator[str]:
    exts = (".py", ".md", ".txt", ".rst")
    seen = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in {"__pycache__", ".git", "brain_data"}]
        for name in files:
            if not name.endswith(exts):
                continue
            path = os.path.join(base, name)
            try:
                if os.path.getsize(path) > max_bytes:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    yield fh.read()
            except OSError:
                continue
            seen += 1
            if seen >= max_files:
                return


def iter_wiki_texts(path: str, max_pages: int = 0,
                    min_chars: int = 400) -> Iterator[str]:
    """bz2 위키 덤프에서 문서 본문만 스트리밍으로 뽑는다."""
    import xml.etree.ElementTree as ET

    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rb") as handle:
        title = ""
        for event, elem in ET.iterparse(handle, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = elem.text or ""
            elif tag == "text":
                body = elem.text or ""
                elem.clear()
                if title.startswith(("위키백과:", "틀:", "분류:", "파일:",
                                     "사용자:", "포털:")):
                    title = ""
                    continue
                title = ""
                if len(body) < min_chars:
                    continue
                yield clean_wiki(body)
                max_pages -= 1
                if max_pages == 0:
                    return
            if tag == "page":
                elem.clear()


# ---------------------------------------------------------------------------
# 만들기
# ---------------------------------------------------------------------------

def _count_unigrams(texts: Iterable[str], max_tokens: int
                    ) -> tuple[Counter, int]:
    freq: Counter = Counter()
    total = 0
    for text in texts:
        for token in tokens_of(text):
            freq[token] += 1
            total += 1
            if total >= max_tokens:
                return freq, total
    return freq, total


def _count_pairs(texts: Iterable[str], vocab: dict[str, int], window: int,
                 max_tokens: int) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    total = 0
    for text in texts:
        ids = [vocab[t] for t in tokens_of(text) if t in vocab]
        for i, a in enumerate(ids):
            total += 1
            for j in range(i + 1, min(i + 1 + window, len(ids))):
                b = ids[j]
                key = (a, b) if a < b else (b, a)
                pairs[key] += 1
        if total >= max_tokens:
            break
    # 한 번만 나온 쌍은 잡음이므로 버린다 (메모리도 줄어든다)
    return {k: v for k, v in pairs.items() if v >= 2}


def build(roots: Iterable[str] = (), wiki_dump: str = "",
          vocab_size: int = 6000, min_count: int = 15, window: int = 5,
          dim: int = 200, max_tokens: int = 40_000_000,
          max_pages: int = 200_000, max_files: int = 2000,
          progress=None) -> DistributionalLexicon:
    """코퍼스에서 분포 통계 어휘 사전을 만든다 (수 분 소요)."""

    def texts() -> Iterator[str]:
        for root in roots:
            yield from iter_local_texts(root, max_files=max_files)
        if wiki_dump:
            yield from iter_wiki_texts(wiki_dump, max_pages=max_pages)

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    say("1/3 단어 빈도 계산…")
    freq, total = _count_unigrams(texts(), max_tokens)
    say(f"    토큰 {total:,}개, 서로 다른 단어 {len(freq):,}개")

    vocab = {word: i for i, (word, count) in
             enumerate(freq.most_common(vocab_size)) if count >= min_count}
    say(f"    어휘 {len(vocab):,}개 선택")

    say("2/3 동시출현 계산…")
    pairs = _count_pairs(texts(), vocab, window, max_tokens)
    say(f"    쌍 {len(pairs):,}개")

    size = len(vocab)
    matrix = np.zeros((size, size), dtype=np.float32)
    for (a, b), count in pairs.items():
        matrix[a, b] = count
        matrix[b, a] = count

    say("3/3 PPMI + SVD…")
    row = matrix.sum(axis=1) + 1e-9
    col = matrix.sum(axis=0) + 1e-9
    grand = float(matrix.sum()) + 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((matrix * grand) / (row[:, None] * col[None, :]))
    ppmi = np.maximum(pmi, 0.0)

    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    dim = min(dim, u.shape[1])
    vectors = (u[:, :dim] * s[:dim]).astype(np.float32)
    words = [""] * size
    for word, index in vocab.items():
        words[index] = word

    meta = {"tokens": int(total), "vocab": size, "dim": int(dim),
            "window": int(window), "min_count": int(min_count),
            "sources": [os.path.basename(str(r)) for r in roots],
            "wiki": os.path.basename(wiki_dump) if wiki_dump else ""}
    say(f"    완료: {size:,}단어 × {dim}차원")
    return DistributionalLexicon(words=words, vectors=vectors, meta=meta)


# ---------------------------------------------------------------------------
# 저장된 사전 접근 (없으면 None)
# ---------------------------------------------------------------------------

_DEFAULT_PATHS = (
    os.path.join(os.path.expanduser("~"), ".rena_lexicon.npz"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "lexicon.npz"),
)
_CACHE: Optional[DistributionalLexicon] = None
_LOADED = False


def load(path: str = "") -> Optional[DistributionalLexicon]:
    """저장된 사전을 읽는다.  없으면 None (런타임은 조용히 건너뛴다)."""
    global _CACHE, _LOADED
    if path:
        try:
            _CACHE = DistributionalLexicon.load(path)
            _LOADED = True
            return _CACHE
        except Exception:
            return None
    if _LOADED:
        return _CACHE
    _LOADED = True
    for candidate in _DEFAULT_PATHS:
        if os.path.exists(candidate):
            try:
                _CACHE = DistributionalLexicon.load(candidate)
                return _CACHE
            except Exception:
                continue
    return None


def synonym_pairs(threshold: float = 0.65, k: int = 4) -> dict[str, list[str]]:
    """저장된 사전으로 유사어 묶음을 만든다 (없으면 빈 dict)."""
    lex = load()
    if lex is None:
        return {}
    out: dict[str, list[str]] = {}
    for word in lex.words:
        found = lex.synonyms(word, k=k, threshold=threshold)
        if found:
            out[word] = found
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="분포 통계 어휘 사전 만들기")
    parser.add_argument("--root", action="append", default=[],
                        help="로컬 코퍼스 루트 (여러 번 가능)")
    parser.add_argument("--wiki-dump", default="", help="kowiki ... .xml.bz2")
    parser.add_argument("--out", default=os.path.expanduser("~/.rena_lexicon.npz"))
    parser.add_argument("--vocab", type=int, default=6000)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--min-count", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=40_000_000)
    parser.add_argument("--max-pages", type=int, default=200_000)
    args = parser.parse_args(argv)

    roots = args.root or [os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))]
    lex = build(roots=roots, wiki_dump=args.wiki_dump, vocab_size=args.vocab,
                dim=args.dim, min_count=args.min_count,
                max_tokens=args.max_tokens, max_pages=args.max_pages,
                progress=lambda m: print(m, flush=True))
    lex.save(args.out)
    print(f"저장: {args.out}")
    for probe in ("리눅스", "커밋", "정렬", "python"):
        found = lex.neighbors(probe, k=6)
        if found:
            print(f"{probe}: " + ", ".join(f"{w}({s})" for w, s in found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
