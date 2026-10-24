"""Concept Store - stores and retrieves programming concepts."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Concept:
    id: str
    name: str
    category: str  # "language_feature", "pattern", "api", "error_type", "ui_component"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)


class ConceptStore:
    def __init__(self, data_dir: str = None):
        self.concepts: dict[str, Concept] = {}
        self.data_dir = data_dir
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._load()

    def add(self, concept: Concept) -> None:
        self.concepts[concept.id] = concept
        self._save()

    def get(self, concept_id: str) -> Optional[Concept]:
        return self.concepts.get(concept_id)

    @staticmethod
    def _tokenize_query(query: str) -> set[str]:
        """Extract meaningful words from a query, stripping punctuation and Korean particles.

        Uses general Korean morphological rules for tokenization:
        - Strips common particles (을, 를, 이, 가, etc.)
        - Strips verb/adjective endings (하는, 된, 할, etc.)
        - Splits compound nouns at common noun suffixes (값, 수, 점, etc.)
        This is linguistic preprocessing, NOT concept-tag-dependent.
        """
        import re
        query_lower = query.lower()
        stop_words = {
            "a", "an", "the", "of", "for", "to", "in", "on", "at", "by",
            "with", "from", "and", "or", "but", "if", "then", "else",
            "this", "that", "it", "is", "are", "was", "were", "be",
            "write", "create", "make",
            "implement", "build", "define", "that", "which", "how",
            "find", "get", "compute", "calculat", "determin",
            "return", "give",
            "만들", "작성", "구현",
            "반환",
        }
        # Latin letters are noise at length 1 (single letters carry no
        # word meaning), but a Hangul syllable IS a morpheme — "앱"
        # (app), "웹" (web) are whole words and must survive.
        raw = re.findall(r'[a-z]{2,}|[가-힣]+', query_lower)
        # Strip common Korean particles/suffixes
        _KO_PARTICLES = ("을", "를", "이", "가", "의", "에", "는", "도",
                          "으로", "로", "만", "부터", "까지", "보다", "한",
                          "서", "서는", "هما", "이면서", "이면", "인",
                          "해", "해줘", "구해줘", "보여줘", "골라서", "뽑아서")
        # Korean verb/adjective endings to strip for morphological normalization.
        # These are general Korean grammar rules, NOT concept-tag-dependent.
        # Order matters: longer endings first to avoid partial matches.
        _KO_ENDINGS = ("하는", "한", "된", "할", "합", "해서", "하여",
                        "되고", "되면", "됨", "될", "되",
                        "으며", "면서",
                        "하며", "하고", "하면", "하지", "하니", "하라",
                        "하는", "하", "게", "었", "었으")
        # Common Korean noun suffixes for compound splitting
        _KO_NOUN_SUFFIXES = ("값", "수", "점", "률", "율", "력", "량",
                              "형", "종", "류", "건", "장", "개", "번",
                              "선", "도", "속", "지", "법", "차")
        result = set()
        for w in raw:
            if w in stop_words:
                continue
            # The word's morphology admits two readings, and a suffix
            # list cannot know the word's CLASS: a CASE PARTICLE on a
            # nominal ("더하는" → 더하 + 는) and a VERBAL ENDING on a
            # stem ("크면서" → 크 + 면서).  Committing to one reading
            # drops the stem the other reading recovers — and with it
            # the concept the sentence names — so the tokenizer produces
            # BOTH stems and the matcher scores the text against each.
            readings = []
            # Reading A — nominal first: the case particle, then the
            # endings a particle left behind ("초과하는" → 초과하 → 초과)
            nominal = w
            for p in sorted(_KO_PARTICLES, key=len, reverse=True):
                if nominal.endswith(p) and len(nominal) > len(p):
                    nominal = nominal[:-len(p)]
                    break
            if nominal and len(nominal) >= 3:
                for ending in sorted(_KO_ENDINGS, key=len, reverse=True):
                    if nominal.endswith(ending) and len(nominal) > len(ending):
                        nominal = nominal[:-len(ending)]
                        break
            readings.append(nominal)
            # Reading B — verbal first: the ending the word actually
            # carries, then any particle left on the stem
            verbal = w
            if verbal and len(verbal) >= 3:
                for ending in sorted(_KO_ENDINGS, key=len, reverse=True):
                    if verbal.endswith(ending) and len(verbal) > len(ending):
                        verbal = verbal[:-len(ending)]
                        break
            for p in sorted(_KO_PARTICLES, key=len, reverse=True):
                if verbal.endswith(p) and len(verbal) > len(p):
                    verbal = verbal[:-len(p)]
                    break
            if verbal != nominal:
                readings.append(verbal)
            stripped = nominal

            for reading in readings:
                # Split compound nouns at common suffix boundaries
                compound_split = False
                if reading and len(reading) >= 3:
                    for suffix in _KO_NOUN_SUFFIXES:
                        if reading.endswith(suffix) and len(reading) > len(suffix):
                            stem = reading[:-len(suffix)]
                            if len(stem) >= 2:
                                result.add(stem)
                                result.add(suffix)
                                compound_split = True
                                # Also normalize stem via consonant assimilation.
                                # E.g. stem "최솟" → "최소" (remove assimilated final consonant).
                                # Korean compound nouns like 최소+값 → 최솟값 use
                                # consonant assimilation at the stem boundary.
                                # We try removing the final consonant (any jong≠0).
                                if len(stem) >= 2:
                                    last_code = ord(stem[-1])
                                    if 0xAC00 <= last_code <= 0xD7A3:
                                        jong = (last_code - 0xAC00) % 28
                                        if jong != 0:
                                            normalized = stem[:-1] + chr(last_code - jong)
                                            result.add(normalized)
                            break
                # Also normalize standalone words via consonant assimilation.
                # E.g. "최솟" alone → "최소". Skip if already compound-split
                # (compound split already normalized the stem above).
                if not compound_split and reading and len(reading) >= 3:
                    last_code = ord(reading[-1])
                    if 0xAC00 <= last_code <= 0xD7A3:
                        jong = (last_code - 0xAC00) % 28
                        if jong != 0:
                            normalized = reading[:-1] + chr(last_code - jong)
                            result.add(normalized)

            # English morphological variants: "sorts"→"sort",
            # "calculates"→"calculate", "sorting"→"sort", "sorted"→"sort".
            # English inflection is general grammar (parallel to the
            # Korean morphology above), NOT concept-tag-dependent — the
            # stem joins the token set so an inflected query word matches
            # its stem tags.  Guards: word long enough to strip, no "ss"
            # ending (class/address), doubled-consonant undo (shipping→ship).
            if stripped and re.fullmatch(r'[a-z]+', stripped) \
                    and len(stripped) >= 4 and not stripped.endswith("ss"):
                if stripped.endswith("ies") and len(stripped) > 4:
                    result.add(stripped[:-3] + "y")     # identifies→identify
                elif stripped.endswith(("ches", "shes", "xes", "zes")):
                    result.add(stripped[:-2])           # boxes→box
                elif stripped.endswith("s"):
                    result.add(stripped[:-1])           # sorts→sort, calculates→calculate
                if stripped.endswith("ing") and len(stripped) > 5:
                    stem = stripped[:-3]
                    if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                        stem = stem[:-1]                # shipping→ship
                    result.add(stem)
                if stripped.endswith("ed") and len(stripped) > 4:
                    stem = stripped[:-2]
                    if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                        stem = stem[:-1]                # stopped→stop

            # The length-2 floor is a LATIN rule (a single letter is
            # noise). A Hangul syllable stripped of its particle IS the
            # word — "앱을" → "앱" must survive as "앱", not fall back
            # to the unstripped "앱을" (which matches no tag).
            added = False
            for reading in readings:
                if reading and reading not in stop_words and (
                        len(reading) >= 2 or not reading.isascii()):
                    result.add(reading)
                    added = True
            if not added and w not in stop_words:
                result.add(w)
        return result

    def search(self, query: str, category: str = None, tags: list[str] = None) -> list[Concept]:
        """Search concepts by semantic content.

        Delegates to :meth:`search_with_scores` so that the two entry
        points share ONE scoring rule.  Duplicating the matcher let a
        hint that was fixed in one copy keep firing in the other.
        """
        return [c for c, _ in self.search_with_scores(query, category, tags)]

    def search_with_scores(self, query: str, category: str = None,
                           tags: list[str] = None) -> list[tuple[Concept, float]]:
        """Search concepts and return (concept, score) pairs sorted by score."""
        query_words = self._tokenize_query(query)
        if not query_words:
            return []

        results = []
        for c in self.concepts.values():
            if category and c.category != category:
                continue
            if tags and not any(t in c.tags for t in tags):
                continue

            score = 0

            # Tag matching: check substring relationships and shared stems
            for qw in query_words:
                for tag in c.tags:
                    tag_lower = tag.lower()
                    if qw == tag_lower:
                        # Exact match — strongest signal
                        score += 4.0
                    elif self._same_morpheme(qw, tag_lower):
                        # The SAME word in its other spelling: a script
                        # written without spaces writes one morpheme with
                        # and without a final consonant depending on what
                        # follows it (큰/크, 높은/높), so the two spellings
                        # are one word for matching — as strong as the
                        # exact match, and never stronger.
                        score += 4.0
                    elif self._is_cjk(qw) and " " not in tag_lower \
                            and len(qw) > len(tag_lower) \
                            and qw.startswith(tag_lower) \
                            and len(tag_lower) >= len(qw) * 0.6:
                        # Query word is a compound form of a SINGLE-WORD
                        # concept — a MORPHOLOGICAL rule for scripts that
                        # write without spaces (e.g. "평균값" starts with
                        # "평균").  A Latin query is word-separated: a
                        # prefix of a longer word is NOT a compound form
                        # ("program" is not a stem of "programming").
                        score += 3.5
                    elif len(tag_lower) > len(qw) \
                            and tag_lower.startswith(qw) \
                            and len(qw) >= 3 \
                            and self._word_in(tag_lower, qw):
                        # Tag starts with the query word at a word
                        # boundary.  A MULTI-WORD tag's meaning is the
                        # whole phrase: one shared word (especially a
                        # function word like "only" or "first") is a
                        # weak hint, never a full tag match.  A very
                        # short query word is noise ('hi' starts
                        # 'highest' but is not related) — require ≥3
                        # chars.
                        score += 1.5 if " " not in tag_lower else 0.5
                    elif self._word_in(tag_lower, qw) \
                            or self._word_in(qw, tag_lower):
                        # A WHOLE-WORD match in either direction — a
                        # bare substring ("only" inside "divisible
                        # only") is not a match: it locates the tags but
                        # never scores them.  In a MULTI-WORD tag the
                        # phrase is the meaning: the phrase's HEAD (its
                        # final word — "language" in "c language") is
                        # the category word and scores fully; any other
                        # shared word is a modifier ("only" in "only by
                        # 1") and is only a weak hint.
                        if len(qw) >= 3:
                            if " " not in tag_lower:
                                score += 1.5
                            elif tag_lower.split()[-1] == qw:
                                score += 1.5
                            else:
                                score += 0.5
                    elif len(qw) >= 4 and len(tag_lower) >= 4:
                        # A shared 4-char prefix is a WEAK hint: a query
                        # word's inflectional variants (the tokenizer
                        # emits both forms) must not accumulate one
                        # weak signal into the evidence threshold
                        # ("interesting"/"interpret" share "inte" but
                        # are unrelated) — evidence needs a real signal.
                        if qw[:4] == tag_lower[:4]:
                            score += 0.3
                    if self._are_semantically_similar(qw, tag_lower):
                        score += 1.2

            # Multi-word tag matching: check if the full query matches
            # a multi-word tag as a substring. These are strong signals
            # (e.g. "highest to lowest" clearly indicates sort_descending).
            for tag in c.tags:
                tag_lower = tag.lower()
                if " " in tag_lower and tag_lower in query:
                    score += 5.0

            # Description/Name matching: fuzzy containment needs a
            # meaningful word — a 2-char fragment ('do' in 'windows')
            # is noise, not semantic overlap.
            desc_lower = c.description.lower()
            for qw in query_words:
                if len(qw) >= 3 and qw in desc_lower:
                    score += 0.5

            name_lower = c.name.lower()
            for qw in query_words:
                if len(qw) >= 3 and qw in name_lower:
                    score += 1.5

            if score >= 2.0:
                results.append((c, score))

        results.sort(key=lambda x: -x[1])
        return results

    def _are_semantically_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are semantically similar using common synonym groups."""
        # Common synonym groups for data operations
        synonym_groups = [
            {"maximum", "largest", "biggest", "greatest", "highest", "peak", "top", "most", "max",
             "high"},
            {"minimum", "smallest", "least", "lowest", "bottom", "least", "min",
             "low"},
            {"average", "mean", "typical", "central", "norm"},
            {"sum", "total", "aggregate", "addition", "combined", "adding", "adds", "added"},
            {"filter", "select", "choose", "pick", "keep", "retain", "筛选", "골라"},
            {"sort", "order", "arrange", "rank", "organize", "정렬"},
            {"duplicate", "redundant", "repeated", "copy", "repeat", "중복",
             "eliminate", "remove", "redundancies", "distinct", "unique"},
            {"even", "divisible", "multiple", "modulus", "짝수"},
            {"내림차순", "큰 순서대로", "내림", "역순"},
            {"오름차순", "작은 순서대로", "오름"},
            {"평균", "average", "mean"},
            {"합계", "total", "sum", "총합"},
            {"최대", "가장 큰", "가장 높은", "maximum", "largest"},
            {"최소", "가장 작은", "가장 낮은", "minimum", "smallest"},
            {"reverse", "invert", "flip", "backwards", "mirror", "뒤집"},
            {"prime", "indivisible", "divisible", "divided", "divisor", "prime number", "소수"},
            {"greater", "larger", "higher", "more", "above", "over", "exceed", "이상"},
            {"less", "smaller", "lower", "fewer", "below", "under", "미만"},
            # New groups for expanded concepts
            {"count", "how many", "number of", "total count", "size", "갯수", "몇 개"},
            {"median", "middle", "central value", "50th percentile", "중앙값"},
            {"mode", "most common", "most frequent", "최빈값"},
            {"variance", "spread", "dispersion", "standard deviation", "분산", "표준편차"},
            {"flatten", "unfold", "unnest", "flat", "평탄화", "펼치기"},
            {"group", "group by", "bucket", "categorize", "partition", "그룹화"},
            {"map", "transform", "apply", "convert", "변환", "적용"},
            {"reduce", "fold", "accumulate", "aggregate", "combine", "축약", "누적"},
            {"take", "first", "head", "top", "prefix", "처음", "앞에서"},
            {"drop", "skip", "tail", "offset", "discard", "건너뛰기", "뒤에서"},
        ]
        w1, w2 = word1.lower(), word2.lower()
        for group in synonym_groups:
            if w1 in group and w2 in group:
                return True
        return False

    @staticmethod
    def _ko_stem(token: str) -> str:
        """The token with the final consonant of its last Hangul syllable
        removed.

        A Korean morpheme is spelled with and without its final
        consonant depending on what follows it (큰/크다/크게, 높은/높이),
        so the alternation is orthography, not a different word.  A token
        that does not end in Hangul is returned unchanged — the rule is a
        property of the script, never of the vocabulary.
        """
        if not token:
            return token
        code = ord(token[-1])
        if not 0xAC00 <= code <= 0xD7A3:
            return token
        jong = (code - 0xAC00) % 28
        if jong == 0:
            return token
        return token[:-1] + chr(code - jong)

    @classmethod
    def _same_morpheme(cls, left: str, right: str) -> bool:
        """`left` and `right` are one word in two spellings."""
        if not left or not right:
            return False
        if " " in left or " " in right:
            return False
        if not cls._is_cjk(left) or not cls._is_cjk(right):
            return False
        return cls._ko_stem(left) == cls._ko_stem(right)

    @staticmethod
    def _is_cjk(text: str) -> bool:
        """The text belongs to a script written without word spaces."""
        return any(ord(ch) > 0x2E7F for ch in text)

    @staticmethod
    def _word_in(text: str, word: str) -> bool:
        """`word` occurs in `text` as a WHOLE word.

        For a Latin word the boundary is alphanumeric; for a script
        written without spaces a substring IS the segment, so the plain
        containment applies.  This is what keeps a tag HINT from
        scoring a concept whose tag merely contains the query word.
        """
        if not text or not word:
            return False
        if ConceptStore._is_cjk(word) or ConceptStore._is_cjk(text) and \
                not word.isascii():
            return word in text
        import re as _re
        return bool(_re.search(
            r"(?<![a-z0-9])" + _re.escape(word) + r"(?![a-z0-9])",
            text))

    def update(self, concept_id: str, **kwargs) -> bool:
        c = self.concepts.get(concept_id)
        if not c:
            return False
        for k, v in kwargs.items():
            if hasattr(c, k):
                setattr(c, k, v)
        self._save()
        return True

    def delete(self, concept_id: str) -> bool:
        if concept_id in self.concepts:
            del self.concepts[concept_id]
            self._save()
            return True
        return False

    def list_all(self) -> list[Concept]:
        return list(self.concepts.values())

    def list_by_category(self, category: str) -> list[Concept]:
        return [c for c in self.concepts.values() if c.category == category]

    def _save(self):
        if not self.data_dir:
            return
        path = os.path.join(self.data_dir, "concepts.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump([vars(c) for c in self.concepts.values()], f, indent=2)
        os.replace(tmp, path)

    def _load(self):
        path = os.path.join(self.data_dir, "concepts.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            # A writer was interrupted mid-save (or two processes
            # raced): the store starts empty and the damaged file is
            # kept aside instead of taking the whole engine down.
            try:
                os.replace(path, path + ".corrupt")
            except OSError:
                pass
            return
        if not isinstance(data, list):
            return
        for d in data:
            try:
                c = Concept(**d)
            except (TypeError, KeyError, ValueError):
                continue
            self.concepts[c.id] = c

    def export_json(self) -> str:
        return json.dumps([vars(c) for c in self.concepts.values()], indent=2)

    def import_json(self, json_str: str) -> int:
        data = json.loads(json_str)
        count = 0
        for d in data:
            c = Concept(**d)
            self.concepts[c.id] = c
            count += 1
        self._save()
        return count


def seed_programming_fundamentals(store: ConceptStore) -> None:
    """Seed the fundamental programming knowledge: the languages, the
    language constructs, and the runtime concepts.  The descriptions
    carry the actual explanation content — the conversation knowledge
    engine reasons them into claims (the knowledge → the evidence → the
    reasoning → the response)."""
    fundamentals = [
        # ── Programming languages (the what-is questions' targets) ──
        Concept("lang_python", "Python", "programming_language",
                "A high-level, general-purpose programming language known for "
                "readable syntax, dynamic typing, and a large standard library; "
                "widely used for scripting, data analysis, web backends, and "
                "machine learning",
                ["python", "파이썬", "python3"],
                []),
        Concept("lang_kotlin", "Kotlin", "programming_language",
                "A statically typed language that runs on the JVM; concise "
                "syntax with null safety and coroutines; the recommended "
                "language for Android app development",
                ["kotlin", "코틀린"],
                []),
        Concept("lang_java", "Java", "programming_language",
                "A class-based, object-oriented language that runs on the "
                "JVM; compiled to bytecode; used for enterprise systems, "
                "Android, and large-scale servers",
                ["java", "자바"],
                []),
        Concept("lang_c", "C", "programming_language",
                "A low-level procedural language with manual memory "
                "management and direct hardware access; the foundation of "
                "operating systems and embedded software",
                ["c language", "c언어", "씨언어"],
                []),
        Concept("lang_cpp", "C++", "programming_language",
                "A language extending C with classes, templates, and the STL; "
                "combines low-level control with high-level abstractions; used "
                "for games, systems, and performance-critical software",
                ["c++", "cpp", "씨플러스"],
                []),
        Concept("lang_javascript", "JavaScript", "programming_language",
                "The scripting language of the web browser; dynamically typed; "
                "also used on servers via Node.js; powers interactive web "
                "pages, and TypeScript adds static types",
                ["javascript", "js", "자바스크립트"],
                []),
        Concept("lang_go", "Go", "programming_language",
                "A statically typed language from Google with goroutines "
                "(lightweight concurrency), fast compilation, and simple "
                "deployment; used for cloud services and CLI tools",
                ["golang", "go language", "고언어"],
                []),
        Concept("lang_rust", "Rust", "programming_language",
                "A systems language with ownership-based memory safety without "
                "garbage collection; prevents data races at compile time; "
                "performance comparable to C++",
                ["rust", "러스트"],
                []),
        # ── Language constructs (the fundamental concepts) ──
        Concept("fund_variable", "Variable", "language_fundamental",
                "A named storage location that holds a value; the name refers "
                "to the value and can be reassigned; types may be static "
                "(checked at compile time) or dynamic (checked at runtime)",
                ["variable", "변수"],
                []),
        Concept("fund_function", "Function", "language_fundamental",
                "A named, reusable block of code that takes inputs "
                "(parameters), performs a computation, and returns a result; "
                "functions break programs into composable parts and enable "
                "reuse",
                ["function", "함수"],
                []),
        Concept("fund_loop", "Loop", "language_fundamental",
                "A control structure that repeats a block of code while a "
                "condition holds; for-loops iterate over sequences, while-loops "
                "repeat until a condition changes; loops let programs process "
                "collections of any size with a fixed amount of code",
                ["loop", "반복", "반복문"],
                []),
        Concept("fund_conditional", "Conditional", "language_fundamental",
                "A control structure (if/else, switch) that executes different "
                "code paths depending on a boolean condition; conditionals let "
                "programs make decisions",
                ["conditional", "if", "else", "조건문"],
                []),
        Concept("fund_class", "Class", "language_fundamental",
                "A blueprint that bundles data (fields) with behavior "
                "(methods); instantiating a class creates objects; classes "
                "enable encapsulation and are the basis of object-oriented "
                "programming",
                ["class", "클래스"],
                []),
        Concept("fund_recursion", "Recursion", "language_fundamental",
                "A technique where a function calls itself on a smaller "
                "subproblem until a base case is reached; elegant for trees, "
                "divide-and-conquer, and mathematical definitions; every "
                "recursion needs a base case to terminate",
                ["recursion", "재귀"],
                []),
        # ── Runtime / systems concepts ──
        Concept("fund_pointer", "Pointer", "language_fundamental",
                "A variable that stores a memory address; dereferencing "
                "accesses the value at that address; pointers enable linked "
                "structures and efficient passing, but incorrect use causes "
                "segmentation faults; higher-level languages replace raw "
                "pointers with references",
                ["pointer", "포인터", "reference", "memory address"],
                []),
        Concept("fund_compiler", "Compiler", "language_fundamental",
                "A program that translates source code into machine code or "
                "bytecode before execution; it checks types and syntax, "
                "optimizes the code, and produces a standalone executable; "
                "compiled programs run fast because translation happens once",
                ["compiler", "컴파일러", "compile", "컴파일"],
                []),
        Concept("fund_interpreter", "Interpreter", "language_fundamental",
                "A program that reads source code and executes it directly, "
                "statement by statement; no separate executable is produced; "
                "enables rapid development (REPL) at the cost of runtime "
                "speed; Python and JavaScript are primarily interpreted",
                ["interpreter", "인터프리터", "interpret"],
                []),
        Concept("fund_memory", "Memory", "language_fundamental",
                "The storage a program uses: the stack for call frames and "
                "local variables, the heap for dynamically allocated "
                "objects; languages manage the heap via garbage collection "
                "(Java, Python) or manual allocation (C, C++)",
                ["memory", "메모리", "stack", "heap", "garbage collection"],
                []),
        Concept("fund_algorithm", "Algorithm", "language_fundamental",
                "A finite sequence of well-defined steps that solves a class "
                "of problems; measured by time complexity (how the running "
                "time grows with input) and space complexity (the memory "
                "used); good algorithms are correct and efficient",
                ["algorithm", "알고리즘"],
                []),
        Concept("fund_data_structure", "Data Structure", "language_fundamental",
                "A way of organizing data for efficient access and update; "
                "arrays give indexed access, linked lists give cheap "
                "insertion, hash maps give key lookup, trees give ordered "
                "traversal; choosing the structure shapes the algorithm",
                ["data structure", "자료구조"],
                []),
        # ── AI / computing concepts ──
        Concept("fund_ai", "Artificial Intelligence", "language_fundamental",
                "The field of building systems that perform tasks requiring "
                "intelligence: perception, reasoning, learning, and language "
                "understanding; modern approaches learn patterns from data "
                "(machine learning), and deep learning uses layered neural "
                "networks",
                ["artificial intelligence", "ai", "인공지능", "machine learning",
                 "deep learning"],
                []),
        Concept("fund_average", "Average (Mean)", "language_fundamental",
                "The sum of a collection of numbers divided by how many "
                "there are; a measure of central tendency; the mean is "
                "sensitive to outliers — the median (the middle value when "
                "sorted) is a robust alternative",
                ["average", "mean", "평균", "평균값", "중앙값"],
                []),
        # ── More languages: the what-is questions' targets ──
        Concept("lang_typescript", "TypeScript", "programming_language",
                "A statically typed superset of JavaScript that compiles "
                "to JavaScript; the types are checked before the code "
                "runs, and an existing JavaScript project can adopt them "
                "gradually",
                ["typescript", "ts", "타입스크립트"],
                []),
        Concept("lang_swift", "Swift", "programming_language",
                "A safe, expressive language for Apple platforms (iOS, "
                "macOS); strong static typing with value and reference "
                "types and automatic memory management",
                ["swift", "스위프트"],
                []),
        Concept("lang_dart", "Dart", "programming_language",
                "A client-optimized, strongly typed language with sound "
                "null safety; best known as the language of the Flutter "
                "UI toolkit, and it compiles to native code or JavaScript",
                ["dart", "다트"],
                []),
        Concept("lang_ruby", "Ruby", "programming_language",
                "A dynamically typed, object-everything language built "
                "for developer happiness: blocks, metaprogramming and a "
                "syntax that reads like prose; the language behind Ruby "
                "on Rails",
                ["ruby", "루비"],
                []),
        Concept("lang_php", "PHP", "programming_language",
                "A dynamically typed language designed for server-side "
                "web pages: it runs embedded in HTML templates, handles "
                "requests and databases, and still powers a large share "
                "of the web",
                ["php", "php lang", "피에이치피"],
                []),
        Concept("lang_csharp", "C#", "programming_language",
                "A statically typed, object-oriented language on the "
                ".NET runtime; language-integrated query, async/await, "
                "and a large standard library; used for games (Unity), "
                "services and desktop applications",
                ["csharp", "c sharp", "씨샵"],
                []),
        # ── The value types: WHY a language needs them ──
        Concept("fund_type", "Data Type", "language_fundamental",
                "A declared kind of value — integer, text, list, object "
                "— that tells the language which operations are valid: a "
                "wrong operation is rejected before the program runs "
                "instead of silently producing corrupt data, the "
                "declaration documents intent, and the compiler can pick "
                "a representation. Languages differ in WHEN they check "
                "(static typing at compile time, dynamic typing at "
                "runtime) and how much they infer from context, which is "
                "why the same family of languages needs different type "
                "disciplines",
                ["type", "types", "data type", "data types", "type "
                 "system", "static typing", "dynamic typing",
                 "type checking", "자료형", "타입"],
                []),
        # ── The data structures: the directions a learner picks ──
        Concept("fund_array", "Array", "language_fundamental",
                "An ordered collection stored as one contiguous block: "
                "every element has the same type and is reached by an "
                "index, so reading is constant-time while inserting in "
                "the middle shifts the rest",
                ["array", "arrays", "배열"],
                []),
        Concept("fund_dictionary", "Dictionary", "language_fundamental",
                "A collection of key to value pairs where every key is "
                "unique: lookup, insert and remove are typically "
                "constant-time, so it maps names to values (a hash map, "
                "an associative array, an object)",
                ["dictionary", "hash map", "associative array",
                 "key value", "딕셔너리", "해시맵"],
                []),
        Concept("fund_set", "Set", "language_fundamental",
                "An unordered collection of unique membership: adding, "
                "removing and asking 'is it in there?' are the "
                "operations that matter, and duplicates cannot occur",
                ["set", "sets", "집합"],
                []),
        Concept("fund_stack", "Stack", "language_fundamental",
                "A last-in first-out collection: push adds on top and "
                "pop takes from the top, which is what makes undo, "
                "backtracking and depth-first traversal work",
                ["stack", "lifo", "push", "pop", "스택"],
                []),
        Concept("fund_queue", "Queue", "language_fundamental",
                "A first-in first-out collection: enqueue adds at the "
                "back and dequeue takes from the front, which is what "
                "makes scheduling, buffering and breadth-first "
                "traversal work",
                ["queue", "fifo", "enqueue", "dequeue", "큐"],
                []),
        Concept("fund_tree", "Tree", "language_fundamental",
                "A hierarchy of nodes with one root: each node has one "
                "parent and zero or more children, which is what makes "
                "ordered search, sorting and nested structures "
                "efficient",
                ["tree", "trees", "binary tree", "트리"],
                []),
        Concept("fund_graph", "Graph", "language_fundamental",
                "A set of vertices connected by edges, directed or not: "
                "it models networks, maps and dependencies, and "
                "breadth-first or depth-first traversal walks it "
                "systematically",
                ["graph", "graphs", "vertex", "edge", "그래프"],
                []),
        Concept("fund_big_o", "Big-O Notation", "language_fundamental",
                "How an algorithm's time or space grows with the input "
                "size, ignoring constants and lower terms: O(1) "
                "constant, O(log n) logarithmic, O(n) linear, O(n log n) "
                "linearithmic, O(n^2) quadratic — the basis for "
                "comparing two algorithms",
                ["big o", "big-o", "complexity", "time complexity",
                 "space complexity", "bigo", "시간복잡도"],
                []),
        # ── The paradigms: the directions a learner picks ──
        Concept("fund_oop", "Object-Oriented Programming",
                "language_feature",
                "A paradigm that models a program as objects bundling "
                "state and behavior, organized by encapsulation "
                "(hiding internals), inheritance (reusing a base) and "
                "polymorphism (one interface, many forms)",
                ["object oriented", "object-oriented", "oop",
                 "encapsulation", "inheritance", "polymorphism",
                 "객체지향"],
                []),
        Concept("fund_functional", "Functional Programming",
                "language_feature",
                "A paradigm that composes pure functions over immutable "
                "data: no side effects, so behavior is easier to test "
                "and reason about; higher-order functions and "
                "composition take the place of loops and mutable state",
                ["functional programming", "functional", "pure function",
                 "immutability", "higher order function", "함수형"],
                []),
    ]
    for c in fundamentals:
        # idempotent: a store loaded from disk keeps its own snapshot,
        # so the batch fills in only what that snapshot predates — the
        # knowledge grows without rewriting (and re-saving) what the
        # learner already has.
        if store.get(c.id) is None:
            store.add(c)


def seed_default_concepts(store: ConceptStore) -> None:
    defaults = [
        Concept("func_def", "Function Definition", "language_feature",
                "A named block of reusable code",
                ["function", "method", "procedure", "함수", "함수 정의",
                 "메서드", "프로시저"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"],
                {"python": "def", "kotlin": "fun", "java": "method", "c": "function",
                 "cpp": "function", "javascript": "function", "go": "func", "rust": "fn"}),
        Concept("class_def", "Class Definition", "language_feature",
                "A blueprint for creating objects",
                ["class", "type", "struct", "클래스", "타입", "구조체"],
                ["python", "kotlin", "java", "cpp", "javascript", "rust"],
                {"python": "class", "kotlin": "class", "java": "class", "cpp": "class",
                 "javascript": "class", "rust": "struct"}),
        Concept("artifact_project", "Software Project", "capability",
                "A complete software system with multiple parts — an "
                "application, program, or service built for a domain",
                ["project", "program", "application", "app", "service",
                 "system", "솔루션",
                 "프로젝트", "프로그램", "애플리케이션", "앱", "서비스", "시스템"],
                []),
        Concept("pointer", "Pointer", "language_feature",
                "A reference that holds another value's memory address",
                ["pointer", "포인터", "dangling pointer", "dereference",
                 "pointer arithmetic"],
                ["c", "cpp", "rust", "go"],
                {"c": "T*", "cpp": "T*", "rust": "&T", "go": "*T"}),
        Concept("interface_def", "Interface Definition", "language_feature",
                "A contract that classes can implement", ["interface", "trait", "protocol"],
                ["kotlin", "java", "go", "rust"],
                {"kotlin": "interface", "java": "interface", "go": "interface", "rust": "trait"}),
        Concept("variable_decl", "Variable Declaration", "language_feature",
                "Declaring a named storage location", ["var", "val", "let", "const"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("loop_for", "For Loop", "language_feature",
                "Iteration over a range or collection", ["for", "foreach", "iterate"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("loop_while", "While Loop", "language_feature",
                "Loop that continues while a condition is true", ["while", "loop"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("condition_if", "If-Else", "language_feature",
                "Conditional branching", ["if", "else", "condition", "branch"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("error_handling", "Error Handling", "language_feature",
                "Try-catch-finally blocks for error management", ["try", "catch", "exception", "error"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "rust"]),
        Concept("ui_screen", "UI Screen", "ui_component",
                "A screen or page in the UI", ["screen", "page", "view"],
                ["kotlin", "javascript", "dart"]),
        Concept("ui_button", "UI Button", "ui_component",
                "A clickable button component", ["button", "click"],
                ["kotlin", "javascript", "dart"]),
        Concept("ui_text", "UI Text", "ui_component",
                "A text display component", ["text", "label", "display"],
                ["kotlin", "javascript", "dart"]),
        Concept("sum_pattern", "Sum Pattern", "pattern",
                "Adding two numbers together", ["add", "sum", "plus", "arithmetic"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("factorial_pattern", "Factorial Pattern", "pattern",
                "Computing factorial of a number", ["factorial", "recursive", "math"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("string_concat", "String Concatenation", "pattern",
                "Combining strings", ["concat", "join", "string", "append"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("list_operations", "List Operations", "pattern",
                "Operations on lists/arrays", ["list", "array", "append", "filter", "map"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("null_check", "Null Check", "pattern",
                "Checking for null/nil/None values", ["null", "nil", "None", "optional"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("import_module", "Import Module", "language_feature",
                "Importing external modules or packages", ["import", "require", "use"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("enum_def", "Enum Definition", "language_feature",
                "Enumeration type with named constants", ["enum", "enumeration"],
                ["kotlin", "java", "c", "cpp", "rust"]),
        Concept("switch_when", "Switch/When", "language_feature",
                "Pattern matching or switch statement", ["switch", "when", "match", "case"],
                ["kotlin", "java", "c", "cpp", "javascript", "rust"]),
        Concept("list_comprehension", "List Comprehension", "pattern",
                "Concise list creation with filtering", ["comprehension", "filter", "map"],
                ["python", "javascript"]),

        # Operation concepts — semantic meaning, not keywords
        Concept("add", "Add", "operation",
                "Add two values together",
                ["add", "plus", "더하", "더하기", "더하는", "더해", "덧셈"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("multiply", "Multiply", "operation",
                "Multiply two values together",
                ["multiply", "times", "곱하", "곱하기", "곱셈"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("calculate_average", "Calculate Average", "operation",
                "Compute the arithmetic mean of a collection of numbers",
                ["average", "mean", "central tendency", "typical value",
                 "평균"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("calculate_sum", "Calculate Sum", "operation",
                "Compute the total of all values in a collection",
                ["sum", "total", "addition", "aggregate",
                 "합계", "총합", "합산", "합", "더하기", "구하는"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("find_max", "Find Maximum", "operation",
                "Identify the largest value in a collection",
                ["maximum", "largest", "greatest", "highest", "peak",
                 "가장 큰", "최대", "가장 높은",
                 "제일 큰", "제일 높은", "제일 큰 값"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("find_min", "Find Minimum", "operation",
                "Identify the smallest value in a collection",
                ["minimum", "smallest", "least", "lowest", "bottom",
                 "가장 작은", "최소", "가장 낮은"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("compare_greater_than", "Compare Greater Than", "operation",
                "Determine if one value exceeds another",
                ["greater", "exceed", "surpass", "above", "higher", "over",
                 "높은", "이상", "크게", "초과", "넘는", "큰",
                 "greater than", "more than", "above the", "surpassing",
                 "exceeding"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("compare_less_than", "Compare Less Than", "operation",
                "Determine if one value is below another",
                ["less", "below", "lower", "under", "fewer", "beneath",
                 "낮은", "미만", "작게", "작은", "이하", "작은 값"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("is_prime", "Check Primality", "operation",
                "Determine if a number has no divisors other than 1 and itself",
                ["prime", "divisor", "factor", "divisible", "primality",
                 "divisible only", "only by 1", "divided evenly",
                 "소수", "소수 판별"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("filter_elements", "Filter Elements", "operation",
                "Select elements from a collection based on a predicate",
                ["filter", "select", "keep", "choose", "pick", "retain"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("search", "Search", "operation",
                "Find the position or presence of a target element in a collection",
                ["search", "look up", "lookup", "position of", "index of",
                 "검색", "찾기", "탐색", "찾는", "위치"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("filter", "Filter", "operation",
                "Select or exclude elements based on a condition",
                ["filter", "select", "keep", "choose", "pick", "retain",
                 "keeps only", "keep only", "only values",
                 "골라", "뽑아", "선택", "제외", "골르", "뽑으",
                 "필터링", "filtering"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("sort_descending", "Sort Descending", "operation",
                "Arrange elements from largest to smallest",
                ["descending", "reverse order", "largest first", "decreasing",
                 "highest to lowest", "from largest to smallest",
                 "biggest to smallest",
                 "high to low", "big to small",
                 "내림차순", "큰 순서대로", "큰 것부터", "역순", "내림"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("sort_ascending", "Sort Ascending", "operation",
                "Arrange elements from smallest to largest",
                ["ascending", "order", "smallest first", "increasing",
                 "lowest to highest", "from smallest to largest",
                 "low to high", "small to big",
                 "오름차순", "작은 순서대로", "작은 것부터"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("remove_duplicates", "Remove Duplicates", "operation",
                "Eliminate repeated elements, keeping only unique values",
                ["deduplicate", "distinct", "unique", "redundancy", "overlap",
                 "eliminate", "redundant", "repeated",
                 "중복", "제거", "없애", "없앤", "없앰", "빼고"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("reverse_sequence", "Reverse Sequence", "operation",
                "Reverse the order of elements in a collection",
                ["reverse", "invert", "flip", "backwards", "거꾸로", "뒤집기"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("factorial", "Factorial", "operation",
                "Compute the factorial of a non-negative integer",
                ["factorial", "계승", "팩토리얼", "팩토리어"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("fibonacci", "Fibonacci", "operation",
                "Compute the Fibonacci sequence value at an index",
                ["fibonacci", "피보나치", "피보"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("concatenate", "Concatenate", "operation",
                "Join strings or sequences end to end",
                ["concatenate", "concat", "join", "결합", "연결", "이어붙"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("generate_range", "Generate Range", "operation",
                "Generate a sequence of numbers within a range",
                ["range", "from to", "interval",
                 "numbers from", "from 1 to",
                 "부터", "까지"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("is_multiple_of", "Check Multiple", "operation",
                "Determine if a number is a multiple of another",
                ["multiple", "divisible", "even", "divisible by", "modulus",
                 "배수", "짝수"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),

        Concept("count_frequency", "Element Frequency", "operation",
                "Count how many times each element occurs in a collection",
                ["appears only once", "appears once", "occurs only once",
                 "occurs once", "only once", "appears exactly once",
                 "frequency", "how many times each",
                 "빈도", "한 번만", "한번만", "한 번씩 나타"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),

        # ── Additional operation concepts ──
        Concept("count_elements", "Count Elements", "operation",
                "Count the number of elements in a collection",
                ["count", "how many", "number of", "total count", "size",
                 "갯수", "몇 개", "개수"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("calculate_median", "Calculate Median", "operation",
                "Find the middle value of a sorted collection",
                ["median", "middle value", "central value", "50th percentile",
                 "중앙값"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("calculate_mode", "Calculate Mode", "operation",
                "Find the most frequently occurring value",
                ["mode", "most common", "most frequent", "highest frequency",
                 "최빈값"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("calculate_variance", "Calculate Variance", "operation",
                "Compute the statistical variance of a collection",
                ["variance", "spread", "dispersion", "standard deviation",
                 "분산", "표준편차"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("flatten_list", "Flatten List", "operation",
                "Collapse nested lists into a single flat list",
                ["flatten", "unfold", "unnest", "deep copy", "flat",
                 "평탄화", "펼치기"],
                ["python", "kotlin", "java", "javascript", "go", "rust"]),
        Concept("group_by_key", "Group By Key", "operation",
                "Group collection elements by a key function",
                ["group by", "group", "bucket", "categorize", "partition",
                 "그룹화"],
                ["python", "kotlin", "java", "javascript", "go", "rust"]),
        Concept("map_transform", "Map Transform", "operation",
                "Apply a function to each element producing a new collection",
                ["map", "transform", "apply", "convert", "each element",
                 "변환", "적용"],
                ["python", "kotlin", "java", "javascript", "go", "rust"]),
        Concept("reduce_aggregate", "Reduce Aggregate", "operation",
                "Combine all elements into a single value using a reducer",
                ["reduce", "fold", "accumulate", "aggregate", "combine",
                 "축약", "누적"],
                ["python", "kotlin", "java", "javascript", "go", "rust"]),
        Concept("take_n", "Take First N", "operation",
                "Take the first N elements from a collection",
                ["take", "first n", "head", "top n", "prefix",
                 "처음", "앞에서"],
                ["python", "kotlin", "java", "javascript", "go", "rust"]),
        Concept("drop_n", "Drop First N", "operation",
                "Skip the first N elements from a collection",
                ["drop", "skip", "tail", "offset", "discard",
                 "건너뛰기", "뒤에서"],
                ["python", "kotlin", "java", "javascript", "go", "rust"]),

        # ── Output / program generation concepts ──
        Concept("emit_output", "Emit Output", "operation",
                "Output or display a value (print, echo, show)",
                ["output", "print", "display", "show", "echo",
                 "prints", "displays", "outputs",
                 "출력", "보여", "보여줘"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("repeat_output", "Repeat Output", "operation",
                "Repeat an output action a specified number of times",
                ["do twice", "do thrice", "repeat output", "print twice",
                 "print thrice", "output twice", "display twice",
                 "두 번", "세 번"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
        Concept("literal_value", "Literal Value", "operation",
                "A concrete literal value (string, number)",
                ["hello", "welcome", "apple", "quoted string",
                 "literal value", "the number", "hello world",
                 "값을", "문자열을", "숫자를"],
                ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    ]
    for c in defaults:
        store.add(c)

# ── Algorithmic Primitive Concepts ──
# These are GENERAL algorithmic techniques stored in the KG.
# NOT per-problem hardcoding — these describe domain-agnostic patterns.
_ALGORITHMIC_CONCEPTS = [
    Concept("cp_linear_scan", "Linear Scan", "algorithmic_primitive",
            "Single pass through data tracking state. O(n) time, O(1) space.",
            ["linear scan", "single pass", "iterate", "one pass",
             "한번에", "선형"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_two_pointer", "Two Pointer", "algorithmic_primitive",
            "Two pointers moving through sorted data or from opposite ends. O(n) time.",
            ["two pointer", "two pointers", "left right", "dual pointer",
             "양쪽 포인터", "투포인터"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_sliding_window", "Sliding Window", "algorithmic_primitive",
            "Fixed or variable-size window over contiguous elements. O(n) time.",
            ["sliding window", "window", "subarray window", "스리핑 윈도우"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_binary_search", "Binary Search", "algorithmic_primitive",
            "Halve the search space repeatedly. O(log n) time.",
            ["binary search", "bisection", "이분 탐색", "이진 탐색"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_prefix_sum", "Prefix Sum", "algorithmic_primitive",
            "Precompute cumulative sums for O(1) range queries. O(n) time, O(n) space.",
            ["prefix sum", "cumulative sum", "running sum", "누적 합", "구간 합"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_hash_map", "Hash Map Lookup", "algorithmic_primitive",
            "Use hash table for O(1) amortized lookups. O(n) time, O(n) space.",
            ["hash map", "hash table", "dictionary", "lookup", "frequency map",
             "해시맵", "딕셔너리"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_sort", "Sorting", "algorithmic_primitive",
            "Sort data first then apply linear scan or two-pointer. O(n log n) time.",
            ["sort", "sorting", "order", "정렬", "오름차순", "내림차순"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_greedy", "Greedy Algorithm", "algorithmic_primitive",
            "Make locally optimal choices at each step. O(n log n) for sorting-based.",
            ["greedy", "optimal choice", "greedy approach", "탐욕", "탐욕 알고리즘"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_dynamic_programming", "Dynamic Programming", "algorithmic_primitive",
            "Break into overlapping subproblems, store results. O(n*m) time/space.",
            ["dynamic programming", "dp", "memoization", "tabulation",
             "점화식", "메모이제이션", "동적 계획법"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_bfs", "Breadth-First Search", "algorithmic_primitive",
            "Level-order traversal of graph/grid. O(V+E) time, O(V) space.",
            ["bfs", "breadth first", "level order", "breadth-first search",
             "너비 우선 탐색"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_dfs", "Depth-First Search", "algorithmic_primitive",
            "Recursive/iterative deep traversal of graph/grid. O(V+E) time.",
            ["dfs", "depth first", "backtracking", "depth-first search",
             "깊이 우선 탐색", "백트래킹"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_bit_manipulation", "Bit Manipulation", "algorithmic_primitive",
            "Use bitwise operations for efficient computation.",
            ["bitwise", "bit manipulation", "mask", "xor", "비트 연산"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_math_formula", "Math Formula", "algorithmic_primitive",
            "Direct mathematical formula or number theory. O(1) to O(n).",
            ["math", "formula", "number theory", "closed form", "수학", "공식"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_monotonic_stack", "Monotonic Stack", "algorithmic_primitive",
            "Monotonic stack for next greater/smaller element. O(n) time.",
            ["monotonic stack", "next greater", "next smaller", "histogram",
             "단조 스택"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_counting", "Counting", "algorithmic_primitive",
            "Count occurrences or enumerate combinations. O(n) time.",
            ["count", "frequency", "enumerate", "counting",
             "빈도", "계산", "횟수"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
    Concept("cp_simulation", "Simulation", "algorithmic_primitive",
            "Directly simulate the described process step by step.",
            ["simulate", "simulation", "step by step", "direct",
             "시뮬레이션", "직접 처리"],
            ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]),
]


def seed_algorithmic_concepts(store: ConceptStore) -> None:
    """Seed the concept store with algorithmic primitive concepts."""
    for c in _ALGORITHMIC_CONCEPTS:
        store.add(c)
