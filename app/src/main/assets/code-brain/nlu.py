# -*- coding: utf-8 -*-
"""자연어 이해 모듈 - 오타 tolerant, 슬롯 추출, 의도 분류, 경험 누적"""
import re
import difflib
import json
import os
from typing import Dict, List, Optional

TYPO_MAP = {
    "만들줘": "만들어줘", "만들어": "만들어줘", "만들": "만들어줘",
    "만ㄹ들어": "만들어줘", "만ㄹ": "만들어줘",
    "짜줘": "작성해줘", "작성": "작성해줘", "생성": "생성해줘",
    "고쳐줘": "고쳐줘", "수정": "수정해줘", "바꿔": "변경해줘",
    "설명": "설명해줘", "알려줘": "설명해줘", "뭐야": "설명해줘",
    "함수": "함수", "클래스": "클래스", "스크립트": "스크립트",
    "파이썬": "python", "파이썬으로": "python",
    "코틀린": "kotlin", "자바": "java",
    "더하기": "더하기", "플러스": "더하기",
    "곱하기": "곱하기", "빼기": "빼기", "나누기": "나누기",
    "읽어": "읽기", "써": "쓰기", "변환": "변환",
}

PARTICLES = ["는", "은", "이", "가", "을", "를", "의", "에", "에서",
             "로", "으로", "와", "과", "도", "만", "부터", "까지",
             "한테", "에게", "께", "처럼", "만 해도"]

INTENT_KEYWORDS = {
    "code.generate": ["만들", "작성", "생성", "코딩", "짜", "함수", "클래스", "스크립트", "짜줘"],
    "code.explain": ["설명", "뭐", "의미", "어떻게", "왜", "분석", "알려줘"],
    "code.verify": ["검증", "테스트", "확인", "올바른", "맞는지", "테케"],
    "status": ["상태", "정보", "현황", "뇌"],
    "help": ["도움", "명령", "뭐하지", "할 수 있는", "무엇"],
}

LANG_MAP = {
    "python": ["파이썬", "python", "파이썬으로"],
    "kotlin": ["코틀린", "kotlin"],
    "java": ["자바", "java"],
    "javascript": ["자바스크립트", "javascript", "js"],
    "c": ["c언어", "씨언어"],
    "cpp": ["씨플러스플러스", "cpp", "c++"],
    "go": ["고언어", "go"],
    "rust": ["러스트", "rust"],
}

EXPERIENCE_DIR = os.path.join(os.path.expanduser("~"), ".rena_bridge", "experience")

def _exp_path() -> str:
    os.makedirs(EXPERIENCE_DIR, exist_ok=True)
    return os.path.join(EXPERIENCE_DIR, "interactions.jsonl")

def record_experience(text: str, intent: str, slots: Dict, success: bool = True):
    try:
        record = {"text": text, "intent": intent, "slots": slots, "success": success}
        with open(_exp_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def load_past_experiences(limit: int = 50) -> List[Dict]:
    path = _exp_path()
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception:
        pass
    return records[-limit:]

def normalize_korean(text: str) -> str:
    text = text.strip()
    for wrong in sorted(TYPO_MAP.keys(), key=len, reverse=True):
        text = text.replace(wrong, TYPO_MAP[wrong])
    for p in PARTICLES:
        text = re.sub(rf"\s*{re.escape(p)}\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fuzzy_match(word: str, candidates: List[str], threshold: float = 0.55) -> Optional[str]:
    best = difflib.get_close_matches(word, candidates, n=1, cutoff=threshold)
    return best[0] if best else None

def extract_language(text: str) -> Optional[str]:
    for lang, keywords in LANG_MAP.items():
        for kw in keywords:
            if kw in text:
                return lang
    return None

def classify_intent(text: str) -> str:
    best_intent = "code.generate"
    best_score = 0
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_intent = intent
    if best_score == 0:
        best_intent = "code.generate"
    return best_intent

def extract_slots(text: str, intent: str) -> Dict:
    slots = {}
    if intent == "code.generate":
        lang = extract_language(text)
        if lang:
            slots["language"] = lang
        work_keywords = ["더하기", "뺄기", "곱하기", "나누기", "정렬", "검색",
                         "변환", "파싱", "읽기", "쓰기", "다운로드", "업로드"]
        for wk in work_keywords:
            if wk in text:
                slots["task"] = wk
                break
    return slots

def understand(text: str) -> Dict:
    normalized = normalize_korean(text)
    intent = classify_intent(normalized)
    slots = extract_slots(normalized, intent)
    missing = []
    if intent == "code.generate" and "language" not in slots:
        missing.append("language")
    record_experience(text, intent, slots, success=(len(missing) == 0))
    return {
        "original": text,
        "normalized": normalized,
        "intent": intent,
        "slots": slots,
        "missing": missing,
        "confidence": 0.85 if not missing else 0.45,
    }
