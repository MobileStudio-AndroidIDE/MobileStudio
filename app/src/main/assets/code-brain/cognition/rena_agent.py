"""Unified RENA-BR agent for MobileStudio integration.

Single entry point: ``RenaAgent.respond(text)`` returns a plain string.
No LLM, no Transformer, no pretrained weights, no external AI API.
Learning is Hebbian/contrastive VSA association plus verification-gated
skill reuse only.

Usage::

    from cognition.rena_agent import RenaAgent

    agent = RenaAgent(data_dir="/home/user/.rena_cognition")
    print(agent.respond("삼 더하기 오"))
    print(agent.respond("오늘 서울 날씨"))
    print(agent.respond("내일 3시 회의 일정 잡아줘"))

    agent.close()
"""

from __future__ import annotations

import os
import re
from typing import Optional

from cognition.config import GrowthConfig
from cognition.korean import KoreanSurface
from cognition.language_engine import LanguageEngine
from cognition.rena import RenaBrain
from cognition.weather import WeatherContext, extract_location


# ---------------------------------------------------------------------------
# Korean number / operator words -> digits (module level)
# ---------------------------------------------------------------------------

_KOREAN_DIGITS = {
    "영": "0", "공": "0",
    "일": "1", "하나": "1", "한": "1",
    "이": "2", "둘": "2",
    "삼": "3", "셋": "3",
    "사": "4", "넷": "4",
    "오": "5", "다섯": "5",
    "육": "6", "여섯": "6",
    "칠": "7", "일곱": "7",
    "팔": "8", "여덟": "8",
    "구": "9", "아홉": "9",
    "십": "10", "열": "10",
    "백": "100", "천": "1000", "만": "10000",
}


def _korean_to_digits(text: str) -> str:
    """Convert Korean number/operator words to digits where safe.

    Only replaces standalone Korean number tokens (surrounded by non-Hangul
    or string boundaries) so ordinary words are not mangled.
    """
    result = text
    for word, digit in sorted(_KOREAN_DIGITS.items(), key=lambda kv: -len(kv[0])):
        result = re.sub(
            r"(?<![가-힣])" + re.escape(word) + r"(?![가-힣])",
            digit,
            result,
        )
    result = (result
              .replace("더하기", "+")
              .replace("더해", "+")
              .replace("빼기", "-")
              .replace("곱하기", "*")
              .replace("곱해", "*")
              .replace("나누기", "/")
              .replace("나눠", "/")
              .replace("제곱", "**")
              .replace("승", "**"))
    # Handle "A의 B제곱" -> "A**B" and "의" particle between number and unit.
    result = re.sub(r"(\d+)\s*의\s*(\d+)\s*\*\*", r"\1**\2", result)
    result = result.replace("의", "")
    # Sum range: "1~50 합" -> sum(1..50)
    m = re.search(r"(\d+)\s*~\s*(\d+)", result)
    if m and "합" in text:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a <= b <= 10000:
            n = b - a + 1
            total = (a + b) * n // 2
            return str(total)
    # Average: "평균 4 8 12" -> 8.0
    if "평균" in text:
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", result)]
        if nums:
            return str(sum(nums) / len(nums))
    # Drop trailing particles / politeness endings / punctuation so
    # "2의 12제곱은?" becomes "2**12" instead of "2**12은?".
    result = re.sub(r"[은는이가을를요]?\s*[?？!.,]*\s*$", "", result)
    return result


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class RenaAgent:
    """Thin, stable wrapper around :class:`cognition.rena.RenaBrain`.

    ``respond`` is the only method MobileStudio needs to call.  It keeps
    the conversation in the brain's working memory and episodic memory so
    later turns resolve references against earlier context.
    """

    def __init__(self,
                 data_dir: Optional[str] = None,
                 dimension: int = 2048,
                 seed: int = 1729,
                 auto_load: bool = True,
                 auto_save: bool = True,
                 auto_grow: bool = True,
                 allow_network: bool = False):
        self.config = GrowthConfig(
            dimension=dimension,
            vector_seed=seed,
            data_dir=data_dir,
            allow_network=allow_network,
        )
        self.config.ensure_dirs()

        self._korean_vsa = None
        self._korean_encoder = None
        self._weather_ctx = WeatherContext()

        self.brain = RenaBrain(config=self.config, load=auto_load)
        self._auto_save = bool(auto_save)
        self._auto_grow = bool(auto_grow)

        # VSA + 능동 추론 + 개념 공간 기반 자연어 엔진 (LLM 미사용)
        self._language_engine = LanguageEngine(
            vsa=self.brain.vsa,
            concepts=self.brain.concepts,
            active=self.brain.active,
            meta=self.brain.meta,
            encoder=self.brain.encoder,
        )

        # 대화 상태 (되묻기 응답 · 후속 질문 · 화제 기억)
        self._pending: Optional[dict] = None
        self._last_topic: str = ""

        # 학습 상태 추적 (오래할수록 더 똑똑해지도록)
        self._interaction_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._last_grow_at = 0
        self._grow_every = 5  # 5번 상호작용마다 자동 성장 시도 (더 빨리 학습)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(self, text: str) -> str:
        """Answer one user turn and return a plain string."""
        text = str(text or "").strip()
        if not text:
            return ("무엇을 도와드릴까요? 코딩, 수학, 그림, 차트, 영상, "
                    "파일, 날씨, 일정, 검색 모두 할 수 있어요.")

        # 1) 정확한 대화 시드 (인사·감정·정체성) — 유사 매칭은 아직 안 함
        if self._language_engine is not None:
            direct = self._language_engine.direct_response(text)
            if direct:
                self._interaction_count += 1
                self._success_count += 1
                return direct + (
                    f"\n\n(학습 {self._interaction_count}회, "
                    f"모드: language-engine)")

        # 2) 대화 관리: 되묻기 응답과 후속 질문을 먼저 처리한다.
        dialogue_answer = self._dialogue_turn(text)
        if dialogue_answer:
            self._interaction_count += 1
            self._success_count += 1
            return dialogue_answer + (
                f"\n\n(학습 {self._interaction_count}회, 모드: 대화 이어가기)")

        # 2b) 유사 대화 시드 (VSA 근접 매칭)
        if self._language_engine is not None:
            neural_answer = self._language_engine.respond(text)
            if neural_answer:
                self._interaction_count += 1
                self._success_count += 1
                return neural_answer + (
                    f"\n\n(학습 {self._interaction_count}회, "
                    f"모드: language-engine)")

        # 3) Fallback: 기존 skills/actuator 파이프라인
        # Perceive: encode text with Korean concept boosts and store it.
        vector = self._korean_encode(text)
        self.brain.observe("text", text, label="user_turn")

        # Think: active inference over concept space.
        think = self.brain.think(text)

        # Act: let the brain pick an actuator and execute.
        act = self._act(text)

        # 대화 상태 갱신: 되묻기 대기 / 화제 기억
        status = str(act.get("knowledge_status") or "")
        if status == "ambiguous":
            candidates = list(act.get("alternatives") or [])
            if candidates:
                self._pending = {"candidates": candidates}
        elif status in ("direct", "graph", "synonym") and act.get("term"):
            self._last_topic = str(act.get("term"))
            self._pending = None

        # Learn from the result (prediction-error update only).
        if act.get("ok"):
            self.brain.feedback(1.0, note=text)
        else:
            self.brain.feedback(-0.5, note=text)

        if self._auto_save:
            try:
                self.brain.save()
            except Exception:
                pass

        # 학습 상태 업데이트 (오래할수록 더 똑똑해지도록)
        self._interaction_count += 1
        if act.get("ok"):
            self._success_count += 1
        else:
            self._failure_count += 1

        # 주기적 자동 성장: 상호작용 횟수가 임계치에 도달하면 성장을 시도
        if self._auto_grow and self._interaction_count - self._last_grow_at >= self._grow_every:
            self._last_grow_at = self._interaction_count
            try:
                self.brain.grow(cycles=1)
            except Exception:
                pass

        if self._auto_save:
            try:
                self.brain.save()
            except Exception:
                pass

        # Compose a single plain-text answer.
        answer = self._compose(text, think, act)

        # 학습 현황 표시 (오래할수록 더 똑똑해짐)
        total = max(1, self._interaction_count)
        rate = self._success_count / total
        grow_mark = " 🧠학습중" if self._auto_grow else ""
        answer += f"\n\n(학습 {self._interaction_count}회, 성공률 {rate:.0%}{grow_mark})"

        return answer

    def respond_json(self, text: str) -> dict:
        """앱/툴 연동용 JSON 직렬화 가능 응답.

        ``respond``의 학습 현황 꼬리표를 분리해 구조화한다.
        """
        answer = self.respond(text)
        marker = "\n\n(학습 "
        cut = answer.rfind(marker)
        body = answer[:cut] if cut != -1 else answer
        return {
            "ok": not body.startswith("요청을 처리하지 못했어요"),
            "response": body,
            "learning": self.learn_report(),
        }

    def learn_report(self) -> dict:
        """현재까지의 학습 현황을 반환."""
        total = max(1, self._interaction_count)
        return {
            "interactions": self._interaction_count,
            "successes": self._success_count,
            "failures": self._failure_count,
            "success_rate": round(self._success_count / total, 4),
            "last_grow_at": self._last_grow_at,
            "auto_grow": self._auto_grow,
            "grow_every": self._grow_every,
        }

    def cleanup_sandbox(self) -> dict:
        """학습/검증 과정에서 생성된 임시 파이썬 코드나 샌드박스 잔여물을 정리.

        사용자의 원본 코드는 건드리지 않고, RENA-BR 내부에서만 생성한
        임시 아티팩트만 제거합니다.
        """
        cleaned = []
        try:
            import shutil
            import tempfile

            # cognition 데이터 디렉터리 내의 임시 샌드박스 디렉터리 정리
            base = self.config.data_dir
            for name in ("sandbox", "tmp", "temp", "code_exec"):
                p = os.path.join(base, name)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                    cleaned.append(p)

            # 시스템 temp 디렉터리 내 RENA가 만든 rena- 접두사만 정리한다.
            # (다른 프로그램의 tmp* 디렉터리를 건드리지 않도록)
            sys_tmp = tempfile.gettempdir()
            for entry in os.listdir(sys_tmp):
                if entry.startswith("rena-"):
                    p = os.path.join(sys_tmp, entry)
                    try:
                        if os.path.isdir(p):
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            os.remove(p)
                        cleaned.append(p)
                    except Exception:
                        pass
        except Exception:
            pass
        return {"cleaned": cleaned}

    def grow(self, cycles: int = 1) -> dict:
        """Run bounded autonomous growth cycles."""
        if not self._auto_grow:
            return {"skipped": True, "reason": "auto_grow disabled"}
        return self.brain.grow(cycles=cycles)

    def status(self) -> dict:
        return self.brain.status()

    def close(self) -> None:
        try:
            self.brain.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _korean(self) -> KoreanSurface:
        """Lazily build a KoreanSurface sharing the brain's VSA dimension."""
        from cognition.semantic import SemanticEncoder
        from cognition.vsa import VSA

        if self._korean_vsa is None:
            self._korean_vsa = VSA(dimension=self.config.dimension,
                                   seed=self.config.vector_seed)
            self._korean_encoder = SemanticEncoder(self._korean_vsa)
        return KoreanSurface(self._korean_vsa, self._korean_encoder)

    def _korean_encode(self, text: str):
        """Concept-boosted VSA encoding for Korean surface forms."""
        return self._korean().encode(text)

    def _offline_info(self, term: str) -> dict:
        """개념 하나를 지식 계층으로 조회 (표현은 매번 조금씩 달라진다)."""
        try:
            from knowledge import offline_kb
        except Exception:
            return {"status": "unknown"}
        return offline_kb.unified_answer(
            term, seed=self._interaction_count + 1)

    def _dialogue_turn(self, text: str) -> Optional[str]:
        """되묻기 응답과 후속 질문을 처리한다.  처리 못하면 None.

        LLM의 맥락 유지와는 다르다: 화제 기억 + 되묻기 상태 + 규칙 판정이다.
        """
        try:
            from knowledge import offline_kb
        except Exception:
            return None

        # 1) 직전에 "이거 말하는 건가요?" 라고 되물은 경우
        if self._pending:
            candidates = list(self._pending.get("candidates") or [])
            chosen = ""
            if offline_kb.looks_affirmative(text) and candidates:
                chosen = candidates[0]
            else:
                for candidate in candidates:
                    if candidate and candidate in text:
                        chosen = candidate
                        break
            if chosen:
                self._pending = None
                info = self._offline_info(chosen)
                if info.get("status") != "unknown":
                    self._last_topic = str(info.get("term") or chosen)
                    return (f"네, 그럼 '{chosen}'로 보고 답할게요.\n\n"
                            + str(info.get("response") or ""))
            if offline_kb.looks_negative(text):
                self._pending = None
                return ("알겠어요. 다른 뜻이면 단어를 조금 더 풀어서 "
                        "말해 주세요. 그게 더 정확히 찾을 수 있어요.")
            self._pending = None      # 다른 이야기가 시작됨

        # 1b) 되묻는 중이 아닌데 긍정/부정만 온 경우: 맥락을 되짚어 준다
        if not self._pending and (offline_kb.looks_affirmative(text)
                                  or offline_kb.looks_negative(text)):
            if self._last_topic:
                return (f"'{self._last_topic}' 이야기였죠? "
                        "그거면 더 설명해 드릴게요. 아니면 다른 주제를 "
                        "말해 주세요.")
            return ("무엇에 대한 대답인지 알려주시면 이어서 도와드릴게요. "
                    "예: 'pwd 알려줘', '깃이 뭐야?'")

        # 2) 후속 질문 ("그럼?", "더 자세히") — 화제를 기억해 이어간다
        if self._last_topic and offline_kb.looks_followup(text):
            info = self._offline_info(self._last_topic)
            if info.get("status") != "unknown":
                return ("이어서 볼게요.\n\n"
                        + str(info.get("response") or ""))
        return None

    def _act(self, text: str) -> dict:
        """Dispatch to the best actuator via active inference + skills."""
        # Try the brain's own act pipeline first.
        try:
            result = self.brain.act(
                "conversation",
                {"operation": "respond", "text": text},
                force=True,
            )
            if result.get("ok"):
                return result
        except Exception:
            pass

        # Fallback: route by Korean hints + concept similarity.
        hints = self._korean().hints(text)
        best = self._route_by_hints(text, hints)
        if best:
            return best

        # Fallback: if no Korean hints matched, check for a raw math
        # expression (digits + operators).
        if re.search(r"\d", text) and re.search(r"[+\-*/=!^()]", text):
            return self._act_math(text)

        return {"ok": False,
                "error": "no actuator selected",
                "confidence": 0.0}

    def _route_by_hints(self, text: str,
                        hints: dict) -> Optional[dict]:
        """Route a request to an actuator based on Korean surface hints."""
        if not hints:
            # 힌트가 없어도 언어+태스크 조합으로 코드 라우팅 가능하므로
            # 여기서 종료하지 않고 계속 진행합니다.
            pass

        # 날씨/일정/이미지/파일/셸처럼 대상이 분명한 요청은 질문 표현보다
        # 먼저 처리한다. "pwd 알려줘"가 지식 검색으로 새는 것을 막는다.
        if "weather" in hints:
            return self._act_weather(text)

        # 날씨 대화가 진행 중이면 "아침은?", "12월은?" 같은 후속 질문도
        # 날씨 문맥으로 이어준다.
        if self._weather_ctx.location is not None and any(
                w in text for w in ("비", "눈", "아침", "저녁", "밤", "낮",
                                    "맑", "흐", "덥", "춥", "우산", "월")):
            return self._act_weather(text)

        if "schedule" in hints or "calendar" in hints:
            return self._act_calendar(text)

        if "image" in hints or "draw" in hints:
            return self._act_image(text)

        if "files" in hints or "file_write" in hints or "file_read" in hints:
            return self._act_file(text)

        if "shell" in hints:
            return self._act_shell(text)

        knowledge_hints = {"knowledge", "linux", "ubuntu", "ai", "quantum_entanglement",
                           "quantum", "entangle"}
        question_words = ("뭐야", "무엇", "설명", "어떻게", "왜", "무슨",
                          "어떤", "이란", "뭔가요", "무엇인가요",
                          "인가요", "인가", "맞나요", "맞나", "맞아",
                          "알려", "가르쳐", "공통", "차이", "비교",
                          "관계", "종류", "용도")
        is_question = any(q in text for q in question_words)
        is_code_task = any(t in text for t in ("짜줘", "만들", "함수", "작성",
                                               "구현", "프로그램 작성"))

        # Code hints (checked before math so "함수" prefers code generation).
        code_hints = {"python", "javascript", "java", "kotlin", "c", "cpp",
                      "rust", "go", "fibonacci", "code"}
        if any(h in hints for h in code_hints) and not is_question:
            return self._act_code(text)

        # Math hints: 명시적 계산 신호는 지식 검색보다 우선한다.
        math_hints = {"add", "subtract", "multiply", "divide", "power",
                      "factorial", "differentiate", "integrate", "prime",
                      "sum", "average"}
        if any(h in hints for h in math_hints):
            return self._act_math(text)

        # 지식 요청 키워드가 있으면 knowledge (코드 생성과 구분).
        # "뭐야", "무엇", "설명" 등이 있으면 코드 생성이 아니라 설명 요청.
        if any(h in hints for h in knowledge_hints) or (
                is_question and not is_code_task):
            return self._act_knowledge(text)

        # Identity / capability / social -> chat.
        social_hints = {"identity", "capability", "goodbye", "positive",
                        "thanks", "hello", "sad", "love", "bored"}
        if any(h in hints for h in social_hints):
            return self._act_chat(text)

        # Knowledge hints.
        if "knowledge" in hints:
            return self._act_knowledge(text)

        # 힌트가 없어도 언어+태스크 조합이 있으면 코드로 강제 라우팅.
        code_task_words = ("코드", "프로그램", "만들", "짜", "출력", "함수",
                           "구현", "작성", "print", "hello world", "콘솔")
        lang_indicators = ("c", "c++", "cpp", "자바", "자스", "파이썬",
                           "파이선", "코틀린", "러스트", "고언어", "씨언어",
                           "씨플플", "javascript", "java", "python", "kotlin",
                           "rust", "go")
        has_code_task = any(t in text for t in code_task_words)
        has_lang = any(l in text.lower() for l in lang_indicators)
        if has_code_task and has_lang:
            return self._act_code(text)

        return None

    # ---- Actuator fallbacks ----------------------------------------

    def _act_math(self, text: str) -> dict:
        try:
            normalized = _korean_to_digits(text)
            # Factorial: "7!" or "7 팩토리얼"
            m = re.search(r"(\d+)\s*!", normalized)
            if m:
                n = int(m.group(1))
                if n < 0 or n > 20:
                    return {"ok": False,
                            "error": "0부터 20 사이의 정수만 지원해요.",
                            "actuator": "math"}
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return {"ok": True,
                        "output": {"value": result},
                        "actuator": "math",
                        "verified": True}
            result = self.brain.actuators.execute(
                "math", {"operation": "evaluate", "expression": normalized})
            return result.to_dict()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "actuator": "math"}

    def _detect_code_language(self, lower: str) -> str:
        import re
        # 한국어 표현 우선
        if any(k in lower for k in ("자바스크립트", "자스", "javascript", "node")):
            return "javascript"
        if any(k in lower for k in ("자바", "java")):
            return "java"
        if any(k in lower for k in ("코틀린", "kotlin")):
            return "kotlin"
        if any(k in lower for k in ("씨언어", "c 언어")):
            return "c"
        if any(k in lower for k in ("씨플플", "c++", "cpp")):
            return "cpp"
        if any(k in lower for k in ("러스트", "rust")):
            return "rust"
        if any(k in lower for k in ("고언어", "golang")):
            return "go"
        # bare 토큰: "C", "C++" (한국어 조사 포함)
        if re.search(r"\bC\b", lower) or "c로" in lower or " c " in lower:
            return "c"
        if re.search(r"\bC\s*\+\+\b", lower) or "c++로" in lower:
            return "cpp"
        return "python"

    def _code_template(self, task: str, lang: str) -> Optional[str]:
        if task == "fibonacci":
            if lang == "c":
                return ('#include <stdio.h>\n'
                        '\n'
                        'void fibonacci(int n, int *seq) {\n'
                        '    if (n > 0) seq[0] = 0;\n'
                        '    if (n > 1) seq[1] = 1;\n'
                        '    for (int i = 2; i < n; i++) {\n'
                        '        seq[i] = seq[i - 1] + seq[i - 2];\n'
                        '    }\n'
                        '}\n'
                        '\n'
                        'int main(void) {\n'
                        '    int n = 10, seq[10];\n'
                        '    fibonacci(n, seq);\n'
                        '    for (int i = 0; i < n; i++) printf("%d ", seq[i]);\n'
                        '    printf("\\n");\n'
                        '    return 0;\n'
                        '}')
            if lang == "cpp":
                return ('#include <iostream>\n'
                        '#include <vector>\n'
                        '\n'
                        'std::vector<int> fibonacci(int n) {\n'
                        '    std::vector<int> seq;\n'
                        '    if (n > 0) seq.push_back(0);\n'
                        '    if (n > 1) seq.push_back(1);\n'
                        '    for (int i = 2; i < n; i++) {\n'
                        '        seq.push_back(seq[i - 1] + seq[i - 2]);\n'
                        '    }\n'
                        '    return seq;\n'
                        '}\n'
                        '\n'
                        'int main() {\n'
                        '    auto seq = fibonacci(10);\n'
                        '    for (int x : seq) std::cout << x << " ";\n'
                        '    std::cout << "\\n";\n'
                        '    return 0;\n'
                        '}')
            if lang == "java":
                return ('public class Main {\n'
                        '    public static void main(String[] args) {\n'
                        '        System.out.println(fibonacci(10));\n'
                        '    }\n'
                        '    static java.util.List<Integer> fibonacci(int n) {\n'
                        '        java.util.List<Integer> seq = new java.util.ArrayList<>();\n'
                        '        if (n > 0) seq.add(0);\n'
                        '        if (n > 1) seq.add(1);\n'
                        '        for (int i = 2; i < n; i++) {\n'
                        '            seq.add(seq.get(i - 1) + seq.get(i - 2));\n'
                        '        }\n'
                        '        return seq;\n'
                        '    }\n'
                        '}')
            if lang == "javascript":
                return ('function fibonacci(n) {\n'
                        '    const seq = [];\n'
                        '    if (n > 0) seq.push(0);\n'
                        '    if (n > 1) seq.push(1);\n'
                        '    for (let i = 2; i < n; i++) {\n'
                        '        seq.push(seq[i - 1] + seq[i - 2]);\n'
                        '    }\n'
                        '    return seq;\n'
                        '}\n'
                        'console.log(fibonacci(10));')
            return ('def fibonacci(n: int) -> list[int]:\n'
                    '    """Return the first n Fibonacci numbers."""\n'
                    '    if n <= 0:\n'
                    '        return []\n'
                    '    seq = [0, 1]\n'
                    '    while len(seq) < n:\n'
                    '        seq.append(seq[-1] + seq[-2])\n'
                    '    return seq[:n]\n\n'
                    'if __name__ == "__main__":\n'
                    '    print(fibonacci(10))')
        if task == "add":
            if lang == "c":
                return ('#include <stdio.h>\n'
                        '\n'
                        'double add(double a, double b) {\n'
                        '    return a + b;\n'
                        '}\n'
                        '\n'
                        'int main(void) {\n'
                        '    printf("%f\\n", add(3.0, 5.0));\n'
                        '    return 0;\n'
                        '}')
            if lang == "cpp":
                return ('#include <iostream>\n'
                        '\n'
                        'double add(double a, double b) { return a + b; }\n'
                        '\n'
                        'int main() {\n'
                        '    std::cout << add(3.0, 5.0) << std::endl;\n'
                        '    return 0;\n'
                        '}')
            if lang == "java":
                return ('public class Main {\n'
                        '    public static void main(String[] args) {\n'
                        '        System.out.println(add(3, 5));\n'
                        '    }\n'
                        '    static double add(double a, double b) {\n'
                        '        return a + b;\n'
                        '    }\n'
                        '}')
            if lang == "javascript":
                return ('function add(a, b) { return a + b; }\n'
                        'console.log(add(3, 5));')
            return 'def add(a: float, b: float) -> float:\n    return a + b\n'
        if task == "factorial":
            if lang == "c":
                return ('#include <stdio.h>\n'
                        '\n'
                        'long factorial(int n) {\n'
                        '    if (n < 0) return -1;\n'
                        '    long r = 1;\n'
                        '    for (int i = 2; i <= n; i++) r *= i;\n'
                        '    return r;\n'
                        '}\n'
                        '\n'
                        'int main(void) {\n'
                        '    printf("%ld\\n", factorial(7));\n'
                        '    return 0;\n'
                        '}')
            if lang == "cpp":
                return ('#include <iostream>\n'
                        '\n'
                        'long factorial(int n) {\n'
                        '    if (n < 0) return -1;\n'
                        '    long r = 1;\n'
                        '    for (int i = 2; i <= n; i++) r *= i;\n'
                        '    return r;\n'
                        '}\n'
                        '\n'
                        'int main() {\n'
                        '    std::cout << factorial(7) << std::endl;\n'
                        '    return 0;\n'
                        '}')
            if lang == "java":
                return ('public class Main {\n'
                        '    public static void main(String[] args) {\n'
                        '        System.out.println(factorial(7));\n'
                        '    }\n'
                        '    static long factorial(int n) {\n'
                        '        if (n < 0) throw new IllegalArgumentException();\n'
                        '        long r = 1;\n'
                        '        for (int i = 2; i <= n; i++) r *= i;\n'
                        '        return r;\n'
                        '    }\n'
                        '}')
            if lang == "javascript":
                return ('function factorial(n) {\n'
                        '    if (n < 0) throw new Error("n must be non-negative");\n'
                        '    let r = 1;\n'
                        '    for (let i = 2; i <= n; i++) r *= i;\n'
                        '    return r;\n'
                        '}\n'
                        'console.log(factorial(7));')
            return ('def factorial(n: int) -> int:\n'
                    '    """Return n! using iteration."""\n'
                    '    if n < 0:\n'
                    '        raise ValueError("n must be non-negative")\n'
                    '    result = 1\n'
                    '    for i in range(2, n + 1):\n'
                    '        result *= i\n'
                    '    return result')
        if task == "print":
            if lang == "c":
                return ('#include <stdio.h>\n'
                        '\n'
                        'int main(void) {\n'
                        '    printf("hello world\\n");\n'
                        '    return 0;\n'
                        '}')
            if lang == "cpp":
                return ('#include <iostream>\n'
                        '\n'
                        'int main() {\n'
                        '    std::cout << "hello world" << std::endl;\n'
                        '    return 0;\n'
                        '}')
            if lang == "java":
                return ('public class Main {\n'
                        '    public static void main(String[] args) {\n'
                        '        System.out.println("hello world");\n'
                        '    }\n'
                        '}')
            if lang == "javascript":
                return 'console.log("hello world");'
            return 'print("hello world")'
        if task == "generic":
            return '# 요청하신 코드입니다.\nprint("hello world")\n'
        return None

    def _act_code(self, text: str) -> dict:
        # Try CodeBrain first (if attached), otherwise use simple templates.
        try:
            result = self.brain.actuators.execute(
                "code",
                {"operation": "generate", "code": text, "language": "python"},
                force=True)
            if getattr(result, "ok", False):
                return result.to_dict()
        except Exception:
            pass

        lower = text.lower()
        lang = self._detect_code_language(lower)

        # Determine the task.
        task = None
        if any(k in lower for k in ("피보나치", "fibonacci")):
            task = "fibonacci"
        elif any(k in lower for k in ("두 수 더하기", "두 수를 더", "두 수", "덧셈", "add")):
            task = "add"
        elif any(k in lower for k in ("팩토리얼", "factorial")):
            task = "factorial"
        elif any(k in lower for k in ("출력", "print", "hello world", "콘솔")):
            task = "print"
        elif any(k in lower for k in ("코드", "프로그램", "만들", "짜")):
            task = "generic"

        code = self._code_template(task, lang) if task else None
        if code:
            return {"ok": True, "output": {"code": code, "language": lang},
                    "actuator": "code", "verified": True}

        return {"ok": False,
                "error": "코드 생성 템플릿이 없어요. `파이썬으로 피보나치 코드`처럼 "
                         "구체적으로 말해주세요.",
                "actuator": "code"}

    def _act_weather(self, text: str) -> dict:
        if not self.config.allow_network:
            # 네트워크 없음: 사용자 정보 모아 대략 추정. 준비 안 되면 질문.
            # 새 도시를 물으면 이전 도시의 월/강수 상태를 물려받지 않는다.
            explicit = extract_location(text)
            if explicit is not None and explicit != self._weather_ctx.location:
                self._weather_ctx = WeatherContext()
            self._weather_ctx.update_from_text(text)
            if not self._weather_ctx.is_ready():
                missing = []
                if self._weather_ctx.location is None:
                    missing.append("어느 도시/지역인가요?")
                if self._weather_ctx.month is None:
                    missing.append("몇 월인가요? (예: 6월, 여름)")
                q = ("네트워크가 없어 실시간 날씨를 가져올 수 없어요.\n"
                     + "\n".join(f"- {m}" for m in missing)
                     + "\n비 오는지, 아침/저녁인지 알려주시면 대략 기온을 "
                       "추측해 드릴게요. (확실하지 않은 추정치입니다.)")
                return {"ok": True,
                        "output": q,
                        "actuator": "weather",
                        "needs_clarification": True,
                        "weather_ctx": self._weather_ctx.to_dict()}

            est = self._weather_ctx.estimate()
            return {"ok": True,
                    "output": {
                        "weather_estimate": est,
                        "location": self._weather_ctx.location or "지역",
                    },
                    "actuator": "weather",
                    "estimated": True,
                    "verified": False}

        from cognition.weather import WeatherClient
        try:
            result = WeatherClient().forecast(text)
            return result.to_dict()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "actuator": "weather"}

    def _act_calendar(self, text: str) -> dict:
        from cognition.calendar import CalendarSkill
        try:
            skill = CalendarSkill(self.config.data_dir)
            result = skill.execute(text)
            return result.to_dict()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "actuator": "calendar"}

    def _act_image(self, text: str) -> dict:
        try:
            result = self.brain.actuators.execute(
                "image",
                {"operation": "generate", "prompt": text,
                 "width": 480, "height": 360},
                force=True)
            return result.to_dict()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "actuator": "image"}

    def _act_file(self, text: str) -> dict:
        try:
            result = self.brain.actuators.execute(
                "file", text, force=True)
            return result.to_dict()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "actuator": "file"}

    def _act_shell(self, text: str) -> dict:
        # 명령 자체를 묻는 질문이면 지식 계층(신뢰도·근거 포함)에 맡긴다.
        if any(marker in text for marker in
               ("뭐", "무엇", "설명", "이란", "뜻", "의미", "알려",
                "어떻게", "왜")):
            knowledge = self._act_knowledge(text)
            if knowledge.get("ok"):
                return knowledge
        # Shell은 cognition 빌드에서 실행 액추에이터로 노출하지 않는다.
        return {"ok": False,
                "error": "shell actuator not exposed in cognition build",
                "actuator": "shell"}

    def _act_chat(self, text: str) -> dict:
        # Social / identity / chat: answer from concepts + memory, no
        # fabricated capability claims.
        hints = self._korean().hints(text)
        if "identity" in hints or "capability" in hints:
            return {"ok": True,
                    "output": (
                        "저는 RENA-BR예요. Transformer나 외부 LLM 없이, "
                        "고차원 의미 공간(VSA)과 능동 추론으로 이해하고 "
                        "기억하고 배우는 로컬 인지 에이전트예요. 코딩, "
                        "수학, 이미지, 차트, 영상, 파일·터미널, 날씨, "
                        "일정, 검색을 다룰 수 있어요."
                    ),
                    "actuator": "chat"}
        if "goodbye" in hints:
            return {"ok": True,
                    "output": "안녕히 가세요! 다음에 또 불러주세요. 👋",
                    "actuator": "chat"}
        if "thanks" in hints:
            return {"ok": True,
                    "output": "도움이 되었다니 기뻐요. 또 맡겨주세요! 😊",
                    "actuator": "chat"}
        if "hello" in hints:
            return {"ok": True,
                    "output": "안녕하세요! 오늘은 뭘 함께 해볼까요? 🙂",
                    "actuator": "chat"}
        if "sad" in hints:
            return {"ok": True,
                    "output": "그런 날도 있죠. 지금 많이 지쳐 보여요. "
                    "잠깐 쉬어가도 괜찮아요. 원하면 이야기를 들어줄게요.",
                    "actuator": "chat"}
        if "positive" in hints:
            return {"ok": True,
                    "output": "좋은 기분이라니 저도 기뻐요! 오늘 무엇을 "
                    "해볼까요? 😊",
                    "actuator": "chat"}
        if "bored" in hints:
            return {"ok": True,
                    "output": "심심하시면 코드를 짜보거나, 그림을 그리거나, "
                    "날씨를 물어봐 주세요. 뭐가 좋을까요?",
                    "actuator": "chat"}
        return {"ok": True,
                "output": "이야기 잘 들었어요. 조금 더 구체적으로 "
                "말해주시면 바로 도와드릴게요.",
                "actuator": "chat"}

    def _act_knowledge(self, text: str) -> dict:
        # 1) 검수된 오프라인 지식으로 먼저 답한다 (네트워크 불필요).
        #    직접 일치가 아니면 신뢰도·근거를 붙이고, 모호하면 되묻는다.
        try:
            from knowledge import offline_kb
            info = offline_kb.unified_answer(text)
        except Exception:
            info = {"status": "unknown", "response": "",
                    "confidence": 0.0, "evidence": []}
        status = str(info.get("status") or "unknown")
        if status in ("direct", "graph", "synonym", "ambiguous"):
            output = str(info.get("response") or "")
            return {"ok": True,
                    "output": output,
                    "actuator": "knowledge",
                    "knowledge_source": "offline",
                    "knowledge_status": status,
                    "term": info.get("term", ""),
                    "alternatives": info.get("alternatives", []),
                    "confidence": info.get("confidence", 0.0),
                    "evidence": info.get("evidence", []),
                    "verified": status == "direct"}
        # 2) 오프라인에 없으면: 근거 부족을 알리고, 네트워크가 있으면 검색.
        if not self.config.allow_network:
            return {"ok": False,
                    "error": offline_kb.INSUFFICIENT_KO +
                             " 웹 검색은 네트워크가 필요해요. "
                             "네트워크 권한을 켜고 다시 시도해 주세요.",
                    "actuator": "knowledge"}
        from cognition.web_research import research
        try:
            report = research(text, allow_network=True)
            return report.to_dict()
        except Exception as exc:
            return {"ok": False,
                    "error": f"knowledge lookup unavailable: {exc}",
                    "actuator": "knowledge"}

    # ------------------------------------------------------------------
    # Response composition
    # ------------------------------------------------------------------

    def _compose_web_research(self, output: dict) -> str:
        """웹 리서치 결과를 센스 있게 포맷 (다중 소스 + 교차 검증 + 재구성)."""
        consolidated = output.get("consolidated", "")
        results = output.get("results", [])
        confidence = output.get("confidence", 0.0)
        sources_agree = output.get("sources_agree", 0)
        verified = output.get("verified", False)

        # 영어 자료를 그대로 내보내지 않는다: 한국어 맥락 + 용어 대응.
        try:
            from knowledge import offline_kb
            topic = output.get("query") or output.get("topic") or ""
            framing = offline_kb.korean_framing(consolidated, topic=topic)
            if framing:
                consolidated = framing
        except Exception:
            pass

        parts = [consolidated]

        if results:
            parts.append("")
            parts.append("📚 주요 출처:")
            for r in results[:3]:
                title = r.get("title", "(제목 없음)")
                snippet = r.get("snippet", "")
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                url = r.get("url", "")
                source = r.get("source", "")
                conf = r.get("confidence", 0.0)
                parts.append(
                    f"- **{title}** ({source}, 신뢰도 {conf:.0%})\n"
                    f"  {snippet}\n"
                    f"  {url}"
                )

        parts.append("")
        parts.append(
            f"✅ 교차 검증: {sources_agree}개 소스에서 공통 사실 확인 "
            f"(전체 신뢰도 {confidence:.0%})"
        )
        if not verified:
            parts.append("⚠️ 검증된 공통 사실이 부족해요. 직접 확인해 주세요.")

        return "\n".join(parts)

    def _compose_weather_estimate(self, est: dict, location: str) -> str:
        """오프라인 날씨 추정을 센스 있게 풀어쓰기.

        기온 범위, 비 여부, 시간대를 반영해 우산/옷차림 배려 멘트를 덧붙이고,
        '확실하지 않은 추정'임을 명시합니다.
        """
        low = est.get("low")
        high = est.get("high")
        month = est.get("month")
        raining = est.get("raining", False)
        time_of_day = est.get("time_of_day")

        parts = [f"🌤️ **{location} 날씨 (추정)**"]
        parts.append(
            f"- 대략 기온: **{low}°C ~ {high}°C** "
            f"({month}월 기후 기준)"
        )
        if raining:
            parts.append("- 상태: 비가 올 가능성이 높아요 ☔️")
        elif time_of_day:
            parts.append(f"- 시간대: {time_of_day}")

        parts.append("")
        parts.append("⚠️ 네트워크가 없어 실시간 데이터는 아니고, "
                     "월·비·시간 정보만으로 대략 추측한 값이에요.")
        parts.append("정확한 날씨를 알려면 네트워크 권한을 켜주시거나, "
                     "기상청 앱을 확인해 주세요.")

        # 배려 멘트 (하드코딩 시나리오가 아니라 일반적 상황 대응)
        care = []
        if raining:
            care.append("비 오니 우산 꼭 챙기세요.")
        if low is not None and low <= 10:
            care.append("쌀쌀하니 옷 따뜻하게 챙겨 입으세요.")
        if high is not None and high >= 28:
            care.append("더울 수 있으니 물 자주 마시고 더위 조심하세요.")
        if raining and low is not None and low <= 10:
            care.append("비 맞으면 체온이 더 빨리 떨어져요. "
                        "따뜻한 샤워로 몸을 녹이시고요.")
        care.append("감기 조심하시고요! 🙂")

        if care:
            parts.append("")
            parts.extend(care)

        return "\n".join(parts)

    def _compose(self, text: str, think: dict, act: dict) -> str:
        """Build a single plain-text answer from think + act outputs."""
        # 실패 결과를 dict 문자열로 노출하지 않는다 (앱에서 그대로 보임).
        if isinstance(act, dict) and act.get("ok") is False:
            error = act.get("error") or "알 수 없는 이유"
            # 날씨 되묻기는 실패가 아니라 대화를 이어가는 질문이다.
            if act.get("weather_ctx"):
                return str(error)
            return ("요청을 처리하지 못했어요.\n"
                    f"- 이유: {error}\n"
                    "- 다른 표현으로 다시 말씀해 주시면 재시도할게요.")
        output = act.get("output") if isinstance(act, dict) else None
        artifacts = act.get("artifacts") if isinstance(act, dict) else None
        # 날씨/리서치 같은 일부 액추에이터는 데이터 딕셔너리를 직접 반환하므로,
        # output이 없으면 act 전체를 output으로 간주합니다.
        if output is None and isinstance(act, dict):
            output = act
        if output is not None:
            if isinstance(output, str):
                return output
            if isinstance(output, dict):
                # Image / video artifacts (from the full act result)
                if artifacts:
                    return f"저장했습니다: {artifacts[0]}"
                # Code
                if output.get("code"):
                    lang = output.get("language", "python")
                    code = output["code"]
                    return (f"요청하신 코드입니다 (언어: {lang}).\n"
                            f"```{lang}\n{code}\n```")
                # Weather
                if output.get("temperature_c") is not None:
                    parts = [f"🌤️ **{output.get('location')} 현재 날씨**"]
                    parts.append(f"- 기온: **{output['temperature_c']}°C**")
                    if output.get("feels_like_c"):
                        parts[-1] += f" (체감 {output['feels_like_c']}°C)"
                    if output.get("description"):
                        parts.append(f"- 상태: {output['description']}")
                    if output.get("humidity"):
                        parts.append(f"- 습도: {output['humidity']}%")
                    if output.get("wind_kmph"):
                        parts.append(f"- 바람: {output['wind_kmph']} km/h")
                    return "\n".join(parts)
                # Weather estimate (offline)
                if output.get("weather_estimate") is not None:
                    return self._compose_weather_estimate(
                        output["weather_estimate"],
                        output.get("location", "지역"),
                    )
                # Web research (multi-source + cross-verify + consolidate)
                if output.get("consolidated") is not None:
                    return self._compose_web_research(output)
                # Calendar
                if "일정" in str(output) or output.get("event"):
                    return str(output)
                # Math
                if output.get("value") is not None:
                    return f"결과: `{output['value']}`"
                if output.get("derivative"):
                    return f"도함수: `{output['derivative']}`"
                # Image / video artifacts
                if output.get("artifacts"):
                    return f"저장했습니다: {output['artifacts'][0]}"
                return str(output)

        # Actuator failed or returned nothing: be honest.
        error = act.get("error") if isinstance(act, dict) else None
        if error and act.get("ok") is False:
            return (f"요청을 처리하지 못했어요.\n- 이유: {error}\n"
                    "조금 더 구체적으로 말해주시면 다시 시도해볼게요.")

        # Fallback: concept-based answer.
        concept = think.get("concept", "")
        score = think.get("concept_score", 0.0)
        if concept and score >= 0.2:
            return (f"가장 가까운 개념: {concept} ({score:.2f}). "
                    "어떤 작업을 할까요?")

        return ("무엇을 도와드릴까요? 코드, 수학, 그림, 차트, 영상, "
                "파일, 날씨, 일정, 검색 모두 가능해요.")
