"""MobileStudio ↔ RENA-BR JSON 브리지.

앱(예: Qt/C++ MobileStudio)이 프로세스를 띄우고 줄 단위 JSON(NDJSON)으로
대화·도구 호출을 주고받는다.  LLM/외부 API 없이 로컬에서 동작한다.

프로토콜 (한 줄 = 한 요청, 한 줄 = 한 응답)::

  요청: {"id": 1, "method": "chat", "params": {"text": "안녕"}}
  응답: {"id": 1, "ok": true, "result": {...}, "protocol": "1.0"}

메서드:
  ping        {}                              준비 확인
  manifest    {}                              도구 목록 + 사용 가능 여부
  chat        {"text": "..."}                 자연어 한 턴
  tool        {"name": "...", "args": {...}}  도구 직접 호출
  status      {}                              뇌 상태/학습 현황

실행::

  python3 agent_bridge.py                 # NDJSON 서버 (stdin/stdout)
  python3 agent_bridge.py --once "안녕"    # 한 번만 처리하고 출력
  python3 agent_bridge.py --manifest       # 도구 목록만 출력
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, Optional

PROTOCOL_VERSION = "1.0"


class AgentBridge:
    """JSON 직렬화 가능한 도구/대화 인터페이스."""

    def __init__(self, data_dir: Optional[str] = None,
                 allow_network: bool = False,
                 auto_grow: bool = True,
                 dimension: int = 2048):
        import os
        from cognition.rena_agent import RenaAgent
        if not data_dir:
            data_dir = os.path.join(os.path.expanduser("~"), ".rena_bridge")
        self.agent = RenaAgent(
            data_dir=data_dir, dimension=dimension,
            auto_save=False, auto_grow=auto_grow,
            allow_network=allow_network)
        self.allow_network = bool(allow_network)

    # ------------------------------------------------------------------
    # 도구 목록
    # ------------------------------------------------------------------
    def manifest(self) -> dict:
        return {
            "protocol": PROTOCOL_VERSION,
            "agent": "RENA-BR",
            "model_free": True,
            "tools": self._tools(),
        }

    def _tools(self) -> list[dict]:
        return [
            {"name": "math.evaluate", "safety": "read",
             "description": "수식을 계산한다 (사칙연산·거듭제곱·함수).",
             "args": {"expression": "str"}, "available": True},
            {"name": "code.generate", "safety": "read",
             "description": "요청에서 알고리즘을 추론해 코드를 생성한다.",
             "args": {"prompt": "str", "language": "str?"},
             "available": True},
            {"name": "code.verify", "safety": "read",
             "description": "생성된 함수를 테스트 케이스로 검증한다.",
             "args": {"code": "str", "function": "str",
                      "cases": "[{args: list, expected: any}]"},
             "available": True},
            {"name": "image.generate", "safety": "write",
             "description": "프롬프트로 이미지(PNG)를 생성해 저장한다.",
             "args": {"prompt": "str", "width": "int?", "height": "int?"},
             "available": True},
            {"name": "image.chart", "safety": "write",
             "description": "값 목록으로 막대/선/파이 차트를 그린다.",
             "args": {"chart": "dict"}, "available": True},
            {"name": "video.generate", "safety": "write",
             "description": "짧은 애니메이션(GIF/영상)을 생성한다.",
             "args": {"prompt": "str", "seconds": "num?"},
             "available": True},
            {"name": "audio.speak", "safety": "write",
             "description": "텍스트를 음성 파일(WAV)로 합성한다.",
             "args": {"text": "str", "voice": "str?"},
             "available": self._audio_available()},
            {"name": "file.read", "safety": "read",
             "description": "허용된 경로의 텍스트 파일을 읽는다.",
             "args": {"path": "str"}, "available": True},
            {"name": "file.write", "safety": "write",
             "description": "허용된 경로에 텍스트를 쓴다.",
             "args": {"path": "str", "text": "str"}, "available": True},
            {"name": "file.list", "safety": "read",
             "description": "디렉터리의 항목을 나열한다.",
             "args": {"path": "str?"}, "available": True},
            {"name": "data.aggregate", "safety": "read",
             "description": "행 목록에 대해 합계·평균 같은 집계를 한다.",
             "args": {"rows": "list", "query": "str"}, "available": True},
            {"name": "calendar.add", "safety": "write",
             "description": "자연어로 일정을 추가한다.",
             "args": {"text": "str"}, "available": True},
            {"name": "calendar.query", "safety": "read",
             "description": "자연어로 일정을 조회한다.",
             "args": {"text": "str"}, "available": True},
            {"name": "weather.estimate", "safety": "read",
             "description": "네트워크 없이 월·강수·시간대로 기온을 추정한다.",
             "args": {"location": "str", "month": "int?",
                      "raining": "bool?", "time_of_day": "str?"},
             "available": True},
            {"name": "knowledge.lookup", "safety": "read",
             "description": "오프라인 지식베이스에서 개념을 찾는다.",
             "args": {"query": "str"}, "available": True},
            {"name": "knowledge.synonyms", "safety": "read",
             "description": "분포 통계 사전에서 유사어를 찾는다.",
             "args": {"word": "str", "k": "int?"},
             "available": self._lexicon_available()},
            {"name": "web.research", "safety": "read",
             "description": "다중 소스 검색 + 교차 검증 (네트워크 필요).",
             "args": {"query": "str"}, "available": self.allow_network},
            {"name": "shell.run", "safety": "write",
             "description": "셸 명령 실행 (이 빌드에서는 노출되지 않음).",
             "args": {"command": "str"}, "available": False},
            {"name": "system.status", "safety": "read",
             "description": "뇌/학습 상태를 조회한다.",
             "args": {}, "available": True},
            {"name": "system.cleanup", "safety": "write",
             "description": "학습 중 생긴 임시 산출물을 정리한다.",
             "args": {}, "available": True},
        ]

    @staticmethod
    def _lexicon_available() -> bool:
        try:
            from knowledge import lexicon
            return lexicon.load() is not None
        except Exception:
            return False

    @staticmethod
    def _audio_available() -> bool:
        try:
            from media.audio import speech
            return bool(speech.SPEECH_AVAILABLE)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 호출
    # ------------------------------------------------------------------
    def call(self, name: str, args: Optional[dict] = None) -> dict:
        args = dict(args or {})
        handler = getattr(self, "_tool_" + name.replace(".", "_"), None)
        if handler is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            result = handler(args)
        except Exception as exc:
            return {"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("tool", name)
        return result

    # ---- 도구 구현 ----
    def _tool_math_evaluate(self, args: dict) -> dict:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return {"ok": False, "error": "expression required"}
        return self.agent._act_math(expression)

    def _tool_code_generate(self, args: dict) -> dict:
        from cognition.rena_agent import RenaAgent
        prompt = str(args.get("prompt", "")).strip()
        language = str(args.get("language", "")).strip().lower()
        if language and language not in prompt.lower():
            prompt = f"{language}로 {prompt}"
        return self.agent._act_code(prompt)

    def _tool_code_verify(self, args: dict) -> dict:
        result = self.agent.brain.actuators.execute(
            "code", {"operation": "verify",
                     "code": str(args.get("code", "")),
                     "function": str(args.get("function", "")),
                     "cases": list(args.get("cases") or [])})
        return result.to_dict()

    def _tool_image_generate(self, args: dict) -> dict:
        result = self.agent.brain.actuators.execute(
            "image", {"operation": "generate",
                      "prompt": str(args.get("prompt", "")),
                      "width": int(args.get("width", 480)),
                      "height": int(args.get("height", 360))})
        return result.to_dict()

    def _tool_image_chart(self, args: dict) -> dict:
        result = self.agent.brain.actuators.execute(
            "image", {"operation": "chart", "chart": dict(args.get("chart") or {})})
        return result.to_dict()

    def _tool_video_generate(self, args: dict) -> dict:
        request = {"operation": "generate",
                   "prompt": str(args.get("prompt", ""))}
        if args.get("seconds") is not None:
            request["seconds"] = float(args["seconds"])
        return self.agent.brain.actuators.execute("video", request).to_dict()

    def _tool_audio_speak(self, args: dict) -> dict:
        result = self.agent.brain.actuators.execute(
            "audio", {"operation": "speak",
                      "text": str(args.get("text", "")),
                      "voice": str(args.get("voice", "default"))})
        return result.to_dict()

    def _tool_file_read(self, args: dict) -> dict:
        return self.agent.brain.actuators.execute(
            "file", {"operation": "read", "path": str(args.get("path", ""))}
        ).to_dict()

    def _tool_file_write(self, args: dict) -> dict:
        return self.agent.brain.actuators.execute(
            "file", {"operation": "write", "path": str(args.get("path", "")),
                     "text": str(args.get("text", ""))}).to_dict()

    def _tool_file_list(self, args: dict) -> dict:
        return self.agent.brain.actuators.execute(
            "file", {"operation": "list", "path": str(args.get("path", ""))}
        ).to_dict()

    def _tool_data_aggregate(self, args: dict) -> dict:
        return self.agent.brain.actuators.execute(
            "data", {"rows": list(args.get("rows") or []),
                     "query": str(args.get("query", ""))}).to_dict()

    def _tool_calendar_add(self, args: dict) -> dict:
        return self.agent._act_calendar(str(args.get("text", "")))

    def _tool_calendar_query(self, args: dict) -> dict:
        return self.agent._act_calendar(str(args.get("text", "")))

    def _tool_weather_estimate(self, args: dict) -> dict:
        from cognition.weather import estimate_temperature_c
        month = args.get("month")
        estimate = estimate_temperature_c(
            int(month) if month is not None else None,
            raining=bool(args.get("raining", False)),
            time_of_day=args.get("time_of_day"))
        return {"ok": True, "output": estimate,
                "location": str(args.get("location", "지역")),
                "estimated": True, "verified": False}

    def _tool_knowledge_lookup(self, args: dict) -> dict:
        from knowledge import offline_kb
        query = str(args.get("query", ""))
        entry = offline_kb.lookup(query)
        if entry is None:
            return {"ok": False, "error": "not_in_offline_kb",
                    "query": query}
        return {"ok": True, "output": offline_kb.render(entry),
                "entry": entry.to_dict()}

    def _tool_knowledge_synonyms(self, args: dict) -> dict:
        from knowledge import lexicon
        lex = lexicon.load()
        if lex is None:
            return {"ok": False, "error": "lexicon_not_built"}
        word = str(args.get("word", ""))
        k = int(args.get("k", 8))
        return {"ok": True, "word": word,
                "neighbors": [{"word": w, "score": s}
                              for w, s in lex.neighbors(word, k=k)],
                "similarity_note": "분포 통계 기반 (신경망 아님)"}

    def _tool_web_research(self, args: dict) -> dict:
        if not self.allow_network:
            return {"ok": False, "error": "network_disabled"}
        from cognition.web_research import research
        report = research(str(args.get("query", "")), allow_network=True)
        return report.to_dict()

    def _tool_shell_run(self, args: dict) -> dict:
        return {"ok": False,
                "error": "shell actuator not exposed in this build"}

    def _tool_system_status(self, args: dict) -> dict:
        return {"ok": True, "output": {
            "status": self.agent.status(),
            "learning": self.agent.learn_report(),
        }}

    def _tool_system_cleanup(self, args: dict) -> dict:
        return {"ok": True, "output": self.agent.cleanup_sandbox()}

    # ------------------------------------------------------------------
    # 프로토콜
    # ------------------------------------------------------------------
    def handle(self, request: Any) -> dict:
        if not isinstance(request, dict):
            return {"id": None, "ok": False, "protocol": PROTOCOL_VERSION,
                    "error": "request must be a JSON object"}
        request_id = request.get("id")
        method = str(request.get("method", ""))
        params = request.get("params") or {}
        try:
            if method == "ping":
                result: Any = {"ready": True, "protocol": PROTOCOL_VERSION}
            elif method == "manifest":
                result = self.manifest()
            elif method == "chat":
                result = self.agent.respond_json(str(params.get("text", "")))
            elif method == "tool":
                result = self.call(str(params.get("name", "")),
                                   params.get("args") or {})
                # 도구 자체가 실패했으면 앱이 알 수 있게 바깥 ok 도 false.
                if isinstance(result, dict) and result.get("ok") is False:
                    return {"id": request_id, "ok": False,
                            "protocol": PROTOCOL_VERSION,
                            "error": str(result.get("error") or "tool failed"),
                            "result": result}
            elif method == "status":
                result = self._tool_system_status({})["output"]
            elif method == "shutdown":
                self.close()
                result = {"stopped": True}
            else:
                return {"id": request_id, "ok": False,
                        "protocol": PROTOCOL_VERSION,
                        "error": f"unknown method: {method}"}
        except Exception as exc:
            return {"id": request_id, "ok": False,
                    "protocol": PROTOCOL_VERSION,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(limit=3)}
        return {"id": request_id, "ok": True, "protocol": PROTOCOL_VERSION,
                "result": result}

    def close(self) -> None:
        try:
            self.agent.close()
        except Exception:
            pass

    def serve(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                response = {"id": None, "ok": False,
                            "protocol": PROTOCOL_VERSION,
                            "error": f"invalid json: {exc}"}
            else:
                response = self.handle(request)
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()
            if isinstance(request, dict) and request.get("method") == "shutdown":
                break
        self.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MobileStudio ↔ RENA-BR bridge")
    parser.add_argument("--once", metavar="TEXT",
                        help="한 턴만 처리하고 결과를 JSON으로 출력")
    parser.add_argument("--manifest", action="store_true",
                        help="도구 목록만 출력")
    parser.add_argument("--tool", metavar="NAME", help="도구 이름")
    parser.add_argument("--args", metavar="JSON", default="{}",
                        help="도구 인자 JSON")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--no-grow", action="store_true")
    args = parser.parse_args(argv)

    bridge = AgentBridge(data_dir=args.data_dir,
                         allow_network=args.allow_network,
                         auto_grow=not args.no_grow)
    try:
        if args.manifest:
            print(json.dumps(bridge.manifest(), ensure_ascii=False, indent=2))
            return 0
        if args.tool:
            try:
                tool_args = json.loads(args.args)
            except json.JSONDecodeError as exc:
                print(json.dumps({"ok": False, "error": f"bad --args: {exc}"},
                                 ensure_ascii=False))
                return 2
            print(json.dumps(bridge.call(args.tool, tool_args),
                             ensure_ascii=False, indent=2))
            return 0
        if args.once is not None:
            print(json.dumps(bridge.handle(
                {"id": 1, "method": "chat", "params": {"text": args.once}}),
                ensure_ascii=False, indent=2))
            return 0
        bridge.serve()
        return 0
    finally:
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
