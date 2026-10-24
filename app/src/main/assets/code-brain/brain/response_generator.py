"""Response Generator - formats ResponseContent into natural language."""
from __future__ import annotations
from brain.response_planner import ResponseContent
from brain.self_knowledge import SelfKnowledge
from brain.capability_registry import CapabilityRegistry


class ResponseGenerator:
    def __init__(self, self_knowledge: SelfKnowledge, capability_registry: CapabilityRegistry):
        self.self_knowledge = self_knowledge
        self.capability_registry = capability_registry

        self._formatters = {
            "ko": self._format_korean,
            "en": self._format_english,
            "ja": self._format_japanese,
            "zh": self._format_chinese,
        }

    def generate(self, content: ResponseContent) -> str:
        formatter = self._formatters.get(content.language, self._format_english)
        return formatter(content)

    def _format_korean(self, c: ResponseContent) -> str:
        if c.response_type == "greeting":
            return self._greeting(c, "안녕하세요", "저는", "입니다", "무엇을 도와드릴까요?")
        if c.response_type == "self_introduction":
            return self._self_intro(c)
        if c.response_type == "capabilities":
            return self._capabilities(c)
        if c.response_type == "architecture":
            return self._architecture(c)
        if c.response_type == "code_generation":
            return self._code_generation(c)
        if c.response_type == "goodbye":
            return self._goodbye(c, "안녕히 가세요", "또 만나요")
        if c.response_type == "code_knowledge":
            return self._code_knowledge(c)
        if c.response_type == "project_question":
            return self._project_question(c)
        if c.response_type == "general_question":
            return self._general_question(c)
        return self._casual(c, "안녕하세요", "저는", "입니다")

    def _format_english(self, c: ResponseContent) -> str:
        if c.response_type == "greeting":
            return self._greeting(c, "Hello", "I'm", "today?", "How can I help you")
        if c.response_type == "self_introduction":
            return self._self_intro(c)
        if c.response_type == "capabilities":
            return self._capabilities(c)
        if c.response_type == "architecture":
            return self._architecture(c)
        if c.response_type == "code_generation":
            return self._code_generation(c)
        if c.response_type == "goodbye":
            return self._goodbye(c, "Goodbye", "See you later")
        if c.response_type == "code_knowledge":
            return self._code_knowledge(c)
        if c.response_type == "project_question":
            return self._project_question(c)
        if c.response_type == "general_question":
            return self._general_question(c)
        return self._casual(c, "Hi", "I'm", "today?")

    def _format_japanese(self, c: ResponseContent) -> str:
        if c.response_type == "greeting":
            return self._greeting(c, "こんにちは", "私は", "です", "何かお手伝いできますか")
        if c.response_type == "self_introduction":
            return self._self_intro(c)
        if c.response_type == "capabilities":
            return self._capabilities(c)
        if c.response_type == "architecture":
            return self._architecture(c)
        if c.response_type == "code_generation":
            return self._code_generation(c)
        if c.response_type == "goodbye":
            return self._goodbye(c, "さようなら", "またお越しください")
        if c.response_type == "code_knowledge":
            return self._code_knowledge(c)
        if c.response_type == "project_question":
            return self._project_question(c)
        if c.response_type == "general_question":
            return self._general_question(c)
        return self._casual(c, "こんにちは", "私は", "です")

    def _format_chinese(self, c: ResponseContent) -> str:
        if c.response_type == "greeting":
            return self._greeting(c, "你好", "我是", "", "有什么可以帮您的吗")
        if c.response_type == "self_introduction":
            return self._self_intro(c)
        if c.response_type == "capabilities":
            return self._capabilities(c)
        if c.response_type == "architecture":
            return self._architecture(c)
        if c.response_type == "code_generation":
            return self._code_generation(c)
        if c.response_type == "goodbye":
            return self._goodbye(c, "再见", "随时欢迎再来")
        if c.response_type == "code_knowledge":
            return self._code_knowledge(c)
        if c.response_type == "project_question":
            return self._project_question(c)
        if c.response_type == "general_question":
            return self._general_question(c)
        return self._casual(c, "你好", "我是", "")

    def _greeting(self, c: ResponseContent, hello: str, im: str, name_suffix: str, closing: str) -> str:
        purpose = c.self_purpose if c.self_purpose else c.self_type
        if c.detail_level == "detailed":
            return f"{hello}! {im} {c.self_name}{name_suffix}\n{purpose}\n{closing}"
        return f"{hello}! {im} {c.self_name}{name_suffix} {purpose}\n{closing}"

    def _self_intro(self, c: ResponseContent) -> str:
        lines = [f"{c.self_name}입니다." if c.language == "ko" else f"I'm {c.self_name}." if c.language == "en" else f"私は{c.self_name}です。" if c.language == "ja" else f"我是{c.self_name}。"]

        if c.self_architecture:
            if c.language == "ko":
                lines.append(f"제 아키텍처는 {c.self_architecture}입니다.")
            elif c.language == "en":
                lines.append(f"My architecture: {c.self_architecture}")
            elif c.language == "ja":
                lines.append(f"アーキテクチャ: {c.self_architecture}")
            else:
                lines.append(f"架构: {c.self_architecture}")

        if c.self_philosophy:
            if c.language == "ko":
                lines.append(f"제 철학은 {c.self_philosophy}입니다.")
            elif c.language == "en":
                lines.append(f"My philosophy: {c.self_philosophy}")
            elif c.language == "ja":
                lines.append(f"哲学: {c.self_philosophy}")
            else:
                lines.append(f"理念: {c.self_philosophy}")

        if c.detail_level == "detailed":
            if c.capability_categories:
                cat_str = ", ".join(c.capability_categories)
                if c.language == "ko":
                    lines.append(f"\n주요 기능: {cat_str}")
                elif c.language == "en":
                    lines.append(f"\nMain capabilities: {cat_str}")
                elif c.language == "ja":
                    lines.append(f"\n主要機能: {cat_str}")
                else:
                    lines.append(f"\n主要功能: {cat_str}")

            if c.supported_languages:
                lang_str = ", ".join(c.supported_languages)
                if c.language == "ko":
                    lines.append(f"지원 언어: {lang_str}")
                elif c.language == "en":
                    lines.append(f"Supported languages: {lang_str}")
                elif c.language == "ja":
                    lines.append(f"対応言語: {lang_str}")
                else:
                    lines.append(f"支持语言: {lang_str}")
        else:
            if c.capability_categories:
                cat_str = ", ".join(c.capability_categories)
                if c.language == "ko":
                    lines.append(f"다음 기능을 수행할 수 있습니다: {cat_str}")
                elif c.language == "en":
                    lines.append(f"I can help with: {cat_str}")
                elif c.language == "ja":
                    lines.append(f"対応機能: {cat_str}")
                else:
                    lines.append(f"支持功能: {cat_str}")

        return "\n".join(lines)

    def _capabilities(self, c: ResponseContent) -> str:
        if c.language == "ko":
            header = "저는 다양한 기능을 수행할 수 있습니다."
            ask = "구체적으로 무엇을 도와드릴까요?"
        elif c.language == "en":
            header = "I can help with various tasks."
            ask = "What would you like me to do?"
        elif c.language == "ja":
            header = "様々な機能をサポートしています。"
            ask = "何をお手伝いしましょうか？"
        else:
            header = "我可以帮助多种任务。"
            ask = "有什么可以帮您的吗？"

        lines = [header]

        if c.detail_level == "detailed" and c.capabilities:
            if c.language == "ko":
                lines.append("\n현재 활성화된 기능:")
            elif c.language == "en":
                lines.append("\nActive capabilities:")
            elif c.language == "ja":
                lines.append("\n対応機能:")
            else:
                lines.append("\n支持功能:")
            for cap in c.capabilities:
                lines.append(f"- {cap}")
        elif c.capability_categories:
            if c.language == "ko":
                cat_names = {"code_generation": "코드 생성", "project_generation": "프로젝트 생성",
                             "debugging": "디버깅", "explanation": "코드 설명", "learning": "학습",
                             "memory": "기억", "refactoring": "리팩토링", "verification": "검증"}
                named = [cat_names.get(x, x) for x in c.capability_categories]
                lines.append(f"주요 기능: {', '.join(named)}")
            elif c.language == "en":
                lines.append(f"Main capabilities: {', '.join(c.capability_categories)}")
            elif c.language == "ja":
                lines.append(f"主要機能: {', '.join(c.capability_categories)}")
            else:
                lines.append(f"主要功能: {', '.join(c.capability_categories)}")

        lines.append(ask)
        return "\n".join(lines)

    def _architecture(self, c: ResponseContent) -> str:
        if c.language == "ko":
            header = f"저는 {c.self_architecture} 기반으로 동작합니다."
            detail_header = "\n상세 구조:"
            process_header = "\n처리 과정:"
        elif c.language == "en":
            header = f"I work with {c.self_architecture}."
            detail_header = "\nArchitecture details:"
            process_header = "\nProcess:"
        elif c.language == "ja":
            header = f"私は{c.self_architecture}ベースで動作します。"
            detail_header = "\nアーキテクチャ詳細:"
            process_header = "\n処理プロセス:"
        else:
            header = f"我基于{c.self_architecture}架构运行。"
            detail_header = "\n架构详情:"
            process_header = "\n处理流程:"

        lines = [header]

        if c.detail_level == "detailed":
            if c.arch_components:
                lines.append(detail_header)
                for comp in c.arch_components:
                    lines.append(f"- {comp}")
            if c.arch_process_steps:
                lines.append(process_header)
                for i, step in enumerate(c.arch_process_steps, 1):
                    lines.append(f"{i}. {step}")
        else:
            if c.language == "ko":
                lines.append("사용자 입력을 분석하여 IR을 생성하고, 각 언어별 코드로 변환한 후 검증합니다.")
            elif c.language == "en":
                lines.append("I analyze user input to generate IR, convert to code for each language, then verify.")
            elif c.language == "ja":
                lines.append("ユーザー入力を分析してIRを生成し、各言語のコードに変換してから検証します。")
            else:
                lines.append("分析用户输入生成IR，转换为各语言代码后进行验证。")

        return "\n".join(lines)

    def _code_generation(self, c: ResponseContent) -> str:
        lang_str = ", ".join(c.supported_languages[:4]) if c.supported_languages else "Python, Kotlin, Java, C"

        if c.language == "ko":
            header = "코드 생성은 IR(Intermediate Representation) 기반으로 동작합니다."
            if c.detail_level == "detailed":
                steps = "\n".join(f"{i}. {s}" for i, s in enumerate(c.arch_process_steps, 1))
                return f"{header}\n\n생성 과정:\n{steps}\n\n지원 언어: {', '.join(c.supported_languages)}"
            return f"{header}\n사용자 입력을 분석하여 IR을 생성하고, {lang_str} 등의 언어로 변환합니다."
        elif c.language == "en":
            header = "Code generation uses IR (Intermediate Representation)."
            if c.detail_level == "detailed":
                steps = "\n".join(f"{i}. {s}" for i, s in enumerate(c.arch_process_steps, 1))
                return f"{header}\n\nProcess:\n{steps}\n\nSupported languages: {', '.join(c.supported_languages)}"
            return f"{header}\nI analyze input to create IR, then convert to {lang_str} and other languages."
        elif c.language == "ja":
            header = "コード生成はIR（中間表現）ベースで動作します。"
            if c.detail_level == "detailed":
                steps = "\n".join(f"{i}. {s}" for i, s in enumerate(c.arch_process_steps, 1))
                return f"{header}\n\nプロセス:\n{steps}\n\n対応言語: {', '.join(c.supported_languages)}"
            return f"{header}\nユーザー入力を分析してIRを生成し、{lang_str}などの言語に変換します。"
        else:
            header = "代码生成基于IR（中间表示）运行。"
            if c.detail_level == "detailed":
                steps = "\n".join(f"{i}. {s}" for i, s in enumerate(c.arch_process_steps, 1))
                return f"{header}\n\n处理流程:\n{steps}\n\n支持语言: {', '.join(c.supported_languages)}"
            return f"{header}\n分析用户输入生成IR，转换为{lang_str}等语言的代码。"

    def _goodbye(self, c: ResponseContent, bye: str, closing: str) -> str:
        return f"{bye}! {closing}."

    def _code_knowledge(self, c: ResponseContent) -> str:
        topic = c.code_knowledge_topic
        explanation = c.code_knowledge_explanation

        if c.language == "ko":
            header = f"**{topic}**에 대해 설명드리겠습니다."
            body = explanation if explanation else f"{topic}는 프로그래밍의 중요한 개념입니다."
            if c.detail_level == "detailed":
                return f"{header}\n\n{body}\n\n더 구체적인 언어나 상황을 말씀해 주시면 더 자세히 설명드릴 수 있습니다."
            return f"{header}\n\n{body}\n\n더 자세한 설명이 필요하시면 특정 언어나 상황을 말씀해 주세요."
        elif c.language == "en":
            header = f"Let me explain **{topic}**."
            body = explanation if explanation else f"{topic} is an important programming concept."
            if c.detail_level == "detailed":
                return f"{header}\n\n{body}\n\nLet me know if you'd like examples in a specific language."
            return f"{header}\n\n{body}\n\nFor more details, please specify a language or use case."
        elif c.language == "ja":
            header = f"**{topic}**について説明します。"
            body = explanation if explanation else f"{topic}はプログラミングの重要な概念です。"
            if c.detail_level == "detailed":
                return f"{header}\n\n{body}\n\n特定の言語での例が必要でしたらお知らせください。"
            return f"{header}\n\n{body}\n\n詳細が必要でしたら言語や使用ケースを指定してください。"
        else:
            header = f"让我解释**{topic}**。"
            body = explanation if explanation else f"{topic}是编程的重要概念。"
            if c.detail_level == "detailed":
                return f"{header}\n\n{body}\n\n如需特定语言的示例，请告诉我。"
            return f"{header}\n\n{body}\n\n如需更多详情，请指定语言或使用场景。"

    def _project_question(self, c: ResponseContent) -> str:
        if c.language == "ko":
            lines = ["네, 프로젝트를 만들 수 있습니다!", "\n지원하는 프로젝트 유형:"]
            for p in c.project_types:
                lines.append(f"- {p}")
            lines.append("\n어떤 종류의 프로젝트를 만들고 싶으신가요?")
            return "\n".join(lines)
        elif c.language == "en":
            lines = ["Yes, I can create projects!", "\nSupported project types:"]
            for p in c.project_types:
                lines.append(f"- {p}")
            lines.append("\nWhat kind of project would you like to build?")
            return "\n".join(lines)
        elif c.language == "ja":
            lines = ["はい、プロジェクトを作成できます！", "\n対応プロジェクトタイプ:"]
            for p in c.project_types:
                lines.append(f"- {p}")
            lines.append("\nどのようなプロジェクトを作りたいですか？")
            return "\n".join(lines)
        else:
            lines = ["是的，我可以创建项目！", "\n支持的项目类型:"]
            for p in c.project_types:
                lines.append(f"- {p}")
            lines.append("\n您想创建什么类型的项目？")
            return "\n".join(lines)

    def _general_question(self, c: ResponseContent) -> str:
        if c.language == "ko":
            return "좋은 질문입니다!\n저는 코드 생성, 프로젝트 생성, 디버깅, 코드 설명을 도와드릴 수 있습니다.\n더 구체적으로 어떤 것이 궁금하신가요?"
        elif c.language == "en":
            return "That's a great question!\nI can help with code generation, project creation, debugging, and code explanation.\nCould you be more specific about what you'd like to know?"
        elif c.language == "ja":
            return "良い質問です！\nコード生成、プロジェクト作成、デバッグ、コード説明をサポートします。\nさらに詳しく知りたいことはありますか？"
        else:
            return "好问题！\n我可以帮助代码生成、项目创建、调试和代码解释。\n您能具体说明想知道什么吗？"

    def _casual(self, c: ResponseContent, hello: str, im: str, name_suffix: str) -> str:
        purpose = c.self_purpose if c.self_purpose else "코드 생성, 프로젝트, 디버깅 등을 도와드릴 수 있습니다" if c.language == "ko" else "I can help with code generation, projects, debugging, and more" if c.language == "en" else "コード生成やプロジェクト作成をサポートします" if c.language == "ja" else "我可以帮助代码生成、项目创建等"
        ask = "무엇을 도와드릴까요?" if c.language == "ko" else "What would you like to do?" if c.language == "en" else "何をしましょうか？" if c.language == "ja" else "有什么可以帮您的吗？"
        return f"{hello}! {im} {c.self_name}{name_suffix} {purpose}\n{ask}"
