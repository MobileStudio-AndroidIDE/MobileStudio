"""Response Planner - plans response content from knowledge, not templates."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from brain.semantic_understanding import SemanticRepresentation, SemanticCategory, Language
from brain.self_knowledge import SelfKnowledge
from brain.capability_registry import CapabilityRegistry


@dataclass
class ResponseContent:
    """Language-independent structured content for responses."""
    response_type: str = ""
    tone: str = "polite"
    detail_level: str = "standard"
    language: str = "en"

    self_name: str = ""
    self_type: str = ""
    self_architecture: str = ""
    self_philosophy: str = ""
    self_purpose: str = ""

    capabilities: list[str] = field(default_factory=list)
    capability_categories: list[str] = field(default_factory=list)

    arch_components: list[str] = field(default_factory=list)
    arch_process_steps: list[str] = field(default_factory=list)

    supported_languages: list[str] = field(default_factory=list)
    project_types: list[str] = field(default_factory=list)

    code_knowledge_topic: str = ""
    code_knowledge_explanation: str = ""

    limitations: list[str] = field(default_factory=list)

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "response_type": self.response_type,
            "tone": self.tone,
            "detail_level": self.detail_level,
            "language": self.language,
            "self_name": self.self_name,
            "self_architecture": self.self_architecture,
            "capabilities": self.capabilities,
            "capability_categories": self.capability_categories,
            "arch_components": self.arch_components,
            "supported_languages": self.supported_languages,
            "project_types": self.project_types,
            "code_knowledge_topic": self.code_knowledge_topic,
        }


class ResponsePlanner:
    def __init__(self, self_knowledge: SelfKnowledge, capability_registry: CapabilityRegistry):
        self.self_knowledge = self_knowledge
        self.capability_registry = capability_registry

    def plan(self, semantic: SemanticRepresentation, user_context: dict = None) -> ResponseContent:
        user_context = user_context or {}

        tone = self._determine_tone(user_context)
        detail_level = self._determine_detail_level(semantic, user_context)
        language = semantic.detected_language.value

        content = ResponseContent(
            tone=tone,
            detail_level=detail_level,
            language=language,
        )

        self._retrieve_identity(content)
        self._retrieve_capabilities(content)
        self._retrieve_architecture(content)
        self._retrieve_languages(content)
        self._retrieve_projects(content)
        self._retrieve_limitations(content)

        if SemanticCategory.GOODBYE in semantic.categories:
            content.response_type = "goodbye"
        elif SemanticCategory.ARCHITECTURE in semantic.categories or semantic.is_about_architecture:
            content.response_type = "architecture"
        elif SemanticCategory.CODE_GENERATION in semantic.categories:
            content.response_type = "code_generation"
        elif SemanticCategory.CAPABILITIES in semantic.categories:
            content.response_type = "capabilities"
        elif SemanticCategory.SELF_INTRODUCTION in semantic.categories or semantic.is_about_self:
            content.response_type = "self_introduction"
        elif semantic.is_greeting:
            content.response_type = "greeting"
        elif semantic.is_about_code:
            content.response_type = "code_knowledge"
            topic = semantic.topic or "general"
            content.code_knowledge_topic = topic
            content.code_knowledge_explanation = self._get_code_knowledge(topic)
        elif semantic.is_about_project:
            content.response_type = "project_question"
        elif semantic.is_question:
            content.response_type = "general_question"
        else:
            content.response_type = "casual"

        return content

    def _determine_tone(self, user_context: dict) -> str:
        speech_style = user_context.get("speech_style", "polite")
        if speech_style == "casual":
            return "casual"
        return "polite"

    def _determine_detail_level(self, semantic: SemanticRepresentation,
                                user_context: dict) -> str:
        if semantic.detail_request == "detailed":
            return "detailed"
        pref = user_context.get("explanation_length", "medium")
        if pref == "short":
            return "brief"
        if pref == "detailed":
            return "detailed"
        return "standard"

    def _retrieve_identity(self, content: ResponseContent):
        lang = content.language
        
        content.self_name = self.self_knowledge.get("identity", "name") or "RENA-BR"
        
        self_type = {
            "ko": "AI 코딩 어시스턴트",
            "en": "AI coding assistant",
            "ja": "AIコーディングアシスタント",
            "zh": "AI编程助手",
        }
        content.self_type = self_type.get(lang, self_type["en"])
        
        content.self_architecture = self.self_knowledge.get("identity", "architecture") or ""
        
        philosophy = {
            "ko": "No transformers, no LLM APIs, no pretrained models",
            "en": "No transformers, no LLM APIs, no pretrained models",
            "ja": "Transformerなし、LLM APIなし、事前学習済みモデルなし",
            "zh": "无Transformer、无LLM API、无预训练模型",
        }
        content.self_philosophy = philosophy.get(lang, philosophy["en"])
        
        purpose = {
            "ko": "코드 생성, 프로젝트 생성, 디버깅, 코드 설명, 학습을 돕는 AI 코딩 어시스턴트",
            "en": "AI coding assistant to help with code generation, project creation, debugging, code explanation, and learning",
            "ja": "コード生成、プロジェクト作成、デバッグ、コード説明、学習を支援するAIコーディングアシスタント",
            "zh": "帮助代码生成、项目创建、调试、代码解释和学习的AI编程助手",
        }
        content.self_purpose = purpose.get(lang, purpose["en"])

    def _retrieve_capabilities(self, content: ResponseContent):
        active = self.capability_registry.get_active()
        lang = content.language
        
        capability_names = {
            "code_generation": {
                "ko": "코드 생성",
                "en": "Code Generation",
                "ja": "コード生成",
                "zh": "代码生成",
            },
            "project_generation": {
                "ko": "프로젝트 생성",
                "en": "Project Generation",
                "ja": "プロジェクト生成",
                "zh": "项目生成",
            },
            "debugging": {
                "ko": "디버깅",
                "en": "Debugging",
                "ja": "デバッグ",
                "zh": "调试",
            },
            "code_explanation": {
                "ko": "코드 설명",
                "en": "Code Explanation",
                "ja": "コード説明",
                "zh": "代码解释",
            },
            "learning": {
                "ko": "학습",
                "en": "Learning",
                "ja": "学習",
                "zh": "学习",
            },
            "memory": {
                "ko": "기억",
                "en": "Memory",
                "ja": "記憶",
                "zh": "记忆",
            },
            "refactoring": {
                "ko": "리팩토링",
                "en": "Refactoring",
                "ja": "リファクタリング",
                "zh": "重构",
            },
            "verification": {
                "ko": "검증",
                "en": "Verification",
                "ja": "検証",
                "zh": "验证",
            },
            "android_generation": {
                "ko": "안드로이드 생성",
                "en": "Android Generation",
                "ja": "Android生成",
                "zh": "Android生成",
            },
            "web_generation": {
                "ko": "웹 생성",
                "en": "Web Generation",
                "ja": "Web生成",
                "zh": "Web生成",
            },
            "cli_generation": {
                "ko": "CLI 생성",
                "en": "CLI Generation",
                "ja": "CLI生成",
                "zh": "CLI生成",
            },
            "error_diagnosis": {
                "ko": "오류 진단",
                "en": "Error Diagnosis",
                "ja": "エラー診断",
                "zh": "错误诊断",
            },
            "auto_fix": {
                "ko": "자동 수정",
                "en": "Auto Fix",
                "ja": "自動修正",
                "zh": "自动修复",
            },
            "multi_language": {
                "ko": "다중 언어",
                "en": "Multi-language",
                "ja": "多言語",
                "zh": "多语言",
            },
        }
        
        for cap in active:
            name = capability_names.get(cap.name, {}).get(lang, cap.name)
            content.capabilities.append(name)
        
        content.capability_categories = list(set(c.category for c in active))

    def _retrieve_architecture(self, content: ResponseContent):
        lang = content.language
        
        arch_components = {
            "ko": [
                "Reasoner: 사용자 입력을 분석하여 의도와 계획을 수립",
                "Planner: IR 노드를 생성하고 실행 계획 수립",
                "Generator: IR을 각 언어별 코드로 변환",
                "Verifier: 생성된 코드의 문법과 구조 검증",
                "Knowledge Graph: 개념과 관계를 구조화하여 저장",
            ],
            "en": [
                "Reasoner: Analyzes user input to determine intent and plan",
                "Planner: Creates IR nodes and execution plan",
                "Generator: Converts IR to code in each language",
                "Verifier: Validates syntax and structure of generated code",
                "Knowledge Graph: Stores concepts and relationships in structured form",
            ],
            "ja": [
                "Reasoner: ユーザー入力を分析して意図と計画を決定",
                "Planner: IRノードを生成し実行計画を策定",
                "Generator: IRを各言語のコードに変換",
                "Verifier: 生成されたコードの構文と構造を検証",
                "Knowledge Graph: 概念と関係を構造化して保存",
            ],
            "zh": [
                "Reasoner: 分析用户输入以确定意图和计划",
                "Planner: 创建IR节点并制定执行计划",
                "Generator: 将IR转换为各语言的代码",
                "Verifier: 验证生成代码的语法和结构",
                "Knowledge Graph: 以结构化形式存储概念和关系",
            ],
        }
        
        content.arch_components = arch_components.get(lang, arch_components["en"])
        
        process_steps = {
            "ko": [
                "사용자 입력 분석 (의미 이해)",
                "IR(Intermediate Representation) 노드 생성",
                "각 언어별 코드 생성",
                "문법 검증",
            ],
            "en": [
                "User input analysis (semantic understanding)",
                "IR (Intermediate Representation) node generation",
                "Code generation for each language",
                "Syntax verification",
            ],
            "ja": [
                "ユーザー入力分析（意味理解）",
                "IR（中間表現）ノード生成",
                "各言語でのコード生成",
                "構文検証",
            ],
            "zh": [
                "用户输入分析（语义理解）",
                "IR（中间表示）节点生成",
                "各语言代码生成",
                "语法验证",
            ],
        }
        
        content.arch_process_steps = process_steps.get(lang, process_steps["en"])

    def _retrieve_languages(self, content: ResponseContent):
        lang_items = self.self_knowledge.get_category("supported_languages")
        content.supported_languages = [i.key for i in lang_items if i.key != "languages"]

    def _retrieve_projects(self, content: ResponseContent):
        lang = content.language
        
        project_types = {
            "ko": [
                "Android 프로젝트: Gradle, Manifest, Activity, Layout XML 생성",
                "웹 프로젝트: HTML, CSS, JavaScript 파일 구조 생성",
                "CLI 프로젝트: 명령줄 인터페이스 프로젝트 생성",
            ],
            "en": [
                "Android Project: Generate Gradle, Manifest, Activity, Layout XML",
                "Web Project: Generate HTML, CSS, JavaScript file structure",
                "CLI Project: Generate command-line interface projects",
            ],
            "ja": [
                "Androidプロジェクト: Gradle, Manifest, Activity, Layout XMLを生成",
                "Webプロジェクト: HTML, CSS, JavaScriptのファイル構造を生成",
                "CLIプロジェクト: コマンドラインインターフェースプロジェクトを生成",
            ],
            "zh": [
                "Android项目: 生成Gradle、Manifest、Activity、Layout XML",
                "Web项目: 生成HTML、CSS、JavaScript文件结构",
                "CLI项目: 生成命令行界面项目",
            ],
        }
        
        content.project_types = project_types.get(lang, project_types["en"])

    def _retrieve_limitations(self, content: ResponseContent):
        lang = content.language
        
        limitations = {
            "ko": [
                "Transformer 기반 모델 없음",
                "LLM API 없음",
                "사전 학습된 모델 없음",
                "Knowledge Graph + Rule Engine + Memory + Reasoning + Planning 기반",
            ],
            "en": [
                "No Transformer-based models",
                "No LLM APIs",
                "No pretrained models",
                "Based on Knowledge Graph + Rule Engine + Memory + Reasoning + Planning",
            ],
            "ja": [
                "Transformerベースのモデルなし",
                "LLM APIなし",
                "事前学習済みモデルなし",
                "Knowledge Graph + Rule Engine + Memory + Reasoning + Planningベース",
            ],
            "zh": [
                "无Transformer基础模型",
                "无LLM API",
                "无预训练模型",
                "基于Knowledge Graph + Rule Engine + Memory + Reasoning + Planning",
            ],
        }
        
        content.limitations = limitations.get(lang, limitations["en"])

    def _get_code_knowledge(self, topic: str) -> str:
        topic_lower = topic.lower()
        knowledge_base = {
            "pointer": {
                "ko": "메모리 주소를 저장하는 변수. &로 주소 획득, *로 역참조. 동적 메모리 할당에 사용.",
                "en": "A variable that stores a memory address. Use & to get address, * to dereference. Used for dynamic memory allocation.",
                "ja": "メモリアドレスを保存する変数。&でアドレス取得、*で逆参照。動的メモリ割り当てに使用。",
                "zh": "存储内存地址的变量。用&获取地址，*解引用。用于动态内存分配。",
            },
            "variable": {
                "ko": "데이터를 저장하는 이름 있는 저장소. 타입에 따라 저장 가능한 값이 달라짐.",
                "en": "A named storage for data. The type determines what values can be stored.",
                "ja": "データを保存する名前付きストレージ。タイプによって保存可能な値が異なる。",
                "zh": "用于存储数据的命名存储区。类型决定可存储的值。",
            },
            "function": {
                "ko": "특정 작업을 수행하는 코드 블록. 입력 매개변수와 반환값으로 구성.",
                "en": "A code block that performs a specific task. Composed of input parameters and a return value.",
                "ja": "特定のタスクを実行するコードブロック。入力パラメータと戻り値で構成。",
                "zh": "执行特定任务的代码块。由输入参数和返回值组成。",
            },
            "class": {
                "ko": "객체를 만들기 위한 청사진. 속성(데이터)과 메서드(동작)를 캡슐화.",
                "en": "A blueprint for creating objects. Encapsulates attributes (data) and methods (behavior).",
                "ja": "オブジェクトを作成するための青写し。属性（データ）とメソッド（動作）をカプセル化。",
                "zh": "创建对象的蓝图。封装属性（数据）和方法（行为）。",
            },
            "object": {
                "ko": "클래스의 인스턴스. 실제 메모리에 존재하며 고유한 상태를 가짐.",
                "en": "An instance of a class. Exists in memory and has its own state.",
                "ja": "クラスのインスタンス。メモリに存在し、独自の状態を持つ。",
                "zh": "类的实例。存在于内存中，具有自己的状态。",
            },
            "inheritance": {
                "ko": "부모 클래스의 특성을 자식 클래스가 상속받는 메커니즘. 코드 재사용 지원.",
                "en": "A mechanism where a child class inherits properties from a parent class. Supports code reuse.",
                "ja": "親クラスの特性を子クラスが継承するメカニズム。コード再利用をサポート。",
                "zh": "子类继承父类特性的机制。支持代码重用。",
            },
            "algorithm": {
                "ko": "문제를 해결하기 위한 단계별 절차. 시간/공간 복잡도로 성능 측정.",
                "en": "A step-by-step procedure for solving a problem. Performance measured by time/space complexity.",
                "ja": "問題を解決するためのステップバイステップの手順。時間/空間複雑度で性能を測定。",
                "zh": "解决问题的逐步过程。通过时间/空间复杂度衡量性能。",
            },
        }

        if topic_lower in knowledge_base:
            return knowledge_base[topic_lower].get("en", "")

        return f"{topic} is a programming concept. Implementation details vary by language."
