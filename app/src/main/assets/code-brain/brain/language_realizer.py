"""Language Realizers - Convert sentence plans to natural language.

This module implements grammar realization for multiple languages.
Each realizer handles morphology, syntax, and language-specific rules.
"""
from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import re

from brain.semantic_representation import (
    Language, Proposition, Predication, Argument, Modifier,
    DiscourseTree, DiscourseNode, RhetoricalRelation,
    ResponseRepresentation, ToneSpec, VerbosityLevel, AudienceModel,
    FocusInfo
)


class Case(Enum):
    NOMINATIVE = "nom"
    ACCUSATIVE = "acc"
    DATIVE = "dat"
    GENITIVE = "gen"
    TOPIC = "top"
    SUBJECT = "subj"
    OBJECT = "obj"


class Politeness(Enum):
    CASUAL = "casual"      # casual form
    POLITE = "polite"      # polite form
    FORMAL = "formal"      # formal form
    HONORIFIC = "honorific" # honorific form


@dataclass
class RealizationContext:
    """Context for language realization."""
    language: Language
    tone: 'ToneSpec'
    verbosity: VerbosityLevel
    audience: 'AudienceModel'
    politeness: Politeness = Politeness.POLITE


class LanguageRealizer(ABC):
    """Abstract base class for language realizers."""
    
    def __init__(self, language: Language):
        self.language = language
    
    @abstractmethod
    def realize(self, response: 'ResponseRepresentation') -> str:
        """Realize a response representation into natural language."""
        pass
    
    @abstractmethod
    def realize_proposition(self, prop: 'Proposition', ctx: RealizationContext) -> str:
        """Realize a single proposition into a sentence."""
        pass
    
    @abstractmethod
    def realize_predication(self, pred: 'Predication', ctx: RealizationContext) -> str:
        """Realize a predication into a clause."""
        pass
    
    @abstractmethod
    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        """Format a discourse tree into text."""
        pass


# ============================================================
# Korean Realizer
# ============================================================

class KoreanRealizer(LanguageRealizer):
    """Korean language realizer with grammar rules."""
    
    def __init__(self):
        super().__init__(Language.KOREAN)
        self._init_morphology()
    
    def _init_morphology(self):
        """Initialize Korean morphology rules."""
        # Verb endings by politeness
        self.verb_endings = {
            Politeness.CASUAL: {"declarative": "아/어", "interrogative": "아/어?", "imperative": "아/어라"},
            Politeness.POLITE: {"declarative": "아요/어요", "interrogative": "아요/어요?", "imperative": "아요/어요"},
            Politeness.FORMAL: {"declarative": "습니다", "interrogative": "습니까?", "imperative": "십시오"},
            Politeness.HONORIFIC: {"declarative": "십니다", "interrogative": "십니까?", "imperative": "십시오"},
        }
        
        # Particles
        self.particles = {
            "subject": ["이", "가"],
            "topic": ["은", "는"],
            "object": ["을", "를"],
            "dative": ["에게", "께"],
            "instrumental": ["로", "으로"],
            "locative": ["에서", "에"],
            "genitive": ["의"],
        }
        
        # Connectives
        self.connectives = {
            "and": "그리고 ",
            "but": "하지만 ",
            "so": "그래서 ",
            "because": "왜냐하면 ",
            "then": "그런 다음 ",
            "if": "만약 ",
            "when": "~면 ",
        }
    
    def realize(self, response: ResponseRepresentation) -> str:
        """Realize full response into Korean."""
        ctx = RealizationContext(
            language=Language.KOREAN,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
            politeness=self._determine_politeness(response.tone)
        )
        
        # Realize discourse tree
        text = self.format_discourse(response.discourse_structure, ctx)
        
        # Apply final polishing
        text = self._polish(text, ctx)
        
        return text.strip()
    
    def _determine_politeness(self, tone: 'ToneSpec') -> Politeness:
        if tone.formality > 0.8:
            return Politeness.FORMAL
        elif tone.formality > 0.5:
            return Politeness.POLITE
        elif tone.formality > 0.2:
            return Politeness.POLITE
        return Politeness.CASUAL
    
    def realize_proposition(self, prop: Proposition, ctx: RealizationContext) -> str:
        """Realize a proposition into a Korean sentence."""
        if not prop.predication:
            return ""
        
        clause = self.realize_predication(prop.predication, ctx)
        
        # Add sentence-final particle based on speech act
        # (Would be determined from context)
        return clause
    
    def realize_predication(self, pred: Predication, ctx: RealizationContext) -> str:
        """Realize a predication into a Korean clause - purely grammatical."""
        predicate = pred.predicate
        arguments = pred.arguments
        
        # Build argument map
        args = {arg.role: arg.value for arg in arguments}
        
        # Korean verb forms from predicate (purely grammatical)
        verb_map = {
            "generate_code": "생성합니다",
            "generate_project": "프로젝트를 생성합니다",
            "explain": "설명합니다",
            "fix_code": "수정합니다",
            "modify_code": "수정합니다",
            "analyze_code": "분석합니다",
            "answer_question": "답변합니다",
            "greet": "인사합니다",
            "farewell": "작별합니다",
            "thank": "감사합니다",
            "introduce_self": "소개합니다",
            "inform": "알립니다",
            "provide_evidence": "근거를 제시합니다",
            "verify_code": "확인했습니다",
            "mention_alternatives": "대안을 언급합니다",
            "mention_capabilities": "기능을 언급합니다",
        }
        
        verb = verb_map.get(predicate, predicate.replace("_", " "))
        
        # Filter out arguments with only English text or placeholder values
        def _is_useful_korean_arg(arg_value: str) -> bool:
            if not arg_value:
                return False
            # Skip placeholder values
            if arg_value.strip() in ("it", "the question", "the topic", "the code", "the issue",
                                      "approach", "capabilities used", "other approaches"):
                return False
            if not re.search(r'[가-힣]', arg_value):
                # An IDENTIFIER is not English prose: a generated name
                # (process_data(), TodoApp, is_prime) is the artifact's
                # own name and stays in a Korean clause.
                if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*\(\)?', arg_value.strip()):
                    return True
                # Everything else without a Korean character is English
                # prose spliced into a Korean clause — dropped.
                return False
            return True
        
        # Build clause components (Korean SOV structure)
        parts = []
        
        # Agent (topic marker)
        agent = args.get("agent", "")
        if agent:
            # Map English "I" to appropriate Korean pronoun based on politeness
            if agent.lower() in ("i", "me"):
                agent = "저" if ctx.politeness == Politeness.POLITE else "나"
            # Use correct topic marker: 는 for no final consonant, 은 for final consonant
            def has_final_consonant(ch: str) -> bool:
                if not ch:
                    return False
                code = ord(ch)
                # Korean syllables: 0xAC00 - 0xD7A3
                if 0xAC00 <= code <= 0xD7A3:
                    return (code - 0xAC00) % 28 != 0
                return False
            if has_final_consonant(agent[-1]):
                parts.append(f"{agent}은")
            else:
                parts.append(f"{agent}는")
        
        # Condition (으)로
        if args.get("condition") and _is_useful_korean_arg(args["condition"]):
            parts.append(f"{args['condition']}(으)로")
        
        # Instrument (으)로
        if args.get("instrument") and _is_useful_korean_arg(args["instrument"]):
            parts.append(f"{args['instrument']}(으)로")
        
        # Theme (object marker)
        if args.get("theme") and _is_useful_korean_arg(args["theme"]):
            parts.append(args['theme'] +
                         self._choose_particle(args['theme'], "object"))

        # Detail
        if args.get("detail") and _is_useful_korean_arg(args["detail"]):
            parts.append(args['detail'] +
                         self._choose_particle(args['detail'], "object"))

        # Result
        if args.get("result") and _is_useful_korean_arg(args["result"]):
            parts.append(args['result'] +
                         self._choose_particle(args['result'], "object"))
        
        # Verb at end
        parts.append(verb)

        # A clause with no useful arguments (verb only, e.g. "근거를
        # 제시합니다", or bare agent+verb "저는 생성합니다") carries no
        # content — it is fixed-sentence noise, not a realization of the
        # semantics.
        has_content = any(args.get(role) and _is_useful_korean_arg(args[role])
                          for role in ("theme", "detail", "result", "instrument"))
        if len(parts) == 1 or not has_content:
            return ""

        clause = " ".join(parts)
        return clause
        ending = self.verb_endings[ctx.politeness]["declarative"]
        clause = self._attach_ending(clause, ending)
        
        return clause
    
    def _choose_particle(self, word: str, case: str) -> str:
        """Choose appropriate particle based on word ending."""
        if not word:
            return ""
        
        last_char = word[-1]
        # Check if final consonant exists (batchim)
        has_batchim = (ord(last_char) - ord('가')) % 28 != 0
        
        if case == "subject":
            return "이" if has_batchim else "가"
        elif case == "topic":
            return "은" if has_batchim else "는"
        elif case == "object":
            return "을" if has_batchim else "를"
        elif case == "instrumental":
            return "으로" if has_batchim else "로"
        return ""
    
    def _attach_ending(self, stem: str, ending: str) -> str:
        """Attach verb ending to stem (simplified)."""
        # This is simplified - real implementation would handle
        # vowel harmony, consonant assimilation, etc.
        return stem + ending
    
    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        """Format discourse tree into coherent Korean text."""
        if not tree.root:
            return ""
        
        sentences = self._traverse_discourse(tree.root, ctx)
        return " ".join(sentences)
    
    def _traverse_discourse(self, node: 'DiscourseNode', ctx: RealizationContext) -> list[str]:
        """Traverse discourse tree and realize each node."""
        sentences = []
        
        # Handle both structures: content directly on node or in nucleus
        content = node.content or (node.nucleus.content if node.nucleus else None)
        if content:
            sent = self.realize_proposition(content, ctx)
            if sent:
                sentences.append(sent)
        
        for sat in node.satellites:
            # Add connective based on relation
            connective = self._get_connective(sat.relation)
            sat_sentences = self._traverse_discourse(sat, ctx)
            for i, s in enumerate(sat_sentences):
                if i == 0 and connective:
                    s = connective + s
                sentences.append(s)
        
        return sentences
    
    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        """Get Korean connective for rhetorical relation."""
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "또한 ",
            "CONTRAST": "하지만 ",
            "SEQUENCE": "그런 다음 ",
            "CAUSE": "왜냐하면 ",
            "CONDITION": "만약 ",
        }
        return connectives.get(relation.value, "")
    
    def _polish(self, text: str, ctx: RealizationContext) -> str:
        """Final polishing of generated text."""
        # Fix spacing around particles
        text = re.sub(r'\s+([이가을를은는로으로에서에게])', r'\1', text)
        # Fix double spaces
        text = re.sub(r'\s+', ' ', text)
        # Ensure proper sentence endings
        text = text.strip()
        if text and not text[-1] in '.!?':
            text += '.'
        return text


# ============================================================
# English Realizer
# ============================================================

class EnglishRealizer(LanguageRealizer):
    """English language realizer."""
    
    def __init__(self):
        super().__init__(Language.ENGLISH)
    
    def realize(self, response: ResponseRepresentation) -> str:
        ctx = RealizationContext(
            language=Language.ENGLISH,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
            politeness=Politeness.POLITE
        )
        text = self.format_discourse(response.discourse_structure, ctx)
        return self._polish(text)
    
    def realize_proposition(self, prop: Proposition, ctx: RealizationContext) -> str:
        if not prop.predication:
            return ""
        return self.realize_predication(prop.predication, ctx)
    
    def realize_predication(self, pred: Predication, ctx: RealizationContext) -> str:
        """Realize a predication into an English clause - purely grammatical."""
        predicate = pred.predicate
        arguments = pred.arguments
        
        # Build argument map
        args = {arg.role: arg.value for arg in arguments}
        
        # English verb forms from predicate (purely grammatical)
        verb_map = {
            "generate_code": "generate",
            "generate_project": "create a project",
            "explain": "explain",
            "fix_code": "fix",
            "modify_code": "modify",
            "analyze_code": "analyze",
            "answer_question": "answer",
            "greet": "greet",
            "farewell": "say goodbye",
            "thank": "thank",
            "introduce_self": "introduce myself",
            "inform": "inform",
            "provide_evidence": "provide evidence",
            "verify_code": "verified",
            "mention_alternatives": "mention alternatives",
            "mention_capabilities": "mention capabilities",
        }
        
        verb = verb_map.get(predicate, predicate.replace("_", " "))
        
        # Build sentence components (English SVO structure)
        agent = args.get("agent", "I")
        
        obj_parts = []
        if args.get("theme"):
            obj_parts.append(args["theme"])
        if args.get("detail"):
            obj_parts.append(args["detail"])
        if args.get("result"):
            obj_parts.append(args["result"])
        
        prep_parts = []
        if args.get("instrument"):
            prep_parts.append(f"using {args['instrument']}")
        if args.get("condition"):
            prep_parts.append(f"with {args['condition']}")
        
        if obj_parts:
            sentence = f"{agent} {verb} {', '.join(obj_parts)}"
        else:
            sentence = f"{agent} {verb}"
        
        if prep_parts:
            sentence += f" {' '.join(prep_parts)}"
        
        # Add modifiers (tense, etc.)
        for mod in pred.modifiers:
            if mod.type == "tense" and mod.value == "past":
                if verb.endswith("e"):
                    verb = verb + "d"
                elif len(verb) > 1 and verb[-1] == "y" \
                        and verb[-2] not in "aeiou":
                    # consonant + y → ied (modify → modified)
                    verb = verb[:-1] + "ied"
                else:
                    verb = verb + "ed"
                sentence = sentence.replace(verb_map.get(predicate, predicate), verb)
        
        return sentence
    
    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        if not tree.root:
            return ""
        sentences = self._traverse_discourse(tree.root, ctx)
        # A clause is not a sentence until it ends: separate the
        # discourse's clauses with sentence boundaries, never spaces
        # (a run-on is not English).
        out = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if not sent.endswith(('.', '!', '?')):
                sent += "."
            out.append(sent)
        return " ".join(out)
    
    def _traverse_discourse(self, node: 'DiscourseNode', ctx: RealizationContext) -> list[str]:
        sentences = []
        # Handle both structures: content directly on node or in nucleus
        content = node.content or (node.nucleus.content if node.nucleus else None)
        if content:
            sent = self.realize_proposition(content, ctx)
            if sent:
                sentences.append(sent)
        for sat in node.satellites:
            connective = self._get_connective(sat.relation)
            sat_sents = self._traverse_discourse(sat, ctx)
            for i, s in enumerate(sat_sents):
                if i == 0 and connective:
                    s = connective + s
                sentences.append(s)
        return sentences
    
    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "Also, ",
            "CONTRAST": "However, ",
            "SEQUENCE": "Then, ",
            "CAUSE": "Because ",
            "CONDITION": "If ",
        }
        return connectives.get(relation.value, "")
    
    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and not text[-1] in '.!?':
            text += '.'
        return text


# ============================================================
# Japanese Realizer
# ============================================================

class JapaneseRealizer(LanguageRealizer):
    """Japanese language realizer."""
    
    def __init__(self):
        super().__init__(Language.JAPANESE)
    
    def realize(self, response: ResponseRepresentation) -> str:
        ctx = RealizationContext(
            language=Language.JAPANESE,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
            politeness=Politeness.POLITE
        )
        text = self.format_discourse(response.discourse_structure, ctx)
        return self._polish(text)
    
    def realize_proposition(self, prop: Proposition, ctx: RealizationContext) -> str:
        if not prop.predication:
            return ""
        return self.realize_predication(prop.predication, ctx)
    
    def realize_predication(self, pred: Predication, ctx: RealizationContext) -> str:
        predicate = pred.predicate
        arguments = pred.arguments
        
        verb_map = {
            "generate_code": "生成します",
            "generate_project": "プロジェクトを生成します",
            "explain": "説明します",
            "fix_code": "修正します",
            "modify_code": "修正します",
            "analyze_code": "分析します",
            "answer_question": "お答えします",
            "greet": "ご挨拶します",
            "inform": "お知らせします",
            "provide_evidence": "根拠を示します",
            "mention_alternatives": "代替案を挙げます",
            "mention_capabilities": "機能を挙げます",
        }
        
        verb = verb_map.get(predicate, predicate.replace("_", " "))
        
        # Japanese: SOV order
        parts = []
        
        for arg in arguments:
            if arg.role == "theme":
                parts.append(f"{arg.value}を")
            elif arg.role == "instrument":
                parts.append(f"{arg.value}で")
            elif arg.role == "condition":
                parts.append(f"{arg.value}という条件で")
            elif arg.role == "result":
                parts.append(f"{arg.value}を")
        
        parts.append(verb)
        return " ".join(parts)
    
    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        if not tree.root:
            return ""
        sentences = self._traverse_discourse(tree.root, ctx)
        return " ".join(sentences)
    
    def _traverse_discourse(self, node: 'DiscourseNode', ctx: RealizationContext) -> list[str]:
        sentences = []
        if node.content:
            sent = self.realize_proposition(node.content, ctx)
            if sent:
                sentences.append(sent)
        for sat in node.satellites:
            connective = self._get_connective(sat.relation)
            sat_sents = self._traverse_discourse(sat, ctx)
            for i, s in enumerate(sat_sents):
                if i == 0 and connective:
                    s = connective + s
                sentences.append(s)
        return sentences
    
    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "また、",
            "CONTRAST": "しかし、",
            "SEQUENCE": "それから、",
            "CAUSE": "なぜなら、",
            "CONDITION": "もし",
        }
        return connectives.get(relation.value, "")
    
    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and not text[-1] in '.!?。':
            text += '。'
        return text


# ============================================================
# Chinese Realizer
# ============================================================

class ChineseRealizer(LanguageRealizer):
    """Chinese language realizer."""
    
    def __init__(self):
        super().__init__(Language.CHINESE)
    
    def realize(self, response: ResponseRepresentation) -> str:
        ctx = RealizationContext(
            language=Language.CHINESE,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
            politeness=Politeness.POLITE
        )
        text = self.format_discourse(response.discourse_structure, ctx)
        return self._polish(text)
    
    def realize_proposition(self, prop: Proposition, ctx: RealizationContext) -> str:
        if not prop.predication:
            return ""
        return self.realize_predication(prop.predication, ctx)
    
    def realize_predication(self, pred: Predication, ctx: RealizationContext) -> str:
        predicate = pred.predicate
        arguments = pred.arguments
        
        verb_map = {
            "generate_code": "生成",
            "generate_project": "创建项目",
            "explain": "解释",
            "fix_code": "修复",
            "modify_code": "修改",
            "analyze_code": "分析",
            "answer_question": "回答",
            "greet": "问候",
            "inform": "告知",
            "provide_evidence": "提供证据",
            "mention_alternatives": "提及替代方案",
            "mention_capabilities": "提及功能",
        }
        
        verb = verb_map.get(predicate, predicate.replace("_", " "))
        
        # Chinese: SVO order
        parts = ["我"]
        
        for arg in arguments:
            if arg.role == "theme":
                parts.append(arg.value)
            elif arg.role == "instrument":
                parts.append(f"使用{arg.value}")
            elif arg.role == "condition":
                parts.append(f"在{arg.value}条件下")
            elif arg.role == "result":
                parts.append(arg.value)
        
        parts.append(verb)
        return " ".join(parts)
    
    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        if not tree.root:
            return ""
        sentences = self._traverse_discourse(tree.root, ctx)
        return " ".join(sentences)
    
    def _traverse_discourse(self, node: 'DiscourseNode', ctx: RealizationContext) -> list[str]:
        sentences = []
        if node.content:
            sent = self.realize_proposition(node.content, ctx)
            if sent:
                sentences.append(sent)
        for sat in node.satellites:
            connective = self._get_connective(sat.relation)
            sat_sents = self._traverse_discourse(sat, ctx)
            for i, s in enumerate(sat_sents):
                if i == 0 and connective:
                    s = connective + s
                sentences.append(s)
        return sentences
    
    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "此外，",
            "CONTRAST": "但是，",
            "SEQUENCE": "然后，",
            "CAUSE": "因为，",
            "CONDITION": "如果",
        }
        return connectives.get(relation.value, "")
    
    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and not text[-1] in '.!?。':
            text += '。'
        return text


# ============================================================
# Spanish Realizer
# ============================================================

class SpanishRealizer(LanguageRealizer):
    def __init__(self):
        super().__init__(Language.SPANISH)

    def realize(self, response: 'ResponseRepresentation') -> str:
        ctx = RealizationContext(
            language=self.language,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
        )
        parts = []
        for prop in response.content.propositions:
            parts.append(self.realize_proposition(prop, ctx))
        text = " ".join(parts)
        return self._polish(text)

    def realize_proposition(self, prop: 'Proposition', ctx: RealizationContext) -> str:
        parts = []
        if prop.predication:
            parts.append(self.realize_predication(prop.predication, ctx))
        return " ".join(parts)

    def realize_predication(self, pred: 'Predication', ctx: RealizationContext) -> str:
        verb = pred.predicate
        args = [a.value for a in pred.arguments if a.value]
        if args:
            return verb + " " + " ".join(args)
        return verb

    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        parts = []
        for node in tree.nodes:
            if node.proposition:
                parts.append(self.realize_proposition(node.proposition, ctx))
        return " ".join(parts)

    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "Además, ",
            "CONTRAST": "Sin embargo, ",
            "SEQUENCE": "Luego, ",
            "CAUSE": "Porque, ",
            "CONDITION": "Si ",
        }
        return connectives.get(relation.value, "")

    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text


# ============================================================
# French Realizer
# ============================================================

class FrenchRealizer(LanguageRealizer):
    def __init__(self):
        super().__init__(Language.FRENCH)

    def realize(self, response: 'ResponseRepresentation') -> str:
        ctx = RealizationContext(
            language=self.language,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
        )
        parts = []
        for prop in response.content.propositions:
            parts.append(self.realize_proposition(prop, ctx))
        text = " ".join(parts)
        return self._polish(text)

    def realize_proposition(self, prop: 'Proposition', ctx: RealizationContext) -> str:
        parts = []
        if prop.predication:
            parts.append(self.realize_predication(prop.predication, ctx))
        return " ".join(parts)

    def realize_predication(self, pred: 'Predication', ctx: RealizationContext) -> str:
        verb = pred.predicate
        args = [a.value for a in pred.arguments if a.value]
        if args:
            return verb + " " + " ".join(args)
        return verb

    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        parts = []
        for node in tree.nodes:
            if node.proposition:
                parts.append(self.realize_proposition(node.proposition, ctx))
        return " ".join(parts)

    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "De plus, ",
            "CONTRAST": "Cependant, ",
            "SEQUENCE": "Ensuite, ",
            "CAUSE": "Parce que, ",
            "CONDITION": "Si ",
        }
        return connectives.get(relation.value, "")

    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text


# ============================================================
# German Realizer
# ============================================================

class GermanRealizer(LanguageRealizer):
    def __init__(self):
        super().__init__(Language.GERMAN)

    def realize(self, response: 'ResponseRepresentation') -> str:
        ctx = RealizationContext(
            language=self.language,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
        )
        parts = []
        for prop in response.content.propositions:
            parts.append(self.realize_proposition(prop, ctx))
        text = " ".join(parts)
        return self._polish(text)

    def realize_proposition(self, prop: 'Proposition', ctx: RealizationContext) -> str:
        parts = []
        if prop.predication:
            parts.append(self.realize_predication(prop.predication, ctx))
        return " ".join(parts)

    def realize_predication(self, pred: 'Predication', ctx: RealizationContext) -> str:
        verb = pred.predicate
        args = [a.value for a in pred.arguments if a.value]
        if args:
            return verb + " " + " ".join(args)
        return verb

    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        parts = []
        for node in tree.nodes:
            if node.proposition:
                parts.append(self.realize_proposition(node.proposition, ctx))
        return " ".join(parts)

    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "Außerdem, ",
            "CONTRAST": "Allerdings, ",
            "SEQUENCE": "Dann, ",
            "CAUSE": "Weil, ",
            "CONDITION": "Wenn ",
        }
        return connectives.get(relation.value, "")

    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text


# ============================================================
# Portuguese Realizer
# ============================================================

class PortugueseRealizer(LanguageRealizer):
    def __init__(self):
        super().__init__(Language.PORTUGUESE)

    def realize(self, response: 'ResponseRepresentation') -> str:
        ctx = RealizationContext(
            language=self.language,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
        )
        parts = []
        for prop in response.content.propositions:
            parts.append(self.realize_proposition(prop, ctx))
        text = " ".join(parts)
        return self._polish(text)

    def realize_proposition(self, prop: 'Proposition', ctx: RealizationContext) -> str:
        parts = []
        if prop.predication:
            parts.append(self.realize_predication(prop.predication, ctx))
        return " ".join(parts)

    def realize_predication(self, pred: 'Predication', ctx: RealizationContext) -> str:
        verb = pred.predicate
        args = [a.value for a in pred.arguments if a.value]
        if args:
            return verb + " " + " ".join(args)
        return verb

    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        parts = []
        for node in tree.nodes:
            if node.proposition:
                parts.append(self.realize_proposition(node.proposition, ctx))
        return " ".join(parts)

    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "Além disso, ",
            "CONTRAST": "No entanto, ",
            "SEQUENCE": "Então, ",
            "CAUSE": "Porque, ",
            "CONDITION": "Se ",
        }
        return connectives.get(relation.value, "")

    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text


# ============================================================
# Italian Realizer
# ============================================================

class ItalianRealizer(LanguageRealizer):
    def __init__(self):
        super().__init__(Language.ITALIAN)

    def realize(self, response: 'ResponseRepresentation') -> str:
        ctx = RealizationContext(
            language=self.language,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
        )
        parts = []
        for prop in response.content.propositions:
            parts.append(self.realize_proposition(prop, ctx))
        text = " ".join(parts)
        return self._polish(text)

    def realize_proposition(self, prop: 'Proposition', ctx: RealizationContext) -> str:
        parts = []
        if prop.predication:
            parts.append(self.realize_predication(prop.predication, ctx))
        return " ".join(parts)

    def realize_predication(self, pred: 'Predication', ctx: RealizationContext) -> str:
        verb = pred.predicate
        args = [a.value for a in pred.arguments if a.value]
        if args:
            return verb + " " + " ".join(args)
        return verb

    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        parts = []
        for node in tree.nodes:
            if node.proposition:
                parts.append(self.realize_proposition(node.proposition, ctx))
        return " ".join(parts)

    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "Inoltre, ",
            "CONTRAST": "Tuttavia, ",
            "SEQUENCE": "Poi, ",
            "CAUSE": "Perché, ",
            "CONDITION": "Se ",
        }
        return connectives.get(relation.value, "")

    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text


# ============================================================
# Russian Realizer
# ============================================================

class RussianRealizer(LanguageRealizer):
    def __init__(self):
        super().__init__(Language.RUSSIAN)

    def realize(self, response: 'ResponseRepresentation') -> str:
        ctx = RealizationContext(
            language=self.language,
            tone=response.tone,
            verbosity=response.verbosity,
            audience=response.audience,
        )
        parts = []
        for prop in response.content.propositions:
            parts.append(self.realize_proposition(prop, ctx))
        text = " ".join(parts)
        return self._polish(text)

    def realize_proposition(self, prop: 'Proposition', ctx: RealizationContext) -> str:
        parts = []
        if prop.predication:
            parts.append(self.realize_predication(prop.predication, ctx))
        return " ".join(parts)

    def realize_predication(self, pred: 'Predication', ctx: RealizationContext) -> str:
        verb = pred.predicate
        args = [a.value for a in pred.arguments if a.value]
        if args:
            return verb + " " + " ".join(args)
        return verb

    def format_discourse(self, tree: 'DiscourseTree', ctx: RealizationContext) -> str:
        parts = []
        for node in tree.nodes:
            if node.proposition:
                parts.append(self.realize_proposition(node.proposition, ctx))
        return " ".join(parts)

    def _get_connective(self, relation: 'RhetoricalRelation') -> str:
        if not relation:
            return ""
        connectives = {
            "ELABORATION": "Кроме того, ",
            "CONTRAST": "Однако, ",
            "SEQUENCE": "Затем, ",
            "CAUSE": "Потому что, ",
            "CONDITION": "Если ",
        }
        return connectives.get(relation.value, "")

    def _polish(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text


# ============================================================
# Realizer Factory
# ============================================================

def get_realizer(language: Language) -> LanguageRealizer:
    """Get the appropriate realizer for a language."""
    realizers = {
        Language.KOREAN: KoreanRealizer,
        Language.ENGLISH: EnglishRealizer,
        Language.JAPANESE: JapaneseRealizer,
        Language.CHINESE: ChineseRealizer,
        Language.SPANISH: SpanishRealizer,
        Language.FRENCH: FrenchRealizer,
        Language.GERMAN: GermanRealizer,
        Language.PORTUGUESE: PortugueseRealizer,
        Language.ITALIAN: ItalianRealizer,
        Language.RUSSIAN: RussianRealizer,
    }
    realizer_class = realizers.get(language, EnglishRealizer)
    return realizer_class()


def realize_response(response: ResponseRepresentation) -> str:
    """Convenience function to realize a response."""
    realizer = get_realizer(response.target_language)
    return realizer.realize(response)