"""NLU Priority Hierarchy - Phase 8.

Defines the priority order for NLU decisions, ensuring that:
- Explicit user semantics take highest priority
- Keyword lists are last resort, not primary mechanism
- Retrieved knowledge never mutates user requirements

Priority order (from user spec):
  1. Explicit User Semantics
  2. Validated Semantic Inference
  3. Contextual Evidence
  4. Retrieved Knowledge
  5. Legacy Mapping
  6. Surface Keyword Match
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NLUPriority(Enum):
    """Priority levels for NLU decisions (highest → lowest)."""
    EXPLICIT_USER = 100           # User explicitly stated something
    VALIDATED_INFERENCE = 80      # Semantic analysis with high confidence
    CONTEXTUAL_EVIDENCE = 60      # Inferred from conversation context
    RETRIEVED_KNOWLEDGE = 40      # From knowledge graph retrieval
    LEGACY_MAPPING = 20           # From keyword lists / sentence mappings
    SURFACE_KEYWORD = 10          # Raw keyword match (weakest)


@dataclass
class NLUDecision:
    """A single NLU decision with priority and evidence."""
    decision_type: str       # e.g., "operation", "entity", "type", "goal"
    value: str               # the decided value
    priority: NLUPriority
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    source: str = ""         # which module produced this
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decision_type": self.decision_type,
            "value": self.value,
            "priority": self.priority.name,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
            "alternatives": self.alternatives,
        }


class NLUPriorityResolver:
    """Resolves NLU decisions using the priority hierarchy.

    When multiple sources propose different values for the same decision,
    the one with higher priority wins. Ties broken by confidence.
    """

    def __init__(self):
        self._decisions: dict[str, list[NLUDecision]] = {}

    def add_decision(self, decision: NLUDecision):
        """Register a decision from any source."""
        key = decision.decision_type
        if key not in self._decisions:
            self._decisions[key] = []
        self._decisions[key].append(decision)

    def resolve(self, decision_type: str) -> Optional[NLUDecision]:
        """Resolve the best decision for a given type."""
        candidates = self._decisions.get(decision_type, [])
        if not candidates:
            return None

        # Sort by priority (highest first), then confidence (highest first)
        candidates.sort(key=lambda d: (-d.priority.value, -d.confidence))
        return candidates[0]

    def resolve_all(self) -> dict[str, NLUDecision]:
        """Resolve best decision for all registered types."""
        result = {}
        for dtype in self._decisions:
            best = self.resolve(dtype)
            if best:
                result[dtype] = best
        return result

    def get_all_candidates(self, decision_type: str) -> list[NLUDecision]:
        """Get all candidates sorted by priority (for multi-hypothesis)."""
        candidates = self._decisions.get(decision_type, [])
        return sorted(candidates, key=lambda d: (-d.priority.value, -d.confidence))

    def would_knowledge_mutate_user_requirements(self,
                                                 retrieved_ops: list[str],
                                                 user_ops: list[str]) -> bool:
        """Check if retrieved knowledge would add operations the user didn't request.

        This is the guard for Phase 2's requirement: retrieved knowledge
        must NOT inject new operations into user requirements.
        """
        new_ops = set(retrieved_ops) - set(user_ops)
        return len(new_ops) > 0

    def filter_retrieved_knowledge(self,
                                   retrieved_concepts: list,
                                   user_semantics: dict) -> list:
        """Filter retrieved concepts to only those that support user's intent.

        Knowledge is used for:
        - Interpreting user meaning (supporting evidence)
        - Suggesting implementation approaches
        - Type/constraint validation
        - Algorithm candidate generation
        - Code generation general knowledge
        - Verification and error analysis

        Knowledge must NOT:
        - Change user requirements
        - Add new requirements
        - Override user's explicit choices
        """
        user_ops = set(user_semantics.get("operations", []))
        user_entities = set(user_semantics.get("entities", []))

        filtered = []
        for concept in retrieved_concepts:
            # Keep concepts that are relevant to user's intent
            concept_tags = set(getattr(concept, 'tags', []))
            if concept_tags & user_ops or concept_tags & user_entities:
                filtered.append(concept)
            # Keep concepts that are general utility (not domain-specific)
            elif getattr(concept, 'category', '') in ('pattern', 'language_feature'):
                filtered.append(concept)
            # Drop domain-specific concepts not matching user intent
            # (prevents "add" injection from "sum_pattern" concept)
        return filtered


# Legacy metadata annotation for keyword lists
@dataclass
class LegacyKeywordEntry:
    """Metadata for a legacy keyword list entry."""
    keyword: str
    mapped_to: str
    source: str = "legacy"
    usage: str = "compatibility"  # compatibility | regression_test | output_format
    confidence: float = 0.5
    deprecated: bool = False
    semantic_equivalents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "keyword": self.keyword,
            "mapped_to": self.mapped_to,
            "source": self.source,
            "usage": self.usage,
            "confidence": self.confidence,
        }
        if self.deprecated:
            d["deprecated"] = True
        if self.semantic_equivalents:
            d["semantic_equivalents"] = self.semantic_equivalents
        return d
