"""Response Generator - Orchestrates response planning and realization.

This module replaces the old template-based response_generator.py and
dynamic_response.py with a semantic pipeline.
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from brain.semantic_representation import (
    ResponseRepresentation, SemanticRepresentation, Language
)
from brain.response_planner_new import ResponsePlanner, PlanningContext
from brain.language_realizer import get_realizer, realize_response
from brain.think_engine import ReasoningState
from brain.concept_store import ConceptStore
from brain.relation_store import RelationStore
from brain.memory_store import MemoryStore
from brain.capability_registry import CapabilityRegistry


@dataclass
class GenerationContext:
    """Context for response generation."""
    semantic: SemanticRepresentation
    reasoning: ReasoningState | None = None
    conversation_history: list[dict] = None
    current_project: dict = None
    user_model: dict = None
    # What the generation ACTUALLY produced (name, operations, verified):
    # the response describes this, never a placeholder theme.
    artifact: dict | None = None


class ResponseGenerator:
    """Generates natural language responses from semantic representations."""
    
    def __init__(
        self,
        concept_store: ConceptStore,
        relation_store: RelationStore,
        memory_store: MemoryStore,
        capability_registry: CapabilityRegistry,
        self_knowledge: "SelfKnowledgeInterface"
    ):
        self.planner = ResponsePlanner(
            concept_store, relation_store, memory_store,
            capability_registry, self_knowledge
        )
        self.self_knowledge = self_knowledge
    
    def generate(
        self,
        semantic: SemanticRepresentation,
        reasoning: ReasoningState = None,
        conversation_history: list[dict] = None,
        current_project: dict = None,
        user_model: dict = None,
        artifact: dict = None
    ) -> str:
        """Generate natural language response from semantic representation."""
        
        # Plan the response
        response_repr = self.planner.plan(
            semantic=semantic,
            reasoning=reasoning,
            conversation_history=conversation_history,
            current_project=current_project,
            user_model=user_model,
            artifact=artifact
        )
        
        # Realize into target language
        text = realize_response(response_repr)
        
        return text
    
    def generate_with_debug(
        self,
        semantic: SemanticRepresentation,
        reasoning: ReasoningState = None,
        conversation_history: list[dict] = None,
        current_project: dict = None,
        user_model: dict = None,
        artifact: dict = None
    ) -> tuple[str, ResponseRepresentation]:
        """Generate response with debug info (returns both text and representation)."""
        
        response_repr = self.planner.plan(
            semantic=semantic,
            reasoning=reasoning,
            conversation_history=conversation_history,
            current_project=current_project,
            user_model=user_model,
            artifact=artifact
        )
        
        text = realize_response(response_repr)
        
        return text, response_repr


class SelfKnowledgeInterface:
    """Interface for self-knowledge (implemented by SelfKnowledge class)."""
    
    def get_identity(self) -> dict:
        raise NotImplementedError
    
    def get_capabilities(self) -> list[str]:
        raise NotImplementedError
    
    def get_architecture_description(self) -> str:
        raise NotImplementedError
    
    def get_supported_languages(self) -> list[str]:
        raise NotImplementedError
    
    def get_code_knowledge(self, topic: str) -> dict | None:
        raise NotImplementedError


def create_response_generator(
    concept_store: ConceptStore,
    relation_store: RelationStore,
    memory_store: MemoryStore,
    capability_registry: CapabilityRegistry,
    self_knowledge: SelfKnowledgeInterface
) -> ResponseGenerator:
    """Factory function to create a ResponseGenerator."""
    return ResponseGenerator(
        concept_store, relation_store, memory_store,
        capability_registry, self_knowledge
    )