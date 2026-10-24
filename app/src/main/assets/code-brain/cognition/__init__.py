"""RENA-BR cognitive growth subsystem.

A model-free (no pretrained weights, no external LLM/API) cognitive
architecture built from:

* vector symbolic architecture (VSA) representations,
* concept / working / episodic memory,
* predictive and compositional reasoning,
* a typed program-synthesis laboratory,
* a verification-gated skill registry,
* an autonomous failure -> goal -> practice -> verify -> skill loop,
* verification-gated self-augmentation that composes existing primitives
  into new verified skills/macros,
* primitive invention (parametric abstraction of verified programs,
  with foundation seeds when no learned expression is available) and
  virtual sensor synthesis from existing measurement channels.

The package is deliberately additive: the existing RENA-BR pipeline
keeps working unchanged, and :class:`cognition.bridge.CognitionRuntime`
attaches this subsystem to it.
"""
from __future__ import annotations

__version__ = "1.3.0"

from cognition.config import GrowthConfig
from cognition.vsa import VSA
from cognition.memory import (ConceptMemory, WorkingMemory, EpisodicMemory,
                              Consolidator)
from cognition.predictive import PredictiveEngine
from cognition.compositional import CompositionalReasoner
from cognition.skills import Skill, SkillRegistry, VerificationGate
from cognition.growth import GrowthEngine, GrowthReport
from cognition.semantic import SemanticEncoder
from cognition.concept_space import Concept, ConceptSpace
from cognition.active import ActiveInference
from cognition.metacognition import MetaCognition
from cognition.curiosity import CuriosityEngine
from cognition.learning import (ContrastiveMemory, DevelopmentalLearner,
                                ReplayBuffer, StrategySelector)
from cognition.math_engine import (differentiate, evaluate, parse_expression,
                                   simplify, solve_linear)
from cognition.actuators import ActuatorRegistry, ActuatorResult
from cognition.rena import RenaBrain
from cognition.autonomous import AutonomousAgent, AutonomousReport
from cognition.augmentation import (AugmentationReport,
                                    SelfAugmentationEngine)
from cognition.invention import (PrimitiveInventionReport,
                                 PrimitiveInventor)
from cognition.sensors import (SensorForge, SensorInventionReport,
                               SensorRegistry, SensorSpec)
from cognition.language_engine import LanguageEngine

__all__ = [
    "GrowthConfig", "VSA", "ConceptMemory", "WorkingMemory",
    "EpisodicMemory", "Consolidator", "PredictiveEngine",
    "CompositionalReasoner", "Skill", "SkillRegistry",
    "VerificationGate", "GrowthEngine", "GrowthReport",
    "SemanticEncoder", "Concept", "ConceptSpace", "ActiveInference",
    "MetaCognition", "CuriosityEngine", "ContrastiveMemory",
    "DevelopmentalLearner", "ReplayBuffer", "StrategySelector",
    "differentiate", "evaluate", "parse_expression", "simplify",
    "solve_linear", "ActuatorRegistry", "ActuatorResult", "RenaBrain",
    "AutonomousAgent", "AutonomousReport", "AugmentationReport",
    "SelfAugmentationEngine", "PrimitiveInventionReport",
    "PrimitiveInventor", "SensorForge", "SensorInventionReport",
    "SensorRegistry", "SensorSpec",
    "LanguageEngine",
]
