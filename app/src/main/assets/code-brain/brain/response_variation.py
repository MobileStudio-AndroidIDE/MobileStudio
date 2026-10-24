"""Non-deterministic response variation.

The same semantic content realizes into DIFFERENT (equally valid)
surface forms across runs — the response generation is
non-deterministic, and no canned answers exist: every sentence is
realized from the semantics with generation-time choices.

Variation sources (all derived from the semantic structure):
  - register: the politeness level re-rolls near the tone's formality
    (formal ↔ polite ↔ casual) — the same meaning, different register
  - connective: the rhetorical relation's connective is chosen among
    the language's connective variants
  - order: the clause order follows the discourse salience (the
    satellite-first re-roll when the relation allows it)
  - hedge: an uncertain meaning hedges (the certainty drives the
    epistemic markers); a certain meaning states directly

The RNG is module-level and re-seeded per realization call — two runs
on the same input produce different phrasings of the same meaning.
"""
from __future__ import annotations

import random

_REALIZATION_RNG = random.Random()


def roll() -> float:
    """One re-roll of the generation-time choice."""
    return _REALIZATION_RNG.random()


def choice(seq):
    """One re-roll among equally valid surface variants."""
    if not seq:
        return None
    return _REALIZATION_RNG.choice(seq)


def reseed() -> None:
    """Re-seed the generation-time choices (per conversation turn)."""
    _REALIZATION_RNG.seed()
