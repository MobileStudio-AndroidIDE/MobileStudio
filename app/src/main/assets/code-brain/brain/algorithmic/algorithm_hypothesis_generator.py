"""Algorithm Hypothesis Generator.

Generates multiple algorithmic hypotheses for a structured problem.
Each hypothesis represents a GENUINELY DIFFERENT algorithmic approach.

Key design:
  - NO problem_id-to-algorithm mapping
  - NO word-to-algorithm mapping
  - NO problem-sentence-to-operation mapping
  - Uses KG relations (FASTER_THAN, ALTERNATIVE_TO, REQUIRES_DATA_STRUCTURE)
  - Uses structural hints + constraint analysis + data type detection
  - Generates hypotheses by COMBINING general algorithmic primitives
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlgorithmicHypothesis:
    """A single algorithmic approach to the problem."""
    id: int
    name: str  # e.g. "two_pointer", "binary_search", "prefix_sum"
    description: str
    algorithmic_primitives: list[str]  # e.g. ["cp_two_pointer", "cp_sort"]
    data_structures: list[str]  # e.g. ["array", "hash_map"]
    time_complexity: str  # e.g. "O(n)", "O(n log n)"
    space_complexity: str  # e.g. "O(1)", "O(n)"
    confidence: float  # 0.0 - 1.0
    reasoning: str  # why this hypothesis was generated
    approach_tags: list[str] = field(default_factory=list)  # general tags


# ============================================================
# Algorithmic Primitive Registry
# Each entry describes a GENERAL algorithmic technique.
# NO per-problem references. These are domain-agnostic patterns.
# ============================================================

@dataclass
class AlgorithmicPrimitive:
    """A general, reusable algorithmic technique."""
    name: str
    description: str
    time_complexity: str
    space_complexity: str
    suitable_when: list[str]  # structural conditions (NOT problem-specific)
    data_structures: list[str]
    tags: list[str]


_ALGORITHMIC_PRIMITIVES: dict[str, AlgorithmicPrimitive] = {
    "linear_scan": AlgorithmicPrimitive(
        name="linear_scan",
        description="Single pass through data, tracking state",
        time_complexity="O(n)",
        space_complexity="O(1)",
        suitable_when=["single_pass", "find_extremum", "count"],
        data_structures=["array"],
        tags=["basic", "iteration"],
    ),
    "two_pointer": AlgorithmicPrimitive(
        name="two_pointer",
        description="Two pointers moving through sorted data or opposite ends",
        time_complexity="O(n)",
        space_complexity="O(1)",
        suitable_when=["sorted_input", "pair_search", "contiguous_subarray"],
        data_structures=["array"],
        tags=["pointer", "pair", "subarray"],
    ),
    "sliding_window": AlgorithmicPrimitive(
        name="sliding_window",
        description="Fixed or variable window over contiguous elements",
        time_complexity="O(n)",
        space_complexity="O(1)",
        suitable_when=["contiguous_subarray", "window", "substring"],
        data_structures=["array", "string"],
        tags=["window", "subarray", "substring"],
    ),
    "binary_search": AlgorithmicPrimitive(
        name="binary_search",
        description="Search by repeatedly halving the search space",
        time_complexity="O(log n)",
        space_complexity="O(1)",
        suitable_when=["sorted_input", "monotonic", "search_space"],
        data_structures=["array"],
        tags=["search", "logarithmic", "monotonic"],
    ),
    "prefix_sum": AlgorithmicPrimitive(
        name="prefix_sum",
        description="Precompute cumulative sums for O(1) range queries",
        time_complexity="O(n)",
        space_complexity="O(n)",
        suitable_when=["range_sum", "subarray_sum", "cumulative"],
        data_structures=["array"],
        tags=["cumulative", "range", "sum"],
    ),
    "hash_map": AlgorithmicPrimitive(
        name="hash_map",
        description="Use hash table for O(1) lookups",
        time_complexity="O(n)",
        space_complexity="O(n)",
        suitable_when=["frequency", "lookup", "pair_sum", "two_sum"],
        data_structures=["hash_map"],
        tags=["hash", "lookup", "frequency"],
    ),
    "sorting": AlgorithmicPrimitive(
        name="sorting",
        description="Sort data first, then apply linear scan or two-pointer",
        time_complexity="O(n log n)",
        space_complexity="O(n)",
        suitable_when=["ordering", "greedy", "pair_search"],
        data_structures=["array"],
        tags=["sort", "ordering"],
    ),
    "greedy": AlgorithmicPrimitive(
        name="greedy",
        description="Make locally optimal choices at each step",
        time_complexity="O(n log n)",
        space_complexity="O(1)",
        suitable_when=["interval", "scheduling"],
        data_structures=["array"],
        tags=["greedy", "optimization"],
    ),
    "dynamic_programming": AlgorithmicPrimitive(
        name="dynamic_programming",
        description="Break problem into overlapping subproblems, store results",
        time_complexity="O(n*m)",
        space_complexity="O(n*m)",
        suitable_when=["overlapping_subproblems", "optimal_substructure",
                        "subsequence", "knapsack"],
        data_structures=["array", "matrix"],
        tags=["dp", "optimization", "recurrence"],
    ),
    "bfs": AlgorithmicPrimitive(
        name="bfs",
        description="Breadth-first search on graph/grid",
        time_complexity="O(V + E)",
        space_complexity="O(V)",
        suitable_when=["graph", "grid", "shortest_path", "level_order"],
        data_structures=["queue", "graph", "matrix"],
        tags=["graph", "traversal", "level"],
    ),
    "dfs": AlgorithmicPrimitive(
        name="dfs",
        description="Depth-first search on graph/grid",
        time_complexity="O(V + E)",
        space_complexity="O(V)",
        suitable_when=["graph", "backtracking", "connected_components",
                        "cycle_detection"],
        data_structures=["stack", "graph", "matrix"],
        tags=["graph", "traversal", "backtrack"],
    ),
    "bit_manipulation": AlgorithmicPrimitive(
        name="bit_manipulation",
        description="Use bitwise operations for efficient computation",
        time_complexity="O(1) per op",
        space_complexity="O(1)",
        suitable_when=["bitwise", "subset", "mask"],
        data_structures=["int"],
        tags=["bit", "mask", "subset"],
    ),
    "math_formula": AlgorithmicPrimitive(
        name="math_formula",
        description="Direct mathematical formula or number theory",
        time_complexity="O(1)",
        space_complexity="O(1)",
        suitable_when=["closed_form", "number_theory", "combinatorics"],
        data_structures=["int"],
        tags=["math", "formula", "number_theory"],
    ),
    "stack_monotonic": AlgorithmicPrimitive(
        name="stack_monotonic",
        description="Monotonic stack for next greater/smaller element",
        time_complexity="O(n)",
        space_complexity="O(n)",
        suitable_when=["next_greater", "next_smaller", "histogram"],
        data_structures=["stack"],
        tags=["stack", "monotonic", "next_element"],
    ),
    "counting": AlgorithmicPrimitive(
        name="counting",
        description="Count occurrences or enumerate valid combinations",
        time_complexity="O(n)",
        space_complexity="O(n)",
        suitable_when=["count", "frequency", "enumerate"],
        data_structures=["hash_map", "array"],
        tags=["count", "frequency"],
    ),
    "find_max": AlgorithmicPrimitive(
        name="find_max",
        description="Find the maximum value by linear scan",
        time_complexity="O(n)",
        space_complexity="O(1)",
        suitable_when=["find_extremum", "maximum", "largest"],
        data_structures=["array"],
        tags=["max", "extremum", "linear"],
    ),
    "find_min": AlgorithmicPrimitive(
        name="find_min",
        description="Find the minimum value by linear scan",
        time_complexity="O(n)",
        space_complexity="O(1)",
        suitable_when=["find_extremum", "minimum", "smallest"],
        data_structures=["array"],
        tags=["min", "extremum", "linear"],
    ),
    "simulation": AlgorithmicPrimitive(
        name="simulation",
        description="Directly simulate the described process step by step",
        time_complexity="O(n) to O(n^2)",
        space_complexity="O(n)",
        suitable_when=["process", "step_by_step", "direct"],
        data_structures=["array"],
        tags=["simulation", "direct"],
    ),
    "parity_check": AlgorithmicPrimitive(
        name="parity_check",
        description="Check even/odd property of elements using modulo",
        time_complexity="O(n)",
        space_complexity="O(1)",
        suitable_when=["parity_check", "even_odd", "divisible_by_2"],
        data_structures=["array"],
        tags=["parity", "modulo", "filter"],
    ),
}


class AlgorithmHypothesisGenerator:
    """Generates multiple algorithmic hypotheses from a StructuredProblem.

    Uses structural analysis, NOT keyword lookups or hard mappings.

    Strategy:
      1. Analyze structural hints, constraint ranges, and data types
      2. For each suitable algorithmic primitive, check applicability
      3. Generate hypotheses by combining applicable primitives
      4. Each hypothesis gets a confidence score based on fit
    """

    def __init__(self, max_hypotheses: int = 5):
        self.max_hypotheses = max_hypotheses
        self.primitives = _ALGORITHMIC_PRIMITIVES

    def generate(self, structured_problem) -> list[AlgorithmicHypothesis]:
        """Generate multiple algorithmic hypotheses for the problem.

        Args:
            structured_problem: A StructuredProblem instance.

        Returns:
            List of AlgorithmicHypothesis sorted by confidence (desc).
        """
        # Analyze the problem to get applicability signals
        signals = self._analyze_problem(structured_problem)

        # Score each primitive
        scored_primitives = self._score_primitives(signals, structured_problem)

        # Generate hypotheses from scored primitives
        hypotheses = self._build_hypotheses(scored_primitives, signals, structured_problem)

        # Sort by confidence descending
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        # Compatibility filter at the selection boundary: nothing that
        # contradicts the semantics may reach the reasoning stage.
        hypotheses = [h for h in hypotheses
                      if all(self._compatible(p, signals)
                             for p in h.algorithmic_primitives)]

        # Add fallback hypothesis if none generated — only when the
        # problem VERIFIES its structure (declared I/O, sample cases,
        # constraints or types).  A problem that states no structure
        # gets no invented approach: it honestly yields no hypotheses.
        if not hypotheses and self._structure_verified(structured_problem):
            hypotheses.append(AlgorithmicHypothesis(
                id=1,
                name="simulation",
                description="General simulation approach - read input and process step by step",
                algorithmic_primitives=["simulation"],
                data_structures=["array"],
                time_complexity="O(n)",
                space_complexity="O(n)",
                confidence=0.5,
                reasoning="No specific pattern detected, using general simulation",
                approach_tags=["simulation", "general", "fallback"],
            ))

        return hypotheses[:self.max_hypotheses]

    @staticmethod
    def _structure_verified(sp) -> bool:
        """Did the problem declare structure a generic approach could bind to?"""
        return bool(
            getattr(sp, "sample_cases", None)
            or getattr(sp, "input_variables", None)
            or getattr(sp, "output_variables", None)
            or getattr(sp, "constraints", None)
            or getattr(sp, "data_types", None)
        )

    def _analyze_problem(self, sp) -> dict:
        """Analyze a StructuredProblem to extract applicability signals.
        
        This method maps STRUCTURAL HINTS (general concepts) to signals
        that can be used to score algorithmic primitives.
        
        The hints come from StructuredProblemExtractor._detect_structural_hints()
        and represent GENERAL structural patterns, not problem-specific keywords.
        """
        signals = {
            "has_sorted_input": False,
            "has_pair_search": False,
            "has_contiguous_subarray": False,
            "has_optimization": False,
            "has_graph": False,
            "has_tree": False,
            "has_number_theory": False,
            "has_subsequence": False,
            "has_range_query": False,
            "has_frequency": False,
            "has_counting": False,
            "has_simulation": False,
            "has_monotonic": False,
            "has_bitwise": False,
            "has_sorting": False,
            "has_parity": False,
            "has_two_dimensional_iteration": False,
            "has_math_formula": False,
            "has_arithmetic_operation": False,
            "has_filtering": False,
            "has_boolean_output": False,
            "has_modular_arithmetic": False,
            "has_aggregation": False,
            "has_string_operation": False,
            "n_large": False,  # N > 10^5 suggests O(n) or O(n log n)
            "n_small": False,  # N <= 20 suggests exponential/DP
            "n_medium": False,  # 20 < N <= 1000 suggests O(n^2)
            "has_sample_cases": len(sp.sample_cases) > 0,
            "constraint_max": 0,
            "data_types": sp.data_types,
            "structural_hints": sp.structural_hints,
        }

        # Analyze constraints for N bounds
        for c in sp.constraints:
            if c.max_value:
                signals["constraint_max"] = max(signals["constraint_max"], c.max_value)
                if c.max_value > 100000:
                    signals["n_large"] = True
                if c.max_value <= 20:
                    signals["n_small"] = True
                if 20 < c.max_value <= 1000:
                    signals["n_medium"] = True

        # Map structural hints (GENERAL concepts) to signals
        hint_set = set(sp.structural_hints)
        
        # Data structure signals
        if "sorting" in hint_set:
            signals["has_sorting"] = True
        if "sorted_input" in hint_set:
            signals["has_sorted_input"] = True
        if "pair_search" in hint_set:
            signals["has_pair_search"] = True
        if "contiguous_subarray" in hint_set:
            signals["has_contiguous_subarray"] = True
        if "graph" in hint_set:
            signals["has_graph"] = True
        if "subsequence" in hint_set:
            signals["has_subsequence"] = True
        if "dynamic_programming" in hint_set:
            signals["has_subsequence"] = True  # DP often for subsequences
        if "prefix_sum" in hint_set:
            signals["has_range_query"] = True
        
        # Operation signals
        if "optimization_minimize" in hint_set or "optimization_maximize" in hint_set:
            signals["has_optimization"] = True
        if "counting" in hint_set:
            signals["has_counting"] = True
        if "parity_check" in hint_set:
            signals["has_parity"] = True
        if "aggregation" in hint_set:
            signals["has_aggregation"] = True
        if "filtering" in hint_set:
            signals["has_filtering"] = True
        if "boolean_output" in hint_set:
            signals["has_boolean_output"] = True
        if "modular_arithmetic" in hint_set:
            signals["has_modular_arithmetic"] = True
        
        # Iteration signals
        if "two_dimensional_iteration" in hint_set:
            signals["has_two_dimensional_iteration"] = True
        if "iteration" in hint_set:
            signals["has_simulation"] = True
        
        # Math signals
        if "math_formula" in hint_set:
            signals["has_math_formula"] = True
        if "arithmetic_operation" in hint_set:
            signals["has_arithmetic_operation"] = True
        
        # String signals
        if "string_concatenation" in hint_set or "string_reversal" in hint_set:
            signals["has_string_operation"] = True
        if "counting" in hint_set or "counting_with_conditions" in hint_set:
            signals["has_counting"] = True

        # Data type analysis
        type_set = set(sp.data_types)
        if "graph" in type_set:
            signals["has_graph"] = True
        if "tree" in type_set:
            signals["has_tree"] = True
        if "matrix" in type_set:
            signals["has_grid"] = True

        # Statement analysis for additional signals
        stmt = sp.statement.lower()
        if "count" in stmt or "number of" in stmt or "몇" in stmt:
            signals["has_counting"] = True
        if "frequency" in stmt or "most common" in stmt or "최빈" in stmt:
            signals["has_frequency"] = True
        if "simulate" in stmt or "process" in stmt or "시뮬" in stmt:
            signals["has_simulation"] = True
        if "sum" in stmt or "합" in stmt:
            signals["has_aggregate_all"] = True
        if "max" in stmt or "largest" in stmt or "최대" in stmt:
            signals["has_optimization"] = True
            signals["has_optimization_maximize"] = True
        if "min" in stmt or "smallest" in stmt or "최소" in stmt:
            signals["has_optimization"] = True
            signals["has_optimization_minimize"] = True

        # Structural hint vocabulary the semantic enrichment contributes:
        # the objective's DIRECTION and the presence of string operations
        # are structure, so they drive signals the same way any other
        # structural hint does (never a sentence keyword).
        if "optimization" in hint_set or "optimization_maximize" in hint_set \
                or "optimization_minimize" in hint_set:
            signals["has_optimization"] = True
        if "optimization_minimize" in hint_set:
            signals["has_optimization_minimize"] = True
        if "optimization_maximize" in hint_set:
            signals["has_optimization_maximize"] = True
        if "string_operation" in hint_set or "string_domain" in hint_set:
            signals["has_string_operation"] = True

        # The DECLARED element domain: which techniques can apply at all.
        # A graph traversal over List<String>, or bit manipulation over
        # strings, is a contradiction of the declared types — an absence
        # of a declaration is NOT a contradiction (unknown means allowed).
        declared = {str(t).lower() for t in (sp.data_types or [])}
        numeric = {"integer", "int", "number", "float", "double"}
        if "string" in declared and not (declared & numeric):
            signals["element_domain"] = "string"
        elif declared & numeric and "string" not in declared:
            signals["element_domain"] = "number"
        elif declared & {"graph", "tree"}:
            signals["element_domain"] = "graph"
        else:
            signals["element_domain"] = ""

        return signals

    # Techniques defined ONLY over a domain: when the problem's declared
    # element domain is a different one, they are not candidates — a
    # contradiction of the semantics, never an absence of evidence.
    _DOMAIN_CONTRADICTIONS = {
        # over strings: arithmetic, bits, index-ordered search and graph
        # traversal are undefined on the declared element type
        "string": {"bit_manipulation", "math_formula", "binary_search",
                   "bfs", "dfs", "parity_check", "prefix_sum"},
        # over numbers: nothing in the registry is string-only
        "number": set(),
        # over a graph structure: numeric VALUES may label the nodes, so
        # no primitive is excluded by the structure alone — only a
        # contradicting ELEMENT domain excludes anything.
        "graph": set(),
    }

    def _contradicted(self, name: str, signals: dict) -> bool:
        """Is this primitive excluded by a DECLARED element domain?

        Only a positive declaration excludes: an undeclared domain
        excludes nothing (unknown is not a contradiction).
        """
        domain = signals.get("element_domain") or ""
        if name in self._DOMAIN_CONTRADICTIONS.get(domain, set()):
            return True
        # A declared DIRECTION excludes the opposite extremum: an
        # objective to maximize has no use for a min-selection candidate
        # (and symmetrically), unless BOTH directions are declared.
        max_only = (signals.get("has_optimization_maximize")
                    and not signals.get("has_optimization_minimize"))
        min_only = (signals.get("has_optimization_minimize")
                    and not signals.get("has_optimization_maximize"))
        if max_only and name == "find_min":
            return True
        if min_only and name == "find_max":
            return True
        return False

    def _compatible(self, name: str, signals: dict) -> bool:
        """Compatibility evaluation: a candidate must not contradict the
        semantics (declared types, structure, objective, constraints)."""
        return not self._contradicted(name, signals)

    def _score_primitives(self, signals: dict, sp) -> list[tuple[str, float]]:
        """Score each algorithmic primitive based on problem signals.

        Candidates are filtered for semantic compatibility BEFORE scoring:
        a primitive the declared domain excludes is never a candidate.
        """
        scored = []

        for name, prim in self.primitives.items():
            if not self._compatible(name, signals):
                continue
            score = 0.0

            for condition in prim.suitable_when:
                if condition == "parity_check" and signals.get("has_parity"):
                    score += 0.9
                elif condition in ("sorted_input",) and signals["has_sorted_input"]:
                    score += 0.8
                elif condition in ("pair_search",) and signals["has_pair_search"]:
                    score += 0.8
                elif condition in ("contiguous_subarray", "window", "substring") and signals["has_contiguous_subarray"]:
                    score += 0.7
                elif condition in ("range_sum", "subarray_sum", "cumulative") and signals["has_range_query"]:
                    score += 0.8
                elif condition in ("graph", "grid", "shortest_path", "level_order") and signals["has_graph"]:
                    score += 0.9
                elif condition in ("backtracking", "connected_components", "cycle_detection") and signals["has_graph"]:
                    score += 0.8
                elif condition in ("tree",) and signals["has_tree"]:
                    score += 0.9
                elif condition in ("number_theory", "combinatorics") and signals["has_number_theory"]:
                    score += 0.8
                elif condition in ("overlapping_subproblems", "optimal_substructure",
                                   "subsequence", "knapsack") and signals["has_subsequence"]:
                    score += 0.8
                elif condition in ("frequency", "lookup", "pair_sum", "two_sum") and signals["has_frequency"]:
                    score += 0.7
                elif condition in ("count", "frequency", "enumerate") and signals["has_counting"]:
                    score += 0.8
                elif condition in ("process", "step_by_step", "direct") and signals["has_simulation"]:
                    score += 0.5
                elif condition in ("find_extremum",) and signals["has_optimization"]:
                    score += 0.8
                elif condition in ("optimization", "interval", "scheduling") and signals["has_optimization"]:
                    score += 0.6
                elif condition in ("next_greater", "next_smaller", "histogram") and signals["has_monotonic"]:
                    score += 0.7

            # Direct name-based scoring for common patterns
            if prim.name == "counting" and signals.get("has_counting"):
                score += 0.8
            if prim.name == "parity_check" and signals.get("has_parity"):
                score += 0.9
            if prim.name == "find_max" and (
                    signals.get("has_optimization_maximize")
                    or (signals.get("has_optimization")
                        and not signals.get("has_optimization_minimize"))):
                score += 0.9
            if prim.name == "find_min" and (
                    signals.get("has_optimization_minimize")
                    or (signals.get("has_optimization")
                        and not signals.get("has_optimization_maximize"))):
                score += 0.9
            if prim.name == "linear_scan" and signals.get("has_optimization"):
                score += 0.5
            if prim.name == "sorting" and signals.get("has_sorting"):
                score += 0.9
            if prim.name == "simulation" and signals.get("has_simulation"):
                score += 0.5

            # Grid/math patterns - these are high-confidence for specific structures
            if signals.get("has_two_dimensional_iteration") and signals.get("has_aggregation"):
                if prim.name == "simulation":
                    score += 0.8  # 2D iteration + aggregation = simulate grid generation
                if prim.name == "math_formula":
                    score += 0.7  # 2D iteration + aggregation = math formula
            if signals.get("has_two_dimensional_iteration") and signals.get("has_filtering"):
                if prim.name == "simulation":
                    score += 0.7  # 2D iteration + filtering = filter while simulating
            if signals.get("has_math_formula") and signals.get("has_arithmetic_operation"):
                if prim.name == "math_formula":
                    score += 0.8  # Arithmetic expression = math formula

            # Boolean output patterns
            if signals.get("has_boolean_output"):
                if prim.name == "sorting":
                    score += 0.6  # Sort + check condition
                if prim.name == "simulation":
                    score += 0.5  # Simulate and check

            # Modular arithmetic patterns
            if signals.get("has_modular_arithmetic"):
                if prim.name == "dynamic_programming":
                    score += 0.7  # Modular = often DP

            # String operation patterns
            if signals.get("has_string_operation"):
                if prim.name == "simulation":
                    score += 0.6  # String operations often need simulation

            # Complexity penalty: prefer simpler algorithms when N is large
            if signals["n_large"] and "O(n^2)" in prim.time_complexity:
                score *= 0.1
            if signals["n_large"] and "O(n*m)" in prim.time_complexity:
                score *= 0.3
            if signals["n_small"] and name == "simulation":
                score += 0.3  # simulation is fine for small N

            if score > 0:
                scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _build_hypotheses(
        self,
        scored_primitives: list[tuple[str, float]],
        signals: dict,
        sp,
    ) -> list[AlgorithmicHypothesis]:
        """Build algorithmic hypotheses from scored primitives."""
        hypotheses = []
        hyp_id = 1

        # Generate individual hypotheses from top primitives
        for prim_name, score in scored_primitives[:6]:
            prim = self.primitives[prim_name]
            h = AlgorithmicHypothesis(
                id=hyp_id,
                name=prim_name,
                description=prim.description,
                algorithmic_primitives=[prim_name],
                data_structures=prim.data_structures,
                time_complexity=prim.time_complexity,
                space_complexity=prim.space_complexity,
                confidence=min(score, 1.0),
                reasoning=f"Structural analysis suggests {prim_name} "
                         f"(score={score:.2f}, conditions met)",
                approach_tags=prim.tags,
            )
            hypotheses.append(h)
            hyp_id += 1

        # Generate combined hypotheses for common patterns
        # Pattern: sort + two_pointer (every member compatible already —
        # scored_primitives is compatibility-filtered)
        has_sort = any(p[0] == "sorting" for p in scored_primitives)
        has_tp = any(p[0] == "two_pointer" for p in scored_primitives)
        if has_sort and has_tp:
            h = AlgorithmicHypothesis(
                id=hyp_id,
                name="sort_then_two_pointer",
                description="Sort data, then use two-pointer technique",
                algorithmic_primitives=["sorting", "two_pointer"],
                data_structures=["array"],
                time_complexity="O(n log n)",
                space_complexity="O(n)",
                confidence=0.7,
                reasoning="Combination: sorting enables two-pointer on pair search",
                approach_tags=["sort", "pointer", "combination"],
            )
            hypotheses.append(h)
            hyp_id += 1

        # Pattern: prefix_sum + hash_map
        has_ps = any(p[0] == "prefix_sum" for p in scored_primitives)
        has_hm = any(p[0] == "hash_map" for p in scored_primitives)
        if has_ps and has_hm:
            h = AlgorithmicHypothesis(
                id=hyp_id,
                name="prefix_sum_hash",
                description="Prefix sum with hash map for subarray sums",
                algorithmic_primitives=["prefix_sum", "hash_map"],
                data_structures=["array", "hash_map"],
                time_complexity="O(n)",
                space_complexity="O(n)",
                confidence=0.7,
                reasoning="Combination: prefix sum + hash for subarray sum queries",
                approach_tags=["prefix", "hash", "combination"],
            )
            hypotheses.append(h)
            hyp_id += 1

        # Pattern: BFS/DFS for graph
        has_bfs = any(p[0] == "bfs" for p in scored_primitives)
        has_dfs = any(p[0] == "dfs" for p in scored_primitives)
        if has_bfs and has_dfs:
            h = AlgorithmicHypothesis(
                id=hyp_id,
                name="graph_traversal",
                description="Graph traversal with BFS/DFS",
                algorithmic_primitives=["bfs", "dfs"],
                data_structures=["graph", "queue"],
                time_complexity="O(V + E)",
                space_complexity="O(V)",
                confidence=0.8,
                reasoning="Problem involves graph structure, use traversal",
                approach_tags=["graph", "traversal", "combination"],
            )
            hypotheses.append(h)
            hyp_id += 1

        # Pattern: 2D iteration + aggregation (e.g., grid sum)
        if signals.get("has_two_dimensional_iteration") and signals.get("has_aggregation"):
            h = AlgorithmicHypothesis(
                id=hyp_id,
                name="grid_aggregate",
                description="Iterate over 2D structure and aggregate values",
                algorithmic_primitives=["simulation", "linear_scan"],
                data_structures=["array"],
                time_complexity="O(n*m)",
                space_complexity="O(1)",
                confidence=0.85,
                reasoning="2D iteration with aggregation = nested loops with accumulation",
                approach_tags=["iteration", "aggregation", "combination"],
            )
            hypotheses.append(h)
            hyp_id += 1

        # Pattern: arithmetic formula
        if signals.get("has_math_formula") and signals.get("has_arithmetic_operation"):
            h = AlgorithmicHypothesis(
                id=hyp_id,
                name="arithmetic_formula",
                description="Direct arithmetic computation",
                algorithmic_primitives=["math_formula"],
                data_structures=["int"],
                time_complexity="O(1)",
                space_complexity="O(1)",
                confidence=0.9,
                reasoning="Arithmetic expression = direct formula computation",
                approach_tags=["math", "formula", "arithmetic"],
            )
            hypotheses.append(h)
            hyp_id += 1

        # Pattern: sorting + boolean output (e.g., check if sortable)
        if signals.get("has_sorting") and signals.get("has_boolean_output"):
            h = AlgorithmicHypothesis(
                id=hyp_id,
                name="sort_check",
                description="Sort data and check condition",
                algorithmic_primitives=["sorting", "linear_scan"],
                data_structures=["array"],
                time_complexity="O(n log n)",
                space_complexity="O(n)",
                confidence=0.8,
                reasoning="Sorting + boolean output = sort and verify condition",
                approach_tags=["sort", "check", "verification"],
            )
            hypotheses.append(h)
            hyp_id += 1

        # Pattern: simulation + filtering
        if signals.get("has_simulation") and signals.get("has_filtering"):
            h = AlgorithmicHypothesis(
                id=hyp_id,
                name="simulate_filter",
                description="Simulate process with filtering",
                algorithmic_primitives=["simulation", "linear_scan"],
                data_structures=["array"],
                time_complexity="O(n)",
                space_complexity="O(1)",
                confidence=0.75,
                reasoning="Simulation with filtering = process and exclude",
                approach_tags=["simulation", "filter", "combination"],
            )
            hypotheses.append(h)
            hyp_id += 1

        # Coverage is EVIDENCE-DRIVEN: a primitive enters the candidate
        # set only through a structural condition this problem met (or
        # a declared hint matched).  Padding the list with generic 0.3
        # families would present irrelevant strategies as evaluated
        # options — an unevidenced candidate is no candidate.  A problem
        # with no surviving candidate resolves through the
        # structure-verified fallback in generate(), never through an
        # invented approach.

        return hypotheses
