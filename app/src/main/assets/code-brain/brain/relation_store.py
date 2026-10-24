"""Relation Store - stores and queries relationships between concepts."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RelationType(Enum):
    IS_A = "is_a"  # inheritance
    IMPLEMENTS = "implements"  # interface implementation
    USES = "uses"  # dependency
    PRODUCES = "produces"  # output relationship
    REQUIRES = "requires"  # prerequisite
    CONTRADICTS = "contradicts"
    ALTERNATIVE = "alternative"  # equivalent concept in another language
    PART_OF = "part_of"  # composition
    CAUSES = "causes"  # error cause
    SOLVES = "solves"  # error solution
    SIMILAR_TO = "similar_to"
    CONTAINS = "contains"
    RETURNS = "returns"
    TAKES = "takes"  # parameter type


@dataclass
class Relation:
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, str] = field(default_factory=dict)
    weight: float = 1.0
    language: str = ""


class RelationStore:
    def __init__(self, data_dir: str = None):
        self.relations: dict[str, Relation] = {}
        self._index_by_source: dict[str, list[str]] = {}
        self._index_by_target: dict[str, list[str]] = {}
        self.data_dir = data_dir
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._load()

    def add(self, relation: Relation) -> None:
        self.relations[relation.id] = relation
        self._index_by_source.setdefault(relation.source_id, []).append(relation.id)
        self._index_by_target.setdefault(relation.target_id, []).append(relation.id)
        self._save()

    def get(self, relation_id: str) -> Optional[Relation]:
        return self.relations.get(relation_id)

    def find_relations(self, source_id: str = None, target_id: str = None,
                       relation_type: RelationType = None) -> list[Relation]:
        candidates = set()
        if source_id:
            candidates.update(self._index_by_source.get(source_id, []))
        if target_id:
            candidates.update(self._index_by_target.get(target_id, []))
        if not candidates:
            candidates = set(self.relations.keys())

        results = []
        for rid in candidates:
            r = self.relations.get(rid)
            if not r:
                continue
            if source_id and r.source_id != source_id:
                continue
            if target_id and r.target_id != target_id:
                continue
            if relation_type and r.relation_type != relation_type:
                continue
            results.append(r)
        return results

    def find_related(self, concept_id: str, relation_type: RelationType = None,
                     direction: str = "both") -> list[Relation]:
        results = []
        if direction in ("out", "both"):
            for rid in self._index_by_source.get(concept_id, []):
                r = self.relations.get(rid)
                if r and (relation_type is None or r.relation_type == relation_type):
                    results.append(r)
        if direction in ("in", "both"):
            for rid in self._index_by_target.get(concept_id, []):
                r = self.relations.get(rid)
                if r and (relation_type is None or r.relation_type == relation_type):
                    results.append(r)
        return results

    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> Optional[list[str]]:
        if start_id == end_id:
            return [start_id]
        visited = {start_id}
        queue = [[start_id]]
        while queue:
            path = queue.pop(0)
            if len(path) > max_depth:
                continue
            current = path[-1]
            neighbors = set()
            for rid in self._index_by_source.get(current, []):
                r = self.relations.get(rid)
                if r:
                    neighbors.add(r.target_id)
            for rid in self._index_by_target.get(current, []):
                r = self.relations.get(rid)
                if r:
                    neighbors.add(r.source_id)
            for neighbor in neighbors:
                if neighbor == end_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def delete(self, relation_id: str) -> bool:
        r = self.relations.pop(relation_id, None)
        if r:
            self._rebuild_indices()
            self._save()
            return True
        return False

    def _rebuild_indices(self):
        self._index_by_source.clear()
        self._index_by_target.clear()
        for rid, r in self.relations.items():
            self._index_by_source.setdefault(r.source_id, []).append(rid)
            self._index_by_target.setdefault(r.target_id, []).append(rid)

    def _save(self):
        if not self.data_dir:
            return
        path = os.path.join(self.data_dir, "relations.json")
        data = []
        for r in self.relations.values():
            d = {
                "id": r.id,
                "source_id": r.source_id,
                "target_id": r.target_id,
                "relation_type": r.relation_type.value,
                "properties": r.properties,
                "weight": r.weight,
                "language": r.language,
            }
            data.append(d)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _load(self):
        path = os.path.join(self.data_dir, "relations.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            try:
                os.replace(path, path + ".corrupt")
            except OSError:
                pass
            return
        if not isinstance(data, list):
            return
        for d in data:
            try:
                d["relation_type"] = RelationType(d["relation_type"])
                r = Relation(**d)
            except (TypeError, KeyError, ValueError):
                continue
            self.relations[r.id] = r
        self._rebuild_indices()

    def export_json(self) -> str:
        data = []
        for r in self.relations.values():
            data.append({
                "id": r.id,
                "source_id": r.source_id,
                "target_id": r.target_id,
                "relation_type": r.relation_type.value,
                "properties": r.properties,
                "weight": r.weight,
                "language": r.language,
            })
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> int:
        data = json.loads(json_str)
        count = 0
        for d in data:
            d["relation_type"] = RelationType(d["relation_type"])
            r = Relation(**d)
            self.add(r)
            count += 1
        return count


def seed_default_relations(store: RelationStore) -> None:
    """Seed the knowledge graph with relations between concepts.

    Relations encode structural knowledge:
    - REQUIRES: data/control dependencies between operations
    - PRODUCES: what an operation outputs (used by downstream ops)
    - SIMILAR_TO: alternative approaches to the same problem
    - CONTRADICTS: mutually exclusive operations
    - PART_OF: composition (op A is part of op B's pipeline)
    - USES: one concept uses another as building block
    """
    rels = [
        # ── Pipeline data flow: what each operation requires/produces ──
        # Aggregates require a sequence input
        Relation("r_avg_req", "calculate_average", "input_sequence",
                 RelationType.REQUIRES, weight=1.0),
        Relation("r_sum_req", "calculate_sum", "input_sequence",
                 RelationType.REQUIRES, weight=1.0),
        Relation("r_max_req", "find_max", "input_sequence",
                 RelationType.REQUIRES, weight=1.0),
        Relation("r_min_req", "find_min", "input_sequence",
                 RelationType.REQUIRES, weight=1.0),

        # Comparisons require an aggregate as threshold
        Relation("r_gt_req_avg", "compare_greater_than", "calculate_average",
                 RelationType.REQUIRES, weight=1.0),
        Relation("r_gt_req_sum", "compare_greater_than", "calculate_sum",
                 RelationType.REQUIRES, weight=0.8),
        Relation("r_lt_req_avg", "compare_less_than", "calculate_average",
                 RelationType.REQUIRES, weight=1.0),
        Relation("r_lt_req_sum", "compare_less_than", "calculate_sum",
                 RelationType.REQUIRES, weight=0.8),

        # Filters require predicates
        Relation("r_filter_req_cmp", "filter", "compare_greater_than",
                 RelationType.REQUIRES, weight=0.9),
        Relation("r_filter_req_clt", "filter", "compare_less_than",
                 RelationType.REQUIRES, weight=0.9),
        Relation("r_filter_req_prime", "filter", "is_prime",
                 RelationType.REQUIRES, weight=0.7),
        Relation("r_filter_req_mult", "filter", "is_multiple_of",
                 RelationType.REQUIRES, weight=0.7),

        # ── Produces: what each op outputs (used by downstream) ──
        Relation("r_avg_prod", "calculate_average", "scalar",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_sum_prod", "calculate_sum", "scalar",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_max_prod", "find_max", "scalar",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_min_prod", "find_min", "scalar",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_gt_prod", "compare_greater_than", "predicate",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_lt_prod", "compare_less_than", "predicate",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_prime_prod", "is_prime", "predicate",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_mult_prod", "is_multiple_of", "predicate",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_filter_prod", "filter", "sequence",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_dedup_prod", "remove_duplicates", "sequence",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_sdesc_prod", "sort_descending", "sequence",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_sasc_prod", "sort_ascending", "sequence",
                 RelationType.PRODUCES, weight=1.0),
        Relation("r_range_prod", "generate_range", "sequence",
                 RelationType.PRODUCES, weight=1.0),

        # ── SimilarTo: alternative approaches ──
        Relation("r_avg_sim_mean", "calculate_average", "mean_concept",
                 RelationType.SIMILAR_TO, weight=1.0),
        Relation("r_dedup_sim_unique", "remove_duplicates", "unique_concept",
                 RelationType.SIMILAR_TO, weight=1.0),
        Relation("r_sdesc_sim_rev", "sort_descending", "reverse_sequence",
                 RelationType.SIMILAR_TO, weight=0.6),
        Relation("r_sasc_sim_sort", "sort_ascending", "sort_descending",
                 RelationType.SIMILAR_TO, weight=0.5),
        Relation("r_filter_sim_sel", "filter", "filter_elements",
                 RelationType.SIMILAR_TO, weight=0.9),
        Relation("r_gt_sim_above", "compare_greater_than", "above_concept",
                 RelationType.SIMILAR_TO, weight=0.8),
        Relation("r_lt_sim_below", "compare_less_than", "below_concept",
                 RelationType.SIMILAR_TO, weight=0.8),

        # ── Contradicts: mutually exclusive operations ──
        Relation("r_prime_contra_mult", "is_prime", "is_multiple_of",
                 RelationType.CONTRADICTS, weight=0.7),
        Relation("r_sdesc_contra_sasc", "sort_descending", "sort_ascending",
                 RelationType.CONTRADICTS, weight=0.8),
        Relation("r_sdesc_contra_rev", "sort_descending", "reverse_sequence",
                 RelationType.CONTRADICTS, weight=0.5),
        Relation("r_gt_contra_lt", "compare_greater_than", "compare_less_than",
                 RelationType.CONTRADICTS, weight=0.6),

        # ── PartOf: composition (op is part of a pipeline pattern) ──
        Relation("r_avg_partof_pipeline", "calculate_average", "aggregate_filter_pipeline",
                 RelationType.PART_OF, weight=1.0),
        Relation("r_filter_partof_pipeline", "filter", "aggregate_filter_pipeline",
                 RelationType.PART_OF, weight=1.0),
        Relation("r_sdesc_partof_pipeline", "sort_descending", "aggregate_filter_pipeline",
                 RelationType.PART_OF, weight=0.9),
        Relation("r_dedup_partof_pipeline", "remove_duplicates", "aggregate_filter_pipeline",
                 RelationType.PART_OF, weight=0.8),
        Relation("r_range_partof_pipeline", "generate_range", "range_check_pipeline",
                 RelationType.PART_OF, weight=1.0),
        Relation("r_prime_partof_pipeline", "is_prime", "range_check_pipeline",
                 RelationType.PART_OF, weight=0.9),
        Relation("r_sdesc_partof_pipeline", "sort_descending", "range_check_pipeline",
                 RelationType.PART_OF, weight=0.8),

        # ── Uses: building block relationships ──
        Relation("r_filter_uses_pred", "filter", "predicate_concept",
                 RelationType.USES, weight=1.0),
        Relation("r_sdesc_uses_cmp", "sort_descending", "comparison_concept",
                 RelationType.USES, weight=0.6),
        Relation("r_sasc_uses_cmp", "sort_ascending", "comparison_concept",
                 RelationType.USES, weight=0.6),

        # ── Cross-cutting: concepts that appear in multiple patterns ──
        Relation("r_avg_cross_dedup", "calculate_average", "remove_duplicates",
                 RelationType.REQUIRES, weight=0.5),
        Relation("r_dedup_cross_avg", "remove_duplicates", "calculate_average",
                 RelationType.REQUIRES, weight=0.5),
        Relation("r_range_cross_avg", "generate_range", "calculate_average",
                 RelationType.REQUIRES, weight=0.5),
        Relation("r_avg_cross_range", "calculate_average", "generate_range",
                 RelationType.REQUIRES, weight=0.4),
        Relation("r_mult_cross_range", "is_multiple_of", "generate_range",
                 RelationType.REQUIRES, weight=0.7),
        Relation("r_range_cross_mult", "generate_range", "is_multiple_of",
                 RelationType.REQUIRES, weight=0.5),

        # ── Language-specific operation support ──
        Relation("r_avg_python", "calculate_average", "python",
                 RelationType.IMPLEMENTS, weight=1.0),
        Relation("r_avg_kotlin", "calculate_average", "kotlin",
                 RelationType.IMPLEMENTS, weight=1.0),
        Relation("r_avg_java", "calculate_average", "java",
                 RelationType.IMPLEMENTS, weight=1.0),
        Relation("r_avg_go", "calculate_average", "go",
                 RelationType.IMPLEMENTS, weight=1.0),
        Relation("r_avg_rust", "calculate_average", "rust",
                 RelationType.IMPLEMENTS, weight=1.0),
        Relation("r_filter_python", "filter", "python",
                 RelationType.IMPLEMENTS, weight=1.0),
        Relation("r_filter_kotlin", "filter", "kotlin",
                 RelationType.IMPLEMENTS, weight=1.0),
        Relation("r_sdesc_python", "sort_descending", "python",
                 RelationType.IMPLEMENTS, weight=1.0),
        Relation("r_sdesc_kotlin", "sort_descending", "kotlin",
                 RelationType.IMPLEMENTS, weight=1.0),

        # ── Error solving relationships ──
        Relation("r_null_solved_by", "null_check", "optional_type",
                 RelationType.SOLVES, weight=1.0),
        Relation("r_index_solved_by", "index_error", "bounds_check",
                 RelationType.SOLVES, weight=1.0),
        Relation("r_type_solved_by", "type_error", "type_cast",
                 RelationType.SOLVES, weight=1.0),
    ]
    for r in rels:
        store.add(r)


def seed_algorithmic_relations(store: RelationStore) -> None:
    """Seed the knowledge graph with algorithmic reasoning relations.

    These are GENERAL knowledge-graph relations:
    - REQUIRES_PRIMITIVE: problem type requires algorithmic primitive
    - ALTERNATIVE_TO: two primitives are alternative approaches
    - FASTER_THAN: one primitive has better complexity than another
    - USES_DATA_STRUCTURE: primitive uses a specific data structure
    - COMPOSES_WITH: two primitives can be combined

    NOT per-problem mappings — these encode general algorithmic knowledge.
    """
    rels = [
        # ── FASTER_THAN: complexity ordering ──
        Relation("r_alog_faster_linear", "cp_binary_search", "cp_linear_scan",
                 RelationType.SIMILAR_TO, weight=0.7,
                 properties={"note": "binary search O(log n) vs linear O(n), but requires sorted data"}),
        Relation("r_alog_sort_vs_linear", "cp_sort", "cp_linear_scan",
                 RelationType.SIMILAR_TO, weight=0.6,
                 properties={"note": "sorting O(n log n) enables pair/two-pointer O(n)"}),

        # ── ALTERNATIVE_TO: equivalent approaches ──
        Relation("r_alog_bfs_dfs", "cp_bfs", "cp_dfs",
                 RelationType.ALTERNATIVE, weight=0.8,
                 properties={"note": "both traverse graphs, BFS for shortest path, DFS for cycles"}),
        Relation("r_alog_hash_vs_sort", "cp_hash_map", "cp_sort",
                 RelationType.ALTERNATIVE, weight=0.7,
                 properties={"note": "both can solve pair/frequency problems"}),
        Relation("r_alog_greedy_vs_dp", "cp_greedy", "cp_dynamic_programming",
                 RelationType.ALTERNATIVE, weight=0.6,
                 properties={"note": "greedy is faster but not always correct; DP is general"}),

        # ── COMPOSES_WITH: combinable primitives ──
        Relation("r_alog_sort_composes_tp", "cp_sort", "cp_two_pointer",
                 RelationType.PART_OF, weight=0.8,
                 properties={"note": "sorting enables two-pointer technique"}),
        Relation("r_alog_ps_composes_hash", "cp_prefix_sum", "cp_hash_map",
                 RelationType.PART_OF, weight=0.7,
                 properties={"note": "prefix sum + hash map for subarray sum problems"}),
        Relation("r_alog_sort_composes_greedy", "cp_sort", "cp_greedy",
                 RelationType.PART_OF, weight=0.7,
                 properties={"note": "sorting is often the first step in greedy algorithms"}),
        Relation("r_alog_sort_composes_bs", "cp_sort", "cp_binary_search",
                 RelationType.PART_OF, weight=0.6,
                 properties={"note": "sorting enables binary search"}),

        # ── USES_DATA_STRUCTURE ──
        Relation("r_alog_hash_uses_dict", "cp_hash_map", "hash_map",
                 RelationType.USES, weight=1.0),
        Relation("r_alog_bfs_uses_queue", "cp_bfs", "queue",
                 RelationType.USES, weight=1.0),
        Relation("r_alog_dfs_uses_stack", "cp_dfs", "stack",
                 RelationType.USES, weight=1.0),
        Relation("r_alog_monotonic_uses_stack", "cp_monotonic_stack", "stack",
                 RelationType.USES, weight=1.0),
        Relation("r_alog_dp_uses_array", "cp_dynamic_programming", "array",
                 RelationType.USES, weight=1.0),
        Relation("r_alog_prefix_uses_array", "cp_prefix_sum", "array",
                 RelationType.USES, weight=1.0),

        # ── REQUIRES_PRIMITIVE: what problem types need ──
        Relation("r_prob_graph_requires_bfs", "graph_problem", "cp_bfs",
                 RelationType.REQUIRES, weight=0.8),
        Relation("r_prob_graph_requires_dfs", "graph_problem", "cp_dfs",
                 RelationType.REQUIRES, weight=0.8),
        Relation("r_prob_sorted_requires_bs", "sorted_search", "cp_binary_search",
                 RelationType.REQUIRES, weight=0.9),
        Relation("r_prob_pair_requires_tp", "pair_search", "cp_two_pointer",
                 RelationType.REQUIRES, weight=0.7),
        Relation("r_prob_subarray_requires_sw", "subarray_problem", "cp_sliding_window",
                 RelationType.REQUIRES, weight=0.7),
        Relation("r_prob_range_requires_ps", "range_query", "cp_prefix_sum",
                 RelationType.REQUIRES, weight=0.8),
    ]
    for r in rels:
        store.add(r)
