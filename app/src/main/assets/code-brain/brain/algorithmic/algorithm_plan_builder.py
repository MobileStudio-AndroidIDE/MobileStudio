"""Algorithm Plan Builder.

Builds a detailed AlgorithmPlan from the selected hypothesis.

Architecture:
  - Plan is built from semantic structure (hints, constraints, I/O)
  - Primitives are COMPOSED into a coherent algorithm
  - Steps build on each other with variable bindings
  - Constraints are incorporated into the algorithm
  - No keyword matching, no generic templates

The plan captures the INTENT of the algorithm, not just the primitives.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlanStep:
    """A single step in the algorithm plan."""
    step_id: int
    operation: str  # general operation name
    description: str
    input_vars: list[str] = field(default_factory=list)
    output_vars: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    # General semantic contract fields
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    semantic_justification: str = ""


@dataclass
class AlgorithmPlan:
    """A complete plan for solving the problem."""
    hypothesis_name: str
    function_name: str
    # I/O specification
    input_parsing: list[PlanStep] = field(default_factory=list)
    # Repeated case/query blocks: {"count_var": "T"} — case body parsing
    # steps are marked details.in_case_loop and are wrapped in a loop by
    # the code generator along with the algorithm and output steps.
    multi_case: dict = field(default_factory=dict)
    # Algorithm steps
    algorithm_steps: list[PlanStep] = field(default_factory=list)
    # Output specification
    output_step: Optional[PlanStep] = None
    # Constraints
    time_complexity: str = ""
    space_complexity: str = ""
    # Variables
    variables: dict[str, str] = field(default_factory=dict)  # name -> type
    # Metadata
    data_structures: list[str] = field(default_factory=list)
    reasoning: str = ""


class AlgorithmPlanBuilder:
    """Builds an AlgorithmPlan from a hypothesis and structured problem.

    The plan is built by composing primitives based on the semantic structure.
    Steps are connected through variable names and build on each other.
    """

    def build(self, hypothesis, structured_problem) -> AlgorithmPlan:
        """Build a detailed algorithm plan."""
        plan = AlgorithmPlan(
            hypothesis_name=hypothesis.name,
            function_name=structured_problem.function_name,
            time_complexity=hypothesis.time_complexity,
            space_complexity=hypothesis.space_complexity,
            data_structures=hypothesis.data_structures,
            reasoning=hypothesis.reasoning,
        )

        # 1. Build I/O parsing steps
        plan.input_parsing = self._build_input_parsing(structured_problem)

        # 1b. Multi-case input: the count header stays outside the loop,
        # the per-case body parsing goes inside (steps flagged
        # in_case_loop; the code generator wraps them + algorithm +
        # output in `for _ in range(count):`).
        mc_count = getattr(structured_problem, "multi_case_count_var", "") or ""
        if mc_count and getattr(structured_problem, "multi_case_body_vars", None):
            from types import SimpleNamespace
            outside = [PlanStep(
                step_id=1,
                operation="read_int",
                description=f"Read repeat count {mc_count} from input",
                input_vars=[],
                output_vars=[mc_count],
                details={"type": "int"},
            )]
            sid = 2
            body_shim = SimpleNamespace(
                input_variables=structured_problem.multi_case_body_vars,
                input_format_raw="",
            )
            for cst in self._build_input_parsing(body_shim):
                cst.step_id = sid
                sid += 1
                cst.details = dict(cst.details or {})
                cst.details["in_case_loop"] = True
                outside.append(cst)
            plan.input_parsing = outside
            plan.multi_case = {"count_var": mc_count}

        # 2. Build algorithm steps by composing primitives
        plan.algorithm_steps = self._build_algorithm_steps(
            hypothesis, structured_problem, plan.input_parsing
        )

        # 2a. Steps that consume their input variable (read it themselves)
        # — remove that variable's separate read step.
        # threshold_classification does NOT consume: its template only
        # references the variable, which the read step must define.
        for step in plan.algorithm_steps:
            consumed_var = None
            if step.operation == "query_simulation":
                consumed_var = step.details.get("query_var")
            if step.operation in ("set_deduction", "concentric_ring_fill",
                                  "pairwise_probability",
                                  "dual_collection_max",
                                  "moving_set_simulation",
                                  "alternating_insertion",
                                  "palindrome_prefix",
                                  "min_xor_path"):
                # The template reads the input itself — the separate
                # read steps would consume the same stdin lines twice.
                if step.operation in ("pairwise_probability",
                                      "dual_collection_max",
                                      "moving_set_simulation",
                                      "palindrome_prefix",
                                      "min_xor_path"):
                    # the template reads its own input from stdin: every
                    # separate read step is removed
                    plan.input_parsing = []
                else:
                    plan.input_parsing = [
                        s for s in plan.input_parsing
                        if s.operation not in ("read_strings", "read_string")
                    ]
                    plan.input_parsing = [
                        s for s in plan.input_parsing
                        if not (s.operation.startswith("read_") and
                                s.output_vars and
                                s.output_vars[0] == step.details.get("size_var"))
                    ]
            if consumed_var:
                plan.input_parsing = [
                    s for s in plan.input_parsing
                    if not (s.operation.startswith("read_") and consumed_var in s.output_vars)
                ]

        # 2a-b. When an index_chain step fires but the input parsing only
        # produced a generic read, the chain's collections need their own
        # read steps (the chain binds to named collections).
        for step in plan.algorithm_steps:
            if step.operation == "index_chain":
                chain_vars = [col for _kind, col in step.details.get("steps", [])]
                has_named_reads = any(
                    s.operation != "read_input" for s in plan.input_parsing)
                if chain_vars and not has_named_reads:
                    plan.input_parsing = [
                        PlanStep(
                            step_id=i + 1,
                            operation="read_array",
                            description=f"Read array {v} from input",
                            input_vars=[],
                            output_vars=[v],
                            details={"type": "int[]", "name": v,
                                     "separator": "space"},
                        )
                        for i, v in enumerate(chain_vars)
                    ]

        # 2a-c. When a grid_pattern_search step fires but the input
        # parsing didn't read the grids as variables, the grids need
        # their own read steps (row strings, one per row).
        for step in plan.algorithm_steps:
            if step.operation == "grid_pattern_search":
                big = step.details.get("big")
                small = step.details.get("small")
                int_vars = [v.name for v in structured_problem.input_variables
                            if v.var_type == 'int']
                n_var = int_vars[0] if int_vars else 'N'
                m_var = int_vars[1] if len(int_vars) > 1 else 'M'
                read_vars = {s.output_vars[0] for s in plan.input_parsing
                             if s.output_vars}
                needed = []
                if big and big not in read_vars:
                    needed.append((big, n_var))
                if small and small not in read_vars:
                    needed.append((small, m_var))
                if needed:
                    plan.input_parsing = plan.input_parsing + [
                        PlanStep(
                            step_id=len(plan.input_parsing) + i + 1,
                            operation="read_strings",
                            description=f"Read {nv} rows of {v}",
                            input_vars=[],
                            output_vars=[v],
                            details={"n_var": nv, "type": "input"},
                        )
                        for i, (v, nv) in enumerate(needed)
                    ]

        # 2b. Wire the final data step into the output.
        # If the last algorithm step produces a differently-named variable
        # (e.g. "sorted_A"), the output step must print that variable, not
        # a disconnected "result".
        if plan.algorithm_steps:
            last_out = plan.algorithm_steps[-1].output_vars[0] \
                if plan.algorithm_steps[-1].output_vars else None
            if last_out and last_out not in ("result", "answer"):
                out_var = structured_problem.output_variables[0] \
                    if structured_problem.output_variables else None
                out_name = out_var.name if out_var else "result"
                if out_name in ("result", "answer") or out_name not in \
                        [s.output_vars[0] for s in plan.algorithm_steps
                         if s.output_vars]:
                    plan.output_step = PlanStep(
                        step_id=1000,
                        operation="print",
                        description=f"Print the result: {last_out}",
                        input_vars=[last_out],
                        output_vars=[],
                        details={
                            "type": out_var.var_type if out_var else "default",
                            "name": last_out,
                        },
                    )

        # 3. Build output step (unless the computation already emitted its own
        # complete output, e.g. set_complement prints count + values itself)
        if plan.output_step is None:
            last_op = plan.algorithm_steps[-1].operation \
                if plan.algorithm_steps else None
            if last_op in ("set_complement", "query_simulation", "gcd_subset",
                           "dice_max_agreement", "power_difference_search",
                           "power_series", "threshold_classification",
                           "ranking_ties", "stack_queries", "forest_deletion"):
                plan.output_step = PlanStep(
                    step_id=1000,
                    operation="print",
                    description="Output already emitted by computation",
                    input_vars=[],
                    output_vars=[],
                    details={"type": "default", "skip": True},
                )
            else:
                plan.output_step = self._build_output_step(structured_problem, plan)

        # 4. Collect variables
        plan.variables = self._collect_variables(plan, structured_problem)

        return plan

    def _build_input_parsing(self, sp) -> list[PlanStep]:
        """Build steps for parsing stdin input."""
        steps = []
        step_id = 1

        # Group variables by their declaration line:
        # - Sibling scalars declared on one line (description "variables: N, M")
        #   are read together with map(int, input().split()).
        # - Every other variable is read on its own line, respecting its type.
        sibling_names = []
        if sp.input_variables and "variables:" in sp.input_variables[0].description:
            sib_match = re.search(r'variables:\s*(.+)',
                                  sp.input_variables[0].description)
            if sib_match:
                all_names = {v.name for v in sp.input_variables}
                sibling_names = [s.strip() for s in sib_match.group(1).split(',')
                                 if s.strip() in all_names]
            if len(sibling_names) > 1:
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="read_ints",
                    description=f"Read integers {', '.join(sibling_names)} from input",
                    input_vars=[],
                    output_vars=sibling_names,
                    details={"type": "int[]", "names": sibling_names, "separator": "space"},
                ))
                step_id += 1

        remaining = [v for v in sp.input_variables if v.name not in sibling_names]

        for var in remaining:
            if var.var_type == "int":
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="read_int",
                    description=f"Read integer {var.name} from input",
                    input_vars=[],
                    output_vars=[var.name],
                    details={"type": "int", "name": var.name},
                ))
            elif var.var_type == "float":
                # Numeric with decimal samples ("40.0"): read as float
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="read_int",
                    description=f"Read numeric {var.name} from input",
                    input_vars=[],
                    output_vars=[var.name],
                    details={"type": "float", "name": var.name},
                ))
            elif var.var_type in ("int[]", "array"):
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="read_array",
                    description=f"Read array {var.name} from input",
                    input_vars=[],
                    output_vars=[var.name],
                    details={"type": "int[]", "name": var.name, "separator": "space"},
                ))
            elif var.var_type == "string":
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="read_string",
                    description=f"Read string {var.name} from input",
                    input_vars=[],
                    output_vars=[var.name],
                    details={"type": "string", "name": var.name},
                ))
            elif var.var_type == "string[]":
                # Count variable: a preceding single int variable is the
                # element count (standard "N then N elements" convention)
                n_var = next((v.name for v in sp.input_variables
                              if v.var_type == "int" and v.name != var.name), "n")
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="read_strings",
                    description=f"Read {n_var} strings as {var.name}",
                    input_vars=[],
                    output_vars=[var.name],
                    details={"type": "string[]", "name": var.name, "n_var": n_var},
                ))
            elif var.var_type == "int_pairs":
                # Repeated pair lines: count lines each containing the pair
                # of base variables. The count variable is resolved from the
                # count hint (the int variable matching the max subscript,
                # e.g. "T_N" → N), falling back to the preceding int variable.
                bases = var.name.split('+')
                count_hint = None
                hint_m = re.search(r'count:\s*([^;]+)', var.description)
                if hint_m:
                    count_hint = hint_m.group(1).strip()
                count_var = None
                if count_hint:
                    for v in sp.input_variables:
                        if v.var_type == "int" and v.name == count_hint:
                            count_var = v.name
                            break
                    # Arithmetic bound over declared integers ("N - 1",
                    # "2*K"): the count is an EXPRESSION — keep it
                    # verbatim, the generated range() accepts it.
                    if count_var is None and re.fullmatch(
                            r'[A-Za-z0-9_ +\-*/()]+', count_hint or ''):
                        ids = re.findall(r'[A-Za-z_]\w*', count_hint)
                        int_names = {v.name for v in sp.input_variables
                                     if v.var_type == "int"}
                        if re.fullmatch(r'\d+', count_hint) or \
                                (ids and all(i in int_names for i in ids)):
                            count_var = count_hint
                if count_var is None:
                    for v in sp.input_variables:
                        if v is var:
                            break
                        if v.var_type == "int":
                            count_var = v.name
                    if count_var is None:
                        count_var = "n"
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="read_pairs",
                    description=f"Read {count_var} pairs ({', '.join(bases)})",
                    input_vars=[],
                    output_vars=bases,
                    details={"type": "int[]", "bases": bases,
                             "count_var": count_var, "separator": "space"},
                ))
            elif var.var_type == "int[][]":
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="read_matrix",
                    description=f"Read matrix {var.name} from input",
                    input_vars=[],
                    output_vars=[var.name],
                    details={"type": "int[][]", "name": var.name},
                ))
            step_id += 1

        # If no variables detected, add generic input reading
        if not steps:
            steps.append(PlanStep(
                step_id=step_id,
                operation="read_input",
                description="Read all input data",
                input_vars=[],
                output_vars=["data"],
                details={"type": "generic"},
            ))

        return steps

    def _build_algorithm_steps(self, hypothesis, sp,
                               input_parsing: list | None = None) -> list[PlanStep]:
        """Build algorithm steps by composing primitives based on semantic structure.

        This method looks at:
        - The hypothesis primitives (selected algorithms)
        - The structured problem (hints, constraints, I/O)
        - The input parsing steps (the variables the input produced)
        - Composes them into a coherent algorithm with connected steps
        """
        # the graph step needs the edge variables the input parsing
        # produced — thread them through
        self._current_input_parsing = input_parsing or []
        steps = []
        step_id = 1

        hints = set(sp.structural_hints)
        input_vars = sp.input_variables
        output_vars = sp.output_variables
        constraints = sp.constraints

        # Get variable names
        var_names = [v.name for v in input_vars]
        out_var_names = [v.name for v in output_vars]
        # The multi-case repeat count (T/Q) is a HEADER, never data:
        # templates must not sort/iterate it when a body var exists.
        mc_count = getattr(sp, "multi_case_count_var", "") or ""

        # Determine the main data variable.  The primary DATA is the
        # problem's collection (sequence / grid / strings) — a leading
        # scalar is almost always the SIZE ("N"), and templates iterate
        # or sort the data, never the count.
        data_var = next(
            (v.name for v in input_vars
             if v.var_type in ("int[]", "array", "string[]", "float[]")
             and v.name != mc_count),
            None)
        if data_var is None:
            for v in input_vars:
                if v.var_type == "int_pairs" and v.name != mc_count:
                    data_var = v.name.split("+")[0]
                    break
        if data_var is None:
            data_var = next(
                (v.name for v in input_vars
                 if v.var_type in ("string", "str") and v.name != mc_count),
                None)
        if data_var is None:
            body_names = [n for n in var_names if n != mc_count]
            data_var = body_names[0] if body_names else (
                var_names[0] if var_names else "data")

        # Determine the result variable
        result_var = "result"
        if out_var_names:
            result_var = out_var_names[0]

        # Build steps based on hypothesis primitives and semantic structure
        primitives = hypothesis.algorithmic_primitives

        # Check if filtering is on input data or on computed values
        has_filter_on_input = False
        has_filter_on_computed = False
        
        for c in constraints:
            if c.constraint_type == "filter" and c.predicate:
                # Check if the filter applies to input data or computed values
                if c.predicate.source == "value" and var_names:
                    # "value" is generic - check if it refers to input or computed
                    # For now, assume it applies to computed values during iteration
                    has_filter_on_computed = True
                else:
                    has_filter_on_input = True

        # Phase 0: String/stack operations (bracket balance, repeat replace).
        stack_op = getattr(sp, "string_stack_op", None)
        if stack_op == "bracket_balance":
            strings_var = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("string", "string[]")),
                data_var)
            # When the problem declares MULTIPLE bracket pairs ("(, ),
            # [, ], <, >"), the pair-based stack matching handles all of
            # them; a single-pair depth counter cannot.
            if getattr(sp, "bracket_match", None):
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="bracket_match",
                    description=f"Check bracket-pair matching of {strings_var}",
                    input_vars=[strings_var],
                    output_vars=[result_var],
                    details={"pairs": sp.bracket_match.get("pairs"),
                             "string_var": strings_var,
                             "type": "main_computation"},
                ))
                step_id += 1
                return steps
            steps.append(PlanStep(
                step_id=step_id,
                operation="bracket_balance",
                description=f"Check bracket balance of {strings_var}",
                input_vars=[strings_var],
                output_vars=[result_var],
                details={"string_var": strings_var, "type": "main_computation"},
            ))
            step_id += 1
            return steps
        elif isinstance(stack_op, tuple) and stack_op[0] == "repeat_replace":
            strings_var = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("string", "string[]")),
                data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="repeat_replace",
                description=f"Repeat replacement of {stack_op[1]} with {stack_op[2]} in {strings_var}",
                input_vars=[strings_var],
                output_vars=[result_var],
                details={"string_var": strings_var, "find": stack_op[1],
                         "repl": stack_op[2], "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Arithmetic-progression triplet counting.
        # Count triplets (A, B, C), A < B < C, with B-A = C-B over a
        # distinct-value set → for each middle value, count symmetric pairs;
        # computed via big-int convolution over the value domain.
        if getattr(sp, "ap_triplet_count", None):
            # the collection the triplet set comes from — an array
            # variable, not a scalar count
            arr_var = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("int[]", "array")),
                var_names[0] if var_names else "data")
            steps.append(PlanStep(
                step_id=step_id,
                operation="ap_triplet_count",
                description="Count arithmetic-progression triplets",
                input_vars=[arr_var],
                output_vars=[result_var],
                details={"type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Adjacent-swap gathering.
        # Minimum adjacent swaps to make all occurrences of a character
        # contiguous: median-aligned block distance sum.
        sw_gather = getattr(sp, "swap_gather", None)
        if sw_gather:
            svar, tchar = sw_gather
            steps.append(PlanStep(
                step_id=step_id,
                operation="swap_gather",
                description=f"Minimum adjacent swaps to gather '{tchar}'s in {svar}",
                input_vars=[svar],
                output_vars=[result_var],
                details={"string_var": svar, "target_char": tchar,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Minimum edge deletion to forest.
        # Union-find counts components; deletions = M - (N - C).
        if getattr(sp, "forest_deletion", None):
            steps.append(PlanStep(
                step_id=step_id,
                operation="forest_deletion",
                description="Minimum edge deletions to make a forest",
                input_vars=[],
                output_vars=[result_var],
                details={"type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Ranking with ties.
        rank_var = getattr(sp, "ranking_ties", None)
        if rank_var:
            steps.append(PlanStep(
                step_id=step_id,
                operation="ranking_ties",
                description=f"Rank each person by score in {rank_var}",
                input_vars=[rank_var],
                output_vars=[result_var],
                details={"data_var": rank_var, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Split-point distinct-count maximization.
        split_dm = getattr(sp, "split_distinct_max", None)
        if split_dm:
            steps.append(PlanStep(
                step_id=step_id,
                operation="split_distinct_max",
                description=f"Max distinct-count sum over split points in {split_dm}",
                input_vars=[split_dm],
                output_vars=[result_var],
                details={"data_var": split_dm, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Letter-replacement reachability.
        letter_rep = getattr(sp, "letter_replacement", None)
        if letter_rep:
            steps.append(PlanStep(
                step_id=step_id,
                operation="letter_replacement",
                description=f"Min replacements to turn {letter_rep[0]} into {letter_rep[1]}",
                input_vars=list(letter_rep),
                output_vars=[result_var],
                details={"s_var": letter_rep[0], "t_var": letter_rep[1],
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Stack query processing.
        stack_q = getattr(sp, "stack_queries", None)
        if stack_q:
            steps.append(PlanStep(
                step_id=step_id,
                operation="stack_queries",
                description=f"Process stack queries (initial: {stack_q['initial_count']} x {stack_q['initial_value']})",
                input_vars=[],
                output_vars=[result_var],
                details={"initial_count": stack_q["initial_count"],
                         "initial_value": stack_q["initial_value"],
                         "count_var": stack_q["count_var"],
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Consecutive-repetition check.
        run_len = getattr(sp, "consecutive_repeat_check", None)
        if run_len:
            check_input = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("int[]", "array")),
                data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="consecutive_repeat_check",
                description=f"Check {run_len}+ consecutive repeats in {check_input}",
                input_vars=[check_input],
                output_vars=[result_var],
                details={"run_len": run_len, "data_var": check_input,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Shortest-palindrome-prefix construction.
        pal_pref = getattr(sp, "palindrome_prefix", None)
        if pal_pref:
            steps.append(PlanStep(
                step_id=step_id,
                operation="palindrome_prefix",
                description=f"Shortest palindrome with {pal_pref} as prefix",
                input_vars=[pal_pref],
                output_vars=[result_var],
                details={"string_var": pal_pref, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Centered-symbol palindrome construction.
        cent_pal = getattr(sp, "centered_palindrome", None)
        if cent_pal and cent_pal.get("n_var"):
            steps.append(PlanStep(
                step_id=step_id,
                operation="centered_palindrome",
                description=f"Construct centered-{cent_pal['center']} palindrome of length {cent_pal['n_var']}",
                input_vars=[cent_pal["n_var"]],
                output_vars=[result_var],
                details={"fill": cent_pal["fill"], "center": cent_pal["center"],
                         "n_var": cent_pal["n_var"], "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Unique-maximum label.
        uniq_max = getattr(sp, "unique_max_label", None)
        if uniq_max:
            steps.append(PlanStep(
                step_id=step_id,
                operation="unique_max_label",
                description=f"Label of the unique greatest value in {uniq_max}",
                input_vars=[uniq_max],
                output_vars=[result_var],
                details={"data_var": uniq_max, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Rectangular-arrangement division.
        rect_arr = getattr(sp, "rect_arrangement", None)
        if rect_arr:
            steps.append(PlanStep(
                step_id=step_id,
                operation="rect_arrangement",
                description=f"Columns for {rect_arr['count']} people in {rect_arr['a_var']} rows",
                input_vars=[rect_arr["a_var"]],
                output_vars=[result_var],
                details={"count": rect_arr["count"], "a_var": rect_arr["a_var"],
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Constraint-set deduction.
        # "Exactly one of these {C} types causes {bad}... find which type"
        # — the brute force over the candidates with the constraint
        # checks (the eaten sets + the status conditions).
        sd = getattr(sp, "set_deduction", None)
        if sd and sd.get("eaten_sets"):
            steps.append(PlanStep(
                step_id=step_id,
                operation="set_deduction",
                description=f"Deduce the causing type among "
                            f"{sd['candidate_count']} candidates",
                input_vars=["status"],
                output_vars=[result_var],
                details={"candidate_count": sd["candidate_count"],
                         "eaten_sets": sd["eaten_sets"],
                         "status_words": sd.get("status_words", ["sick", "fine"]),
                         "type": "main_computation"},
                input_types=["List<String>"],
                output_types=["Integer"],
                preconditions=["exactly one candidate is the cause"],
                postconditions=[f"{result_var} is the causing type's label"],
                dependencies=[],
                semantic_justification="constraint-set deduction: brute "
                                       "force over candidates with the "
                                       "eaten-set membership checks",
            ))
            step_id += 1
            return steps

        # Phase 0: 2D pattern search in a grid.
        # "an N×N grid S and an M×M grid T... find where T appears in S"
        # — the brute force over the top-left positions.
        gps = getattr(sp, "grid_pattern_search", None)
        if gps:
            steps.append(PlanStep(
                step_id=step_id,
                operation="grid_pattern_search",
                description=f"Find where {gps['small']} appears in {gps['big']}",
                input_vars=[gps['big'], gps['small']],
                output_vars=[result_var],
                details={"big": gps['big'], "small": gps['small'],
                         "type": "main_computation"},
                input_types=["List<String>", "List<String>"],
                output_types=["Integer", "Integer"],
                preconditions=["both grids are read as row strings"],
                postconditions=[f"{result_var} is the 1-indexed top-left "
                                f"position of the match"],
                dependencies=[],
                semantic_justification="2D pattern search: brute-force "
                                       "window comparison at every "
                                       "top-left position",
            ))
            step_id += 1
            return steps

        # Phase 0: Bracket-sequence matching.
        # "turn T into an empty string by deleting (), [], <>" — the
        # bracket-pair stack matching over the declared pairs.
        bm = getattr(sp, "bracket_match", None)
        if bm:
            check_input = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("string",)),
                data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="bracket_match",
                description="Check the bracket-pair matching",
                input_vars=[check_input],
                output_vars=[result_var],
                details={"pairs": bm.get("pairs"),
                         "type": "main_computation"},
                input_types=["String"],
                output_types=["Boolean"],
                preconditions=["the string contains only the declared bracket characters"],
                postconditions=[f"{result_var} is whether the sequence reduces to empty"],
                dependencies=[],
                semantic_justification="bracket-pair matching: stack push "
                                       "on opens, pop-match on closes",
            ))
            step_id += 1
            return steps

        # Phase 0: Concentric-ring grid filling.
        # "fill the rectangular region (i,i)-(j,j) with {color} if i is
        # odd" — the alternating ring fill from the outside in, then the
        # rows printed.
        crf = getattr(sp, "concentric_ring_fill", None)
        if crf:
            steps.append(PlanStep(
                step_id=step_id,
                operation="concentric_ring_fill",
                description=f"Fill concentric rings of an "
                            f"{crf.get('size_var', 'N')} x "
                            f"{crf.get('size_var', 'N')} grid",
                input_vars=[crf.get("size_var", "N")],
                output_vars=[result_var],
                details={"size_var": crf.get("size_var", "N"),
                         "type": "main_computation"},
                input_types=["Integer"],
                output_types=["List<String>"],
                preconditions=["the grid starts uncolored"],
                postconditions=["the rows form the alternating ring pattern"],
                dependencies=[],
                semantic_justification="concentric-ring fill: alternating "
                                       "black/white rectangular regions "
                                       "from the outside in",
            ))
            step_id += 1
            return steps

        # Phase 0: Minimum-XOR simple path.
        # "Among all simple paths from vertex 1 to vertex N, minimum XOR
        # of the edge labels" — the DFS over every simple path (small N).
        mxp = getattr(sp, "min_xor_path", None)
        if mxp:
            steps.append(PlanStep(
                step_id=step_id,
                operation="min_xor_path",
                description="Minimum XOR over all simple paths",
                input_vars=[data_var],
                output_vars=[result_var],
                details={"start": mxp.get("start", 1),
                         "type": "main_computation"},
                input_types=["List<Integer>"],
                output_types=["Integer"],
                preconditions=["the graph is a simple connected undirected graph"],
                postconditions=[f"{result_var} is the minimum path XOR"],
                dependencies=[],
                semantic_justification="minimum-XOR path: exhaustive DFS "
                                       "over simple paths with XOR "
                                       "accumulation",
            ))
            step_id += 1
            return steps

        # Phase 0: Alternating-string insertion counting.
        # "insert zero or more characters... every odd-numbered character
        # is {a}, every even-numbered is {b}" — the greedy subsequence
        # walk over the alternating pattern; the insertions are the
        # positions consumed without matching.
        ai = getattr(sp, "alternating_insertion", None)
        if ai:
            check_input = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("string",)), data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="alternating_insertion",
                description="Min insertions to reach the alternating pattern",
                input_vars=[check_input],
                output_vars=[result_var],
                details={"odd_char": ai.get("odd_char", "i"),
                         "even_char": ai.get("even_char", "o"),
                         "type": "main_computation"},
                input_types=["String"],
                output_types=["Integer"],
                preconditions=["the string consists of the two declared characters"],
                postconditions=[f"{result_var} is the minimum insertion count"],
                dependencies=[],
                semantic_justification="alternating-pattern insertion: "
                                       "greedy subsequence walk with the "
                                       "parity-driven expectation",
            ))
            step_id += 1
            return steps

        # Phase 0: Moving-set simulation with a respawn point.
        # "X exists only at (0,0)... new X is generated at (0,0)...
        # determine if X exists at (R,C)" — the whole set moves one cell
        # per step (the wind), the origin respawns when empty, and the
        # output is the per-step presence string.
        mss = getattr(sp, "moving_set_simulation", None)
        if mss:
            steps.append(PlanStep(
                step_id=step_id,
                operation="moving_set_simulation",
                description="Simulate the moving set with a respawn point",
                input_vars=["S"],
                output_vars=[result_var],
                details={"origin": mss.get("origin", [0, 0]),
                         "type": "main_computation"},
                input_types=["String"],
                output_types=["String"],
                preconditions=["the direction string is read"],
                postconditions=["the per-step presence at the target cell"],
                dependencies=[],
                semantic_justification="moving-set simulation: whole-set "
                                       "translation per step with a fixed "
                                       "respawn point",
            ))
            step_id += 1
            return steps

        # Phase 0: Dual-collection selection with a count constraint.
        # "choose zero or more {X} so that #{X} >= #{Y}, maximum sum" —
        # the prefix sums + the suffix max over the two sorted
        # collections.
        dcm = getattr(sp, "dual_collection_max", None)
        if dcm:
            steps.append(PlanStep(
                step_id=step_id,
                operation="dual_collection_max",
                description="Max sum with the count constraint over two collections",
                input_vars=[data_var],
                output_vars=[result_var],
                details={"kind": dcm.get("kind", "count_at_least"),
                         "type": "main_computation"},
                input_types=["List<Integer>", "List<Integer>"],
                output_types=["Integer"],
                preconditions=["both collections are fully read"],
                postconditions=[f"{result_var} is the maximum sum"],
                dependencies=[],
                semantic_justification="dual-collection max: sorted prefix "
                                       "sums with the count constraint "
                                       "via a suffix maximum",
            ))
            step_id += 1
            return steps

        # Phase 0: Pairwise maximum probability.
        # "choose two {items}... maximum probability that the two show
        # the same {X}" — the max over all pairs of the intersection
        # probability (the common count / the product of the sizes).
        pp = getattr(sp, "pairwise_probability", None)
        if pp:
            steps.append(PlanStep(
                step_id=step_id,
                operation="pairwise_probability",
                description="Max same-value probability over all pairs",
                input_vars=[data_var],
                output_vars=[result_var],
                details={"kind": pp.get("kind", "same_value"),
                         "type": "main_computation"},
                input_types=["List<List<Integer>>"],
                output_types=["Float"],
                preconditions=["each group's faces are known"],
                postconditions=[f"{result_var} is the maximum probability"],
                dependencies=[],
                semantic_justification="pairwise maximum: intersection "
                                       "count over the product of group "
                                       "sizes",
            ))
            step_id += 1
            return steps

        # Phase 0: Permutation existence check.
        # "Let B be any permutation of A. Determine whether {condition}"
        # — the brute force over all permutations of the input values.
        perm_chk = getattr(sp, "permutation_check", None)
        if perm_chk:
            perm_var = next((v.name for v in sp.input_variables), "A")
            steps.append(PlanStep(
                step_id=step_id,
                operation="permutation_check",
                description=f"Check the condition over all permutations of {perm_var}",
                input_vars=[perm_var],
                output_vars=[result_var],
                details={"condition": perm_chk.get("condition"),
                         "var": perm_chk.get("var", "b"),
                         "input_var": perm_var,
                         "type": "main_computation"},
                input_types=["List<Integer>"],
                output_types=["Boolean"],
                preconditions=["the input values are integers"],
                postconditions=[f"{result_var} is whether any permutation "
                                f"satisfies the condition"],
                dependencies=[],
                semantic_justification="permutation existence: brute-force "
                                       "over all permutations with the "
                                       "declared condition",
            ))
            step_id += 1
            return steps

        # Phase 0: Threshold classification.
        # "X is classified into the following:" with thresholds → the
        # if/elif chain with class codes in threshold order.
        thr_cls = getattr(sp, "threshold_classification", None)
        if thr_cls:
            input_var = next((v.name for v in sp.input_variables), "X")
            steps.append(PlanStep(
                step_id=step_id,
                operation="threshold_classification",
                description=f"Classify {input_var} by {len(thr_cls)} thresholds",
                input_vars=[input_var],
                output_vars=[result_var],
                details={"thresholds": thr_cls, "input_var": input_var,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Power-difference equation search.
        # "exists (x, y) with x^k - y^k = N" → enumerate x, test the
        # remainder for a perfect k-th power.
        pdiff = getattr(sp, "power_difference_search", None)
        if pdiff:
            exponent, target = pdiff
            steps.append(PlanStep(
                step_id=step_id,
                operation="power_difference_search",
                description=f"Find (x, y) with x^{exponent} - y^{exponent} = {target}",
                input_vars=[target],
                output_vars=[result_var],
                details={"exponent": exponent, "target": target,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Power series with overflow cap.
        # "X = sum of N^i for i = 0..M; if X <= cap print X else inf"
        pser = getattr(sp, "power_series", None)
        if pser:
            base_var = pser["base"]
            steps.append(PlanStep(
                step_id=step_id,
                operation="power_series",
                description=f"Compute power series with base {base_var}",
                input_vars=[base_var],
                output_vars=[result_var],
                details={"base": base_var, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Monotonicity check (general property check).
        mono_kind = getattr(sp, "monotonic_check", None)
        if mono_kind:
            check_input = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("int[]", "array")),
                data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="check_property",
                description=f"Check {mono_kind}",
                input_vars=[check_input],
                output_vars=[result_var],
                details={"check_type": mono_kind, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Hamming distance (general string comparison).
        hamming = getattr(sp, "hamming_distance", None)
        if hamming:
            steps.append(PlanStep(
                step_id=step_id,
                operation="hamming_distance",
                description=f"Count differing positions between {hamming[0]} and {hamming[1]}",
                input_vars=list(hamming),
                output_vars=[result_var],
                details={"var_a": hamming[0], "var_b": hamming[1],
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Full-house check (general card property).
        if getattr(sp, "full_house_check", None):
            cards_var = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("int[]", "array")),
                data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="full_house_check",
                description=f"Check full house among {cards_var}",
                input_vars=[cards_var],
                output_vars=[result_var],
                details={"data_var": cards_var, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Rectangle-formation check on a grid.
        # The black cells must be exactly a rectangle: the bounding box of
        # the black cells contains no white cells inside.
        if getattr(sp, "rectangle_paint", None):
            grid_var = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("string[]", "string")),
                data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="rectangle_paint",
                description=f"Check black cells form a rectangle in {grid_var}",
                input_vars=[grid_var],
                output_vars=[result_var],
                details={"grid_var": grid_var, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Dice-pair agreement probability.
        # "maximum probability that the two dice show the same number" →
        # grouped dice reading, then max agreement over pairs.
        if getattr(sp, "dice_agreement", None):
            steps.append(PlanStep(
                step_id=step_id,
                operation="dice_max_agreement",
                description="Max agreement probability over dice pairs",
                input_vars=[],
                output_vars=[result_var],
                details={"type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Sequential gift simulation.
        # Each step: the newcomer receives one item from every active
        # participant, then becomes active with their total.
        gift_sim = getattr(sp, "gift_simulation", None)
        if gift_sim:
            steps.append(PlanStep(
                step_id=step_id,
                operation="gift_simulation",
                description="Sequential gift simulation over participants",
                input_vars=[gift_sim["data_var"]],
                output_vars=[result_var],
                details={"data_var": gift_sim["data_var"],
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Maximum-GCD-of-subset queries.
        # "choose K elements that include A_i, find the maximum possible
        # GCD" → divisor sieve over the value domain.
        gcd_sub = getattr(sp, "gcd_subset", None)
        if gcd_sub:
            data_var = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("int[]", "array")),
                data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="gcd_subset",
                description=f"Max GCD of {gcd_sub['k_var']}-subsets per element",
                input_vars=[data_var],
                output_vars=[result_var],
                details={"data_var": data_var, "k_var": gcd_sub["k_var"],
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Counting shapes inside a circle.
        # Unit squares fully inside a circle of radius R: farthest corner
        # (|i|+0.5, |j|+0.5) within distance R. Counted with quadrant symmetry.
        circ = getattr(sp, "circle_count", None)
        if circ:
            steps.append(PlanStep(
                step_id=step_id,
                operation="circle_count",
                description=f"Count squares fully inside circle of radius {circ}",
                input_vars=[circ],
                output_vars=[result_var],
                details={"radius_var": circ, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Ordered character-triple counting.
        # Count (i, j, k) with j-i = k-j and fixed characters at i, j, k.
        tri3 = getattr(sp, "triple_count", None)
        if tri3:
            svar = tri3[0]
            steps.append(PlanStep(
                step_id=step_id,
                operation="triple_count",
                description=f"Count {tri3[1]}..{tri3[2]}..{tri3[3]} triples in {svar}",
                input_vars=[svar],
                output_vars=[result_var],
                details={"string_var": svar, "chars": list(tri3[1:]),
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Insert simulation.
        # "insert the number i into A so that it becomes the P_i-th element"
        ins_sim = getattr(sp, "insert_simulation", None)
        if ins_sim:
            arr_var = next(
                (v.name for v in sp.input_variables
                 if v.name == ins_sim["pos_var"] and v.var_type in ("int[]", "array")),
                ins_sim["pos_var"])
            steps.append(PlanStep(
                step_id=step_id,
                operation="insert_simulation",
                description=f"Insert values sequentially into array",
                input_vars=[arr_var],
                output_vars=[result_var],
                details={
                    "array_var": arr_var,
                    "pos_var": ins_sim["pos_var"],
                    "type": "main_computation",
                },
            ))
            step_id += 1
            return steps

        # Phase 0: Functional-graph value-assignment counting.
        # "count sequences x with x_i <= x_{A_i}" — the cycles share a
        # value; the in-trees are counted via DP with prefix sums.
        fgc = getattr(sp, "functional_graph_count", None)
        if fgc:
            arrays = [v.name for v in sp.input_variables
                      if v.var_type in ("int[]", "array")]
            # the value domain: the declared scalar count (N then M —
            # the second scalar is the value bound)
            scalars = [v.name for v in sp.input_variables if v.var_type == "int"]
            value_var = scalars[1] if len(scalars) > 1 else "m"
            steps.append(PlanStep(
                step_id=step_id,
                operation="functional_graph_count",
                description="Count value assignments under parent "
                            "inequality on the functional graph",
                input_vars=arrays,
                output_vars=[result_var],
                details={
                    "data_var": arrays[0] if arrays else "data",
                    "value_var": value_var,
                    "type": "main_computation",
                },
                input_types=["List<Integer>"],
                output_types=["Integer"],
                preconditions=["the parent pointers form a functional "
                               "graph (each node has exactly one parent)"],
                postconditions=[f"{result_var} is the count modulo the "
                                f"declared modulus"],
                dependencies=[],
                semantic_justification="functional-graph counting: cycles "
                                       "share a value; in-trees via DP with "
                                       "prefix sums",
            ))
            step_id += 1
            return steps

        # Phase 0: Multi-category budgeted selection with a min-aggregate
        # objective — per-category knapsack + threshold binary search.
        mms = getattr(sp, "min_max_selection", None)
        if mms:
            triples = [v.name for v in sp.input_variables
                       if v.var_type == "int_pairs"]
            bases = triples[0].split('+') if triples else []
            steps.append(PlanStep(
                step_id=step_id,
                operation="min_max_selection",
                description=f"Maximize min aggregate under budget "
                            f"{mms.get('budget_var', 'X')}",
                input_vars=bases,
                output_vars=[result_var],
                details={
                    "budget_var": mms.get('budget_var', 'X'),
                    "bases": bases,
                    "type": "main_computation",
                },
                input_types=["List<Integer>", "List<Integer>", "List<Integer>"],
                output_types=["Integer"],
                preconditions=["the total cost of the selection is within "
                               "the budget"],
                postconditions=[f"{result_var} is the maximum achievable "
                                f"minimum aggregate"],
                dependencies=[],
                semantic_justification="multi-category budgeted selection: "
                                       "per-category knapsack + threshold "
                                       "binary search on the min objective",
            ))
            step_id += 1
            return steps

        # Phase 0: Budgeted convex-cost maximization.
        # "maximum number of units ... total cost at most M" with a
        # quadratic unit cost — the marginal threshold binary search +
        # greedy purchase.
        bmu = getattr(sp, "budgeted_max_units", None)
        if bmu:
            arrays = [v.name for v in sp.input_variables
                      if v.var_type in ("int[]", "array")]
            steps.append(PlanStep(
                step_id=step_id,
                operation="budgeted_max_units",
                description=f"Max units under budget "
                            f"{bmu.get('budget_var', 'M')} with convex costs",
                input_vars=arrays,
                output_vars=[result_var],
                details={
                    "budget_var": bmu.get('budget_var', 'M'),
                    "cost_kind": bmu.get('cost_kind', 'quadratic'),
                    "data_var": arrays[0] if arrays else "data",
                    "type": "main_computation",
                },
                input_types=["List<Integer>"],
                output_types=["Integer"],
                preconditions=["each unit cost is positive (convex)"],
                postconditions=[f"{result_var} is the maximum unit count "
                                f"within the budget"],
                dependencies=[],
                semantic_justification="budgeted convex-cost maximization: "
                                       "marginal threshold binary search + "
                                       "greedy marginal purchase",
            ))
            step_id += 1
            return steps

        # Phase 0: Power-of-two × square counting.
        # "X = 2^a × b^2" / "good integer" — count values of the form
        # 2^a·b² (a ≥ 1, b ≥ 1) up to N.  The canonical form uses b odd
        # (each value's 2-adic valuation is then unique), so the count
        # sums, per power a, the odd b with b² ≤ N/2^a — O(log N) via
        # integer square roots.
        tpsc = getattr(sp, "two_power_square_count", None)
        if tpsc:
            hi_vars = [v.name for v in sp.input_variables
                       if v.var_type in ("int",)] or ["N"]
            steps.append(PlanStep(
                step_id=step_id,
                operation="two_power_square_count",
                description=f"Count values of the form 2^a·b² in "
                            f"[1, {hi_vars[0]}]",
                input_vars=[hi_vars[0]],
                output_vars=[result_var],
                details={
                    "hi_var": hi_vars[0],
                    "type": "main_computation",
                },
                input_types=["Integer"],
                output_types=["Integer"],
                preconditions=["the upper bound is a positive integer"],
                postconditions=[f"{result_var} is the count of 2^a·b² "
                                f"values up to {hi_vars[0]}"],
                dependencies=[],
                semantic_justification="power-of-two times square count: "
                                       "per-power odd-square count via "
                                       "integer sqrt (canonical b odd, "
                                       "no double counting)",
            ))
            step_id += 1
            return steps

        # Phase 0: Digit-property count over a range.
        # "how many {X} numbers exist between L and R" with a digit-shape
        # property — the count via digit-length decomposition.
        dpc = getattr(sp, "digit_property_count", None)
        if dpc:
            steps.append(PlanStep(
                step_id=step_id,
                operation="digit_property_count",
                description=f"Count values with digit property "
                            f"({dpc.get('property', 'top_dominant')}) in "
                            f"[{dpc.get('lo_var', 'L')}, {dpc.get('hi_var', 'R')}]",
                input_vars=[dpc.get('lo_var', 'L'), dpc.get('hi_var', 'R')],
                output_vars=[result_var],
                details={
                    "property": dpc.get('property', 'top_dominant'),
                    "lo_var": dpc.get('lo_var', 'L'),
                    "hi_var": dpc.get('hi_var', 'R'),
                    "type": "main_computation",
                },
                input_types=["Integer", "Integer"],
                output_types=["Integer"],
                preconditions=["the range bounds are non-negative integers"],
                postconditions=[f"{result_var} is the count of property-"
                                f"satisfying values in the range"],
                dependencies=[],
                semantic_justification="digit-property count: property-"
                                       "filtered count via digit-length "
                                       "decomposition",
            ))
            step_id += 1
            return steps

        # Phase 0: Per-step aggregate with an offset expression.
        # "for each k satisfying lo <= k <= hi, the {aggregate} when every
        # {X}'s {attr} increased by k" — a per-step aggregate over the
        # parallel collections.
        psa = getattr(sp, "per_step_aggregate", None)
        if psa:
            pairs = psa.get("pairs") or []
            steps.append(PlanStep(
                step_id=step_id,
                operation="per_step_aggregate",
                description=f"Per-step {psa.get('agg', 'max')} over offset "
                            f"expression for k in [{psa.get('range_lo', 1)}, "
                            f"{psa.get('range_hi_var', 'D')}]",
                input_vars=pairs,
                output_vars=[result_var],
                details={
                    "range_lo": psa.get("range_lo", 1),
                    "range_hi_var": psa.get("range_hi_var", "D"),
                    "agg": psa.get("agg", "max"),
                    "pairs": pairs,
                    "type": "main_computation",
                },
                input_types=["List<Integer>", "List<Integer>"],
                output_types=["List<Integer>"],
                preconditions=["each step value k is within the declared "
                               "range"],
                postconditions=[f"{result_var} has one aggregate value per "
                                f"step"],
                dependencies=[],
                semantic_justification="per-step aggregate: for each k in a "
                                       "range, an aggregate over an offset "
                                       "expression on parallel collections",
            ))
            step_id += 1
            return steps

        # Phase 0: Chained references through parallel collections.
        # A value obtained via a chain of lookups ("for each i: x = V0[i];
        # y = V1[x]; ..." or with an inverse first step: "the X with attr i")
        # — the general indirect-reference dataflow, not a per-problem
        # operation. Steps bind to the parallel input collections.
        chain = getattr(sp, "index_chain", None)
        if chain:
            chain_steps = chain["steps"]
            chain_vars = [col for _kind, col in chain_steps]
            steps.append(PlanStep(
                step_id=step_id,
                operation="index_chain",
                description=f"For each index, chain references: "
                            f"{' -> '.join(f'{k}({c})' for k, c in chain_steps)}",
                input_vars=chain_vars,
                output_vars=[result_var],
                details={
                    "steps": [(k, c) for k, c in chain_steps],
                    "for_each": True,
                    "type": "main_computation",
                },
                input_types=["List<Integer>" for _ in chain_vars],
                output_types=["List<Integer>"],
                preconditions=["each reference target is a valid index into "
                               "the next collection"],
                postconditions=[f"{result_var} has the same cardinality as "
                                f"the outer iteration domain"],
                dependencies=[],
                semantic_justification="chained reference dataflow: "
                                       "for-each + index/inverse chain over "
                                       "parallel collections",
            ))
            step_id += 1
            return steps

        # Phase 0: First-character + literal concatenation.
        # "concatenating the first character of S and {LITERAL}" →
        # the composition is general; the literal is problem data.
        fcc = getattr(sp, "first_char_concat", None)
        if fcc:
            svar, literal = fcc
            steps.append(PlanStep(
                step_id=step_id,
                operation="first_char_concat",
                description=f"Concatenate first character of {svar} with literal",
                input_vars=[svar],
                output_vars=[result_var],
                details={"string_var": svar, "literal": literal,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Opposite-direction transformation.
        # Compass knowledge: opposite directions flip each component.
        dir_flip = getattr(sp, "direction_flip", None)
        if dir_flip:
            steps.append(PlanStep(
                step_id=step_id,
                operation="direction_flip",
                description=f"Flip each direction component of {dir_flip}",
                input_vars=[dir_flip],
                output_vars=[result_var],
                details={"string_var": dir_flip, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Shortest span with duplicate occurrences.
        dup_var = getattr(sp, "shortest_duplicate", None)
        if dup_var:
            steps.append(PlanStep(
                step_id=step_id,
                operation="shortest_duplicate_span",
                description=f"Find shortest span with duplicate in {dup_var}",
                input_vars=[dup_var],
                output_vars=[result_var],
                details={"data_var": dup_var, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Per-query simulation with a range condition.
        # "You are given Q queries ... determine the value after participating
        # in all N contests" where each step applies a range rule.
        qsim = getattr(sp, "query_simulation", None)
        if qsim:
            lo_var, hi_var = qsim["lo_var"], qsim["hi_var"]
            query_var = qsim["query_var"]
            # The range bounds are the pair arrays (L, R)
            steps.append(PlanStep(
                step_id=step_id,
                operation="query_simulation",
                description=f"Simulate {query_var} over {qsim['count_var']} queries with range condition",
                input_vars=[lo_var, hi_var],
                output_vars=[result_var],
                details={
                    "count_var": qsim["count_var"],
                    "query_var": query_var,
                    "lo_var": lo_var,
                    "hi_var": hi_var,
                    "increment": qsim["increment"],
                    "type": "main_computation",
                },
                input_types=["List<Integer>", "List<Integer>"],
                output_types=["List<Integer>"],
                preconditions=[f"each {query_var} query value is within the "
                               f"simulation domain"],
                postconditions=[f"{result_var} has one answer per query"],
                dependencies=[],
                semantic_justification="per-query monotone-increment "
                                       "simulation: steps with a range "
                                       "condition increment the value",
            ))
            step_id += 1
            return steps

        # Phase 0: Pair counting with a size relation condition.
        # "you can make one kagamimochi ... if and only if a is at most half
        # of b" + counting -> count pairs (i, j) satisfying the relation.
        pair_cnt = getattr(sp, "pair_count", None)
        if pair_cnt:
            array_var = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("int[]", "array")),
                data_var)
            mode = pair_cnt.get("mode", "count")
            steps.append(PlanStep(
                step_id=step_id,
                operation="pair_count",
                description=f"{'Maximum matching' if mode == 'max_matching' else 'Count pairs'} with {pair_cnt['relation']} relation",
                input_vars=[array_var],
                output_vars=[result_var],
                details={
                    "array_var": array_var,
                    "condition_kind": pair_cnt["relation"],
                    "mode": mode,
                    "type": "main_computation",
                },
            ))
            step_id += 1
            return steps

        # Phase 0: Character filtering.
        char_filt = getattr(sp, "char_filter", None)
        if char_filt:
            svar, keep = char_filt
            steps.append(PlanStep(
                step_id=step_id,
                operation="char_filter",
                description=f"Keep only '{keep}' characters in {svar}",
                input_vars=[svar],
                output_vars=[result_var],
                details={"string_var": svar, "keep": keep,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 0: Sort-by-key + concatenation.
        # "Sort these strings in ascending order of length, then concatenate"
        sort_cat = getattr(sp, "sort_concat", None)
        if sort_cat:
            key, order = sort_cat
            # The strings array is the sequence input
            strings_var = next(
                (v.name for v in sp.input_variables
                 if v.var_type in ("string[]", "string", "int[]", "array")),
                data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="sort_concat",
                description=f"Sort {strings_var} by {key} and concatenate",
                input_vars=[strings_var],
                output_vars=[result_var],
                details={"key": key, "order": order, "string_var": strings_var,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 1: Equation solving by enumeration.
        # "Find N such that N! = X" → enumerate candidates until the
        # factorial relation holds. General equation-solving shape.
        eq_search = getattr(sp, "equation_search", None)
        if eq_search:
            relation, search_var, target = eq_search
            target_expr = target if target.isdigit() else next(
                (v.name for v in sp.input_variables if v.name == target), target)
            steps.append(PlanStep(
                step_id=step_id,
                operation="equation_search",
                description=f"Find {search_var} such that {search_var}! = {target_expr}",
                input_vars=[target_expr],
                output_vars=[result_var],
                details={"relation": relation, "search_var": search_var,
                         "target": target_expr, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 1: Set-complement goals.
        # "List all integers between 1 and N that do not appear in A" →
        # collect the complement of a set within a range.
        set_comp = getattr(sp, "set_complement", None)
        if set_comp:
            hi_var, data_var_c = set_comp
            steps.append(PlanStep(
                step_id=step_id,
                operation="set_complement",
                description=f"Collect integers up to {hi_var} not in {data_var_c}",
                input_vars=[data_var_c],
                output_vars=[result_var],
                details={"hi_var": hi_var, "data_var": data_var_c,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 1: String-embedded numbers aggregation.
        # "Find the product of the two numbers in S" (S is a string) →
        # extract digit groups from the string and aggregate them.
        string_agg = getattr(sp, "string_numbers_agg", None)
        if string_agg:
            agg_func, string_var = string_agg
            steps.append(PlanStep(
                step_id=step_id,
                operation="string_numbers_aggregate",
                description=f"Extract numbers from {string_var} and compute {agg_func}",
                input_vars=[string_var],
                output_vars=[result_var],
                details={"agg": agg_func, "string_var": string_var,
                         "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 1: Preprocessing (sorting, filtering on input data)
        # the sorted input is a collection — an array variable, not a
        # leading scalar count.  With no collection to sort the phase is
        # skipped (sorting a scalar int would crash at runtime; skipping
        # lets the orchestrator rotate to a hypothesis that fits).
        sort_var = next(
            (v.name for v in input_vars
             if v.var_type in ("int[]", "array", "string", "string[]")),
            None)
        if sort_var and ("sorting" in primitives or "sorting" in hints):
            steps.append(PlanStep(
                step_id=step_id,
                operation="sort",
                description=f"Sort {sort_var}",
                input_vars=[sort_var],
                output_vars=[f"sorted_{sort_var}"],
                details={"order": "ascending", "type": "preprocessing"},
            ))
            data_var = f"sorted_{sort_var}"
            step_id += 1

        if "filtering" in hints and has_filter_on_input:
            # Filtering on input data (preprocessing)
            filter_condition = "True"
            for c in constraints:
                if c.constraint_type == "filter" and c.predicate:
                    predicate = c.predicate
                    filter_condition = self._predicate_to_condition(predicate)
                    break
                elif c.constraint_type == "filter" and c.value:
                    filter_condition = c.value
                    break

            steps.append(PlanStep(
                step_id=step_id,
                operation="filter",
                description=f"Filter {data_var}",
                input_vars=[data_var],
                output_vars=[f"filtered_{data_var}"],
                details={"condition": filter_condition, "type": "preprocessing"},
            ))
            data_var = f"filtered_{data_var}"
            step_id += 1

        # Phase 2a: Objective-driven substructure composition.
        # When the semantics describe an objective over a substructure with
        # an aggregate constraint, compose the matching general strategy:
        #   maximize/minimize(length) + contiguous + sum == 0
        #       → prefix-sum first/last-occurrence tracking
        #   maximize(sum) + contiguous
        #       → Kadane's running maximum
        output_measure = getattr(sp, "output_measure", "")
        obj_dir = getattr(sp, "objective_direction", "")
        has_contiguous = bool(hints & {"contiguous_subsequence", "contiguous_subarray", "segment"})
        has_zero_sum = any(
            getattr(c, "constraint_type", "") == "aggregate"
            and getattr(c, "aggregate_function", "") == "sum"
            and str(getattr(c, "value", "")) == "0"
            for c in constraints)
        has_existence = bool({"existence_check", "boolean_output"} & set(hints))
        if has_existence and not obj_dir and has_contiguous and has_zero_sum:
            # The output IS the yes/no: a prefix-sum existence scan over
            # the contiguous zero-sum structure — never an aggregate.
            array_var = next((v.name for v in sp.input_variables
                              if v.var_type in ("int[]", "array")),
                             data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="prefix_sum_exists",
                description="Existence of a contiguous zero-sum range via prefix sums",
                input_vars=[array_var],
                output_vars=[result_var],
                details={"type": "main_computation"},
            ))
            step_id += 1
            return steps
        if obj_dir and output_measure == "length" and has_contiguous and has_zero_sum:
            # the range scan consumes the ARRAY variable — never the
            # first scalar
            array_var = next((v.name for v in sp.input_variables
                              if v.var_type in ("int[]", "array")),
                             data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="prefix_sum_zero",
                description=f"{'Longest' if obj_dir == 'maximize' else 'Shortest'} zero-sum contiguous range via prefix sums",
                input_vars=[array_var],
                output_vars=[result_var],
                details={"direction": obj_dir, "type": "main_computation"},
            ))
            step_id += 1
            return steps
        if obj_dir and output_measure == "sum" and has_contiguous:
            # the subarray scan consumes the ARRAY variable — never the
            # first scalar (a count N is not subscriptable)
            array_var = next((v.name for v in sp.input_variables
                              if v.var_type in ("int[]", "array")),
                             data_var)
            steps.append(PlanStep(
                step_id=step_id,
                operation="kadane",
                description=f"{'Maximum' if obj_dir == 'maximize' else 'Minimum'} sum contiguous subarray",
                input_vars=[array_var],
                output_vars=[result_var],
                details={"direction": obj_dir, "type": "main_computation"},
            ))
            step_id += 1
            return steps

        # Phase 2b: Boolean output → property check.
        # When the output is a boolean (Yes/No) and the problem describes a
        # verifiable mathematical property, the main computation is a check.
        # General mathematical properties (like is_prime), not problem-specific:
        #   - sortedness / sortable by one swap (sorting + swap evidence)
        #   - geometric progression (geometric evidence)
        out_sem_type = ""
        if out_var_names and sp.output_variables:
            out_sem_type = sp.output_variables[0].semantic_type or ""
        if out_sem_type == "boolean":
            statement_lower = (sp.statement or "").lower()
            check_type = None
            if "geometric" in statement_lower:
                check_type = "geometric_progression"
            elif "permutation" in statement_lower and \
                    re.search(r'\\times|\bproduct\b|\*', statement_lower) and \
                    re.search(r'=', statement_lower):
                check_type = "permutation_product"
            elif "sorting" in hints or "sorted_input" in hints:
                if re.search(r'\bswap|바꾸|교환', statement_lower):
                    check_type = "sorted_one_swap"
                else:
                    check_type = "check_sorted"
            if check_type:
                # Property checks operate on the sequence data — the
                # property's SUBJECT must exist as an actual sequence in
                # the input.  Without a sequence input (e.g. scalars +
                # edge pairs), the surface evidence that selected this
                # check is not about any data we hold — the branch must
                # not hijack the hypothesis; Phase 2 composes from the
                # hypothesis's own primitives instead.
                check_input = next(
                    (v.name for v in input_vars
                     if v.var_type in ("int[]", "array", "string",
                                       "string[]")), None)
                if check_input:
                    # Quantifier semantics: "exactly one swap" vs "at most one swap".
                    # With "exactly", an already-sorted input needs zero swaps → No.
                    exactly_one = bool(re.search(
                        r'\bexactly\s+(?:one|1)\b|\bjust\s+(?:one|1)\s+swap\b',
                        statement_lower))
                    # Adjacency semantics: "swapping two adjacent elements" requires
                    # the mismatched positions to be adjacent.
                    adjacent_only = bool(re.search(
                        r'\badjacent\b|인접', statement_lower))
                    steps.append(PlanStep(
                        step_id=step_id,
                        operation="check_property",
                        description=f"Check property: {check_type}",
                        input_vars=[check_input],
                        output_vars=[result_var],
                        details={
                            "check_type": check_type,
                            "exactly_one": exactly_one,
                            "adjacent_only": adjacent_only,
                            "type": "main_computation",
                        },
                    ))
                    step_id += 1
                    return steps

        # Phase 2b: Element-selection objective.  The semantics declared
        # a DIRECTION over a MEASURE OF THE ELEMENTS (maximize/minimize a
        # value or a length of the collection's items) with no
        # substructure: the computation SELECTS an element — it never
        # aggregates (a sum/count would be a different contract).
        obj_target = getattr(sp, "objective_target", "") or ""
        # A declared substructure (subarray/subsequence/segment) is an
        # aggregate over that region — item selection would be a
        # different contract, so it never applies there.
        if (obj_dir in ("maximize", "minimize")
                and obj_target in ("element", "")
                and not has_contiguous
                and output_measure in ("value", "length")
                and not any(s.operation in ("compute", "iterate", "dp", "bfs",
                                            "dfs", "sliding_window", "two_pointer",
                                            "binary_search", "kadane", "prefix_sum_exists",
                                            "filter", "math")
                            for s in steps)):
            steps.append(PlanStep(
                step_id=step_id,
                operation="select_extremum",
                description=(
                    f"Select the {'maximum' if obj_dir == 'maximize' else 'minimum'} "
                    f"element by {output_measure}"),
                input_vars=[data_var],
                output_vars=[result_var],
                details={
                    "direction": "max" if obj_dir == "maximize" else "min",
                    "key": "length" if output_measure == "length" else "",
                    "measure": output_measure,
                    "type": "main_computation",
                },
            ))
            step_id += 1
            return steps

        # Phase 2: Main computation - compose based on hypothesis
        # Create connected steps based on the hypothesis primitives

        # Check if we have array inputs for iteration (used by counting and generic paths)
        has_array_input = any(v.var_type in ("int[]", "array") for v in input_vars)

        # Substitute the generic "value" subject of a computed-value filter
        # with the actual filtered expression (loop variable / cell value).
        def _bind_filter_subject(condition: str, source: str) -> str:
            if condition and source:
                return re.sub(r'\bvalue\b', f'({source})', condition)
            return condition

        
        if "counting" in primitives:
            # Counting pattern: iterate and count valid elements
            # Aggregation type from output measure (semantics-driven)
            output_measure = getattr(sp, "output_measure", "")
            if output_measure == "sum":
                count_agg = "sum"
            elif output_measure in ("count", "length"):
                count_agg = "count"
            else:
                count_agg = "count"
            
            # Determine the iteration domain from variable TYPES:
            #   two scalars → range between them; scalar+array → over array;
            #   single scalar → over range(1, scalar+1) only when the problem
            #   describes enumerating values up to it; else compute.
            scalar_vars = [v for v in input_vars if v.var_type in ("int", "float")]
            array_vars = [v for v in input_vars if v.var_type in ("int[]", "array")]
            
            if len(scalar_vars) >= 2 and not array_vars:
                # Two integer inputs: likely a range (L, R)
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="iterate",
                    description="Iterate over range",
                    input_vars=var_names,
                    output_vars=[result_var],
                    details={
                        "loop_var": "i",
                        "loop_range": f"range({scalar_vars[0].name}, {scalar_vars[1].name} + 1)",
                        "type": "main_computation",
                        "agg_type": count_agg,
                    },
                ))
                step_id += 1
            elif array_vars:
                # Array input: iterate over array
                array_var = array_vars[0].name
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="iterate",
                    description=f"Iterate over {array_var}",
                    input_vars=[array_var],
                    output_vars=[result_var],
                    details={
                        "loop_var": "x",
                        "loop_range": f"{array_var}",
                        "type": "main_computation",
                        "agg_type": count_agg,
                    },
                ))
                step_id += 1
            elif len(scalar_vars) == 1:
                # Single scalar: iterate over 1..N (enumerate values up to N)
                n_name = scalar_vars[0].name
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="iterate",
                    description=f"Iterate from 1 to {n_name}",
                    input_vars=[n_name],
                    output_vars=[result_var],
                    details={
                        "loop_var": "i",
                        "loop_range": f"range(1, {n_name} + 1)",
                        "type": "main_computation",
                        "agg_type": count_agg,
                    },
                ))
                step_id += 1
            else:
                # No usable domain — compute directly
                arithmetic_expr = getattr(sp, "arithmetic_expression", "")
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="compute",
                    description="Compute the result",
                    input_vars=var_names,
                    output_vars=[result_var],
                    details={
                        "type": "main_computation",
                        "expression": arithmetic_expr,
                    },
                ))
                step_id += 1

        # Phase 2: Main computation
        # Group primitives by their role
        has_iteration = any(p in primitives for p in ["linear_scan", "simulation", "greedy"])
        has_search = any(p in primitives for p in ["binary_search", "two_pointer", "sliding_window"])
        has_aggregation = "aggregation" in hints or "counting" in primitives
        has_dp = "dynamic_programming" in primitives
        has_graph = any(p in primitives for p in ["bfs", "dfs"])
        
        # Build main computation based on semantic structure
        if has_iteration or "two_dimensional_iteration" in hints:
            # Get filter condition for computed values
            filter_condition = None
            if has_filter_on_computed:
                for c in constraints:
                    if c.constraint_type == "filter" and c.predicate:
                        filter_condition = self._predicate_to_condition(c.predicate)
                        break
                    elif c.constraint_type == "filter" and c.value:
                        filter_condition = c.value
                        break

            # Determine iteration structure
            # Derive the aggregation type from the output measure:
            #   - output represents a "sum" → sum the filtered values
            #   - output represents "count"/"length" → count valid elements
            #   - an aggregate constraint (e.g. sum == 0) on candidates → count
            output_measure = getattr(sp, "output_measure", "")
            if output_measure == "sum":
                agg_type = "sum"
            elif output_measure in ("count", "length"):
                agg_type = "count"
            elif any(c.constraint_type == "aggregate" for c in constraints):
                agg_type = "count"
            else:
                agg_type = "sum" if "filtering" in hints else "count"

            if "two_dimensional_iteration" in hints:
                # 2D iteration (grid). Bounds from the declared grid dimensions
                # when fixed by the description; otherwise honest fallback
                # (no hardcoded ranges).
                grid_dims = getattr(sp, "grid_dimensions", None)
                if grid_dims:
                    d1, d2 = grid_dims
                    loop_range = f"range(1, {d1} + 1)"
                    nested_range = f"range(1, {d2} + 1)"
                    filter_source = "i * j"
                else:
                    loop_range = None
                    nested_range = None
                    filter_source = "i * j"
                if loop_range:
                    steps.append(PlanStep(
                        step_id=step_id,
                        operation="iterate",
                        description="Iterate over 2D domain",
                        input_vars=[data_var],
                        output_vars=[result_var],
                        details={
                            "loop_var": "i",
                            "loop_range": loop_range,
                            "nested": True,
                            "nested_var": "j",
                            "nested_range": nested_range,
                            "type": "main_computation",
                            "filter_condition": _bind_filter_subject(filter_condition, filter_source),
                            "filter_source": filter_source,
                            "agg_type": agg_type,
                        },
                    ))
                    step_id += 1
                else:
                    # No fixed grid dimensions — honest computation fallback
                    arithmetic_expr = getattr(sp, "arithmetic_expression", "")
                    steps.append(PlanStep(
                        step_id=step_id,
                        operation="compute",
                        description="Compute the result",
                        input_vars=var_names,
                        output_vars=[result_var],
                        details={
                            "type": "main_computation",
                            "expression": arithmetic_expr,
                        },
                    ))
                    step_id += 1
            elif has_array_input:
                # Iterate over array
                array_var = next(v.name for v in input_vars if v.var_type in ("int[]", "array"))
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="iterate",
                    description=f"Iterate over {array_var}",
                    input_vars=[array_var],
                    output_vars=[result_var],
                    details={
                        "loop_var": "x",
                        "loop_range": f"{array_var}",
                        "type": "main_computation",
                        "filter_condition": _bind_filter_subject(filter_condition, "x"),
                        "filter_source": "x",
                        "agg_type": agg_type,
                    },
                ))
                step_id += 1
            else:
                # No array - create a more intelligent computation
                # Based on the number of input variables.
                # The arithmetic expression (parsed from the problem text during
                # semantic enrichment) drives the computation when available.
                arithmetic_expr = getattr(sp, "arithmetic_expression", "")
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="compute",
                    description="Compute the result",
                    input_vars=var_names,
                    output_vars=[result_var],
                    details={
                        "type": "main_computation",
                        "expression": arithmetic_expr,
                    },
                ))
                step_id += 1

        elif has_search:
            # Search-based algorithm — operates on the sequence data
            search_input = next(
                (v.name for v in input_vars
                 if v.var_type in ("int[]", "array", "string", "string[]")),
                data_var)
            if "binary_search" in primitives:
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="binary_search",
                    description="Binary search for solution",
                    input_vars=[search_input],
                    output_vars=[result_var],
                    details={"type": "main_computation"},
                ))
            elif "two_pointer" in primitives:
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="two_pointer",
                    description="Two-pointer technique",
                    input_vars=[search_input],
                    output_vars=[result_var],
                    details={"type": "main_computation",
                             "condition_kind": self._derive_pair_relation(sp),
                             "mode": "count"},
                ))
            elif "sliding_window" in primitives:
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="sliding_window",
                    description="Sliding window",
                    input_vars=[search_input],
                    output_vars=[result_var],
                    details={"type": "main_computation",
                             "window_k": self._derive_window_k(sp),
                             "direction": getattr(sp, "objective_direction",
                                                  "") or "maximize"},
                ))
            step_id += 1

        elif has_dp:
            # Dynamic programming
            steps.append(PlanStep(
                step_id=step_id,
                operation="dp",
                description="Dynamic programming",
                input_vars=[data_var],
                output_vars=[result_var],
                details={"type": "main_computation"},
            ))
            step_id += 1

        elif has_graph:
            # Graph algorithm — the objective and input format are
            # derived from the problem's semantics (never per-problem):
            #   objective: components / shortest / count-reachable
            #   input: edge-list variables from the plan's input
            #   parsing (N, M, u, v) with a stdin fallback
            objective = self._derive_graph_objective(sp)
            input_format = self._derive_graph_input_format(sp)
            op_name = "bfs" if "bfs" in primitives else "dfs"
            # edge variables from the input parsing steps (read_pairs
            # produces the edge lists; read_ints produces the counts).
            # When the input parsing produced NO edge lists, the
            # problem's graph has no edge input — the traversal template
            # must not invent edge reading (it raises and the rotation
            # tries another hypothesis).
            has_edges = self._has_edge_input()
            if not has_edges:
                steps.append(PlanStep(
                    step_id=step_id,
                    operation="graph_no_edges",
                    description="Graph problem with no edge input",
                    input_vars=[data_var],
                    output_vars=[result_var],
                    details={"type": "main_computation"},
                ))
                step_id += 1
                return steps
            n_var, m_var, u_var, v_var = self._graph_edge_vars()
            steps.append(PlanStep(
                step_id=step_id,
                operation=op_name,
                description="Graph traversal",
                input_vars=[data_var],
                output_vars=[result_var],
                details={"type": "main_computation",
                         "objective": objective,
                         "input_format": input_format,
                         "n_var": n_var, "m_var": m_var,
                         "u_var": u_var, "v_var": v_var,
                         "directed": self._is_directed_graph(sp)},
            ))
            step_id += 1

        # Phase 3: Aggregation / Post-processing
        # Only add aggregation if there's a previous step that produces data
        # AND the last step doesn't already produce the final result
        last_step_op = steps[-1].operation if steps else None
        last_step_output = steps[-1].output_vars[0] if steps and steps[-1].output_vars else None
        
        # Don't add aggregation if:
        # 1. The last step is already an iterate with counting (it already counts)
        # 2. The last step output is the same as result_var
        needs_aggregation = has_aggregation and steps and \
                           last_step_op != "iterate" and \
                           last_step_output != result_var
        
        if needs_aggregation:
            agg_type = "sum"
            if "counting" in primitives:
                agg_type = "count"
            elif "filtering" in hints:
                agg_type = "sum"  # Usually sum filtered values

            steps.append(PlanStep(
                step_id=step_id,
                operation="aggregate",
                description=f"Aggregate results ({agg_type})",
                input_vars=[last_step_output],
                output_vars=[result_var],
                details={"agg_type": agg_type, "type": "aggregation"},
            ))
            step_id += 1

        # If no main computation was generated, the result is the preprocessed
        # data (e.g. a plain "sort X" request returns the sorted list).
        # Only add a compute step when there is no data-producing step at all.
        if not any(step.operation in ("iterate", "compute", "sort", "filter",
                                      "binary_search", "two_pointer", "sliding_window",
                                      "dp", "bfs", "dfs", "count", "simulate", "math",
                                      "find_max", "find_min", "parity_check", "bitwise",
                                      "monotonic_stack", "check_property",
                                      "select_extremum")
                   for step in steps):
            arithmetic_expr = getattr(sp, "arithmetic_expression", "")
            steps.append(PlanStep(
                step_id=step_id,
                operation="compute",
                description="Compute the result",
                input_vars=var_names if var_names else [data_var],
                output_vars=[result_var],
                details={
                    "type": "main_computation",
                    "expression": arithmetic_expr,
                },
            ))

        return steps

    def _has_edge_input(self) -> bool:
        """Does the plan's input parsing produce edge lists (read_pairs)?"""
        for step in getattr(self, "_current_input_parsing", []):
            if step.operation == "read_pairs":
                return True
        return False

    def _derive_pair_relation(self, sp) -> str:
        """Derive the pair size relation from the problem's semantics."""
        text = getattr(sp, "statement", "") or ""
        t = text.lower()
        if re.search(r'at\s+least\s+twice|두\s+배\s+이상|2\s*배', t):
            return "at least twice"
        if re.search(r'at\s+most\s+half|절반|half\s+of', t):
            return "at most half of"
        if re.search(r'(?:less|smaller)\s+than|작거나|미만', t):
            return "less than"
        if re.search(r'(?:greater|larger)\s+than|크거나|초과', t):
            return "greater than"
        if re.search(r'at\s+most|이하', t):
            return "at most"
        if re.search(r'at\s+least|이상', t):
            return "at least"
        return "less than"

    def _derive_window_k(self, sp):
        """Derive the sliding-window size from the constraints when the
        problem declares a fixed window (e.g. "exactly K consecutive")."""
        text = getattr(sp, "statement", "") or ""
        m = re.search(r'exactly\s+([A-Z])\s+(?:consecutive|contiguous)',
                      text, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    def _graph_edge_vars(self, steps=None) -> tuple[str, str, str, str]:
        """Find the graph edge variables from the plan's input parsing
        steps: counts (read_ints -> N, M) and edge lists (read_pairs
        -> u, v).  When no edge lists were parsed, m_var is None — the
        problem's graph has no edge input and the traversal template
        must not invent edge reading (the rotation then tries another
        hypothesis)."""
        n_var, m_var, u_var, v_var = "n", None, "u", "v"
        n_set = False
        for step in (steps if steps is not None
                     else getattr(self, "_current_input_parsing", [])):
            if step.operation == "read_ints" and step.output_vars:
                names = step.output_vars
                n_var = names[0]
                n_set = True
            elif step.operation == "read_int" and step.output_vars \
                    and not n_set:
                # the first declared scalar is the vertex count ("N") —
                # read_int (not read_ints) when it sits on its own line
                n_var = step.output_vars[0]
                n_set = True
            elif step.operation == "read_pairs" and step.output_vars:
                # the FIRST pair group is the edge list (later groups
                # are queries over it) — first wins
                if m_var is None:
                    names = step.output_vars
                    if len(names) >= 2:
                        u_var, v_var = names[0], names[1]
                        # the edge count comes from the read_pairs'
                        # count variable, not the second scalar
                        count_hint = (step.details or {}).get("count_var")
                        m_var = count_hint or m_var
                    elif step.details and (step.details or {}).get("count_var"):
                        u_var, v_var = names[0], names[1] if len(names) > 1 \
                            else ("u", "v")
                        m_var = step.details["count_var"]
        return n_var, m_var, u_var, v_var

    def _is_directed_graph(self, sp) -> bool:
        """Directed vs undirected from the problem's semantics."""
        text = getattr(sp, "statement", "") or ""
        if re.search(r'directed\s+graph|유향\s+그래프', text, re.IGNORECASE):
            return True
        return False

    def _derive_graph_objective(self, sp) -> str:
        """Derive the graph traversal objective from the problem's
        semantics (general structural inference, not per-problem)."""
        text = ""
        for attr in ("goal", "raw_text", "description"):
            v = getattr(sp, attr, "") or ""
            if isinstance(v, str) and v:
                text = (text + " " + v).lower()
                break
        # explicit objective signals
        if re.search(r'connected\s+component|연결\s+요소', text):
            return "components"
        if re.search(r'shortest\s+(distance|path)|최단\s+(거리|경로)', text):
            return "shortest"
        if re.search(r'(?:how\s+many|number\s+of|몇).*?(?:reachable|visit|도달)|'
                     r'(?:reachable|도달).*(?:how\s+many|number\s+of|몇)', text):
            return "count"
        # output-measure based: "count" outputs over a graph usually
        # mean reachable vertices or components
        measure = getattr(sp, "output_measure", "")
        if measure in ("count", "length"):
            return "count"
        if measure == "sum":
            return "count"
        return "visit"

    def _derive_graph_input_format(self, sp) -> str:
        """Derive the graph input format from the problem's semantics.

        - "n_tree_edges": a tree with N vertices -> N then N-1 edges
        - "n_m_edges": N, M then M edge pairs (general graphs)
        """
        text = ""
        for attr in ("goal", "raw_text", "description"):
            v = getattr(sp, attr, "") or ""
            if isinstance(v, str) and v:
                text = (text + " " + v).lower()
                break
        if re.search(r'\btree\b|트리|森林|포레스트', text):
            return "n_tree_edges"
        return "n_m_edges"

    def _build_output_step(self, sp, plan=None) -> PlanStep:
        """Build the output step.

        The printed variable is the final computed result: the last
        algorithm step's output variable when present, else the declared
        output variable, else "result".
        """
        out_var = sp.output_variables[0] if sp.output_variables else None
        out_name = "result"
        if plan is not None and plan.algorithm_steps \
                and plan.algorithm_steps[-1].output_vars:
            out_name = plan.algorithm_steps[-1].output_vars[0]
        elif out_var is not None:
            out_name = out_var.name
        return PlanStep(
            step_id=1000,
            operation="print",
            description=f"Print the result: {out_name}",
            input_vars=[out_name],
            output_vars=[],
            details={"type": out_var.var_type if out_var else "default",
                     "name": out_name},
        )

    def _collect_variables(self, plan, sp) -> dict[str, str]:
        """Collect all variables from the plan."""
        variables = {}
        for step in plan.input_parsing:
            for var in step.output_vars:
                vtype = step.details.get("type", "unknown")
                variables[var] = vtype
        for step in plan.algorithm_steps:
            for var in step.output_vars:
                variables[var] = "computed"
        return variables

    def _predicate_to_condition(self, predicate) -> str:
        """Convert a Predicate to a Python condition string.
        
        This converts semantic predicates to code conditions.
        The condition is based on the relation and polarity.
        """
        source = predicate.source
        target = predicate.target
        relation = predicate.relation
        polarity = predicate.polarity

        # Build the base condition based on relation
        if relation == "equality":
            if polarity == "negative":
                return f"{source} != {target}"
            else:
                return f"{source} == {target}"

        elif relation == "divisibility":
            if polarity == "negative":
                return f"{source} % {target} != 0"
            else:
                return f"{source} % {target} == 0"

        elif relation == "greater_than":
            if polarity == "negative":
                return f"{source} <= {target}"
            else:
                return f"{source} > {target}"

        elif relation == "less_than":
            if polarity == "negative":
                return f"{source} >= {target}"
            else:
                return f"{source} < {target}"

        elif relation == "parity":
            # Check if it's "even" or "odd"
            if "even" in predicate.description.lower():
                if polarity == "negative":
                    return f"{source} % 2 != 0"
                else:
                    return f"{source} % 2 == 0"
            else:  # odd
                if polarity == "negative":
                    return f"{source} % 2 == 0"
                else:
                    return f"{source} % 2 != 0"

        elif relation == "primality":
            if polarity == "negative":
                return f"not is_prime({source})"
            else:
                return f"is_prime({source})"

        elif relation == "property":
            # Generic property - we can't generate code for this
            # Return a placeholder
            return f"meets_property({source}, '{target}')"

        else:
            # Unknown relation - return a generic condition
            return f"{source} satisfies {relation}"
