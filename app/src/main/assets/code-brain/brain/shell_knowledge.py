"""Shell Knowledge - structured shell domain knowledge for RENA-BR.

This module provides shell concepts and relations as a Knowledge Graph domain.
It does NOT execute shell commands — it provides semantic understanding of
shell concepts for reasoning about natural language shell-related queries.

Design principle: Intent → Operation → Shell semantics → Platform constraints
→ Relevant knowledge → Reasoning → Generated response.
"""
from __future__ import annotations
from brain.concept_store import ConceptStore, Concept
from brain.relation_store import RelationStore, Relation, RelationType


# ═══════════════════════════════════════════════════════════════════════
# Shell Concept Definitions
# ═══════════════════════════════════════════════════════════════════════

_SHELL_CONCEPTS = [
    # ── Platforms ──
    Concept("platform_windows", "Windows", "shell_platform",
            "Microsoft Windows operating system",
            ["windows", "win", "microsoft", "ms", "윈도우"],
            languages=["powershell", "cmd"]),
    Concept("platform_linux", "Linux", "shell_platform",
            "Linux operating system family",
            ["linux", "unix", "posix", "리눅스"],
            languages=["bash", "sh", "zsh"]),
    Concept("platform_ubuntu", "Ubuntu", "shell_platform",
            "Ubuntu Linux distribution",
            ["ubuntu", "deb", "apt", "우분투"],
            languages=["bash"]),

    # ── Shell Languages ──
    Concept("shell_powershell", "PowerShell", "shell_language",
            "Microsoft PowerShell — object-based shell and scripting language",
            ["powershell", "pwsh", "ps", "파워셀"],
            languages=["powershell"]),
    Concept("shell_bash", "Bash", "shell_language",
            "Bourne Again Shell — POSIX-compatible command language",
            ["bash", "bourne", "sh", "shell", "바쉬"],
            languages=["bash", "sh"]),
    Concept("shell_cmd", "cmd", "shell_language",
            "Windows Command Prompt — legacy command interpreter",
            ["cmd", "command prompt", "dos", "명령 프롬프트"],
            languages=["cmd"]),

    # ── Abstract Shell Operations ──
    Concept("shell_command", "Shell Command", "shell_operation",
            "A discrete unit of work executed in a shell environment",
            ["command", "cmdlet", "명령", "명령어"]),
    Concept("shell_parameter", "Parameter", "shell_operation",
            "An option or argument that modifies shell command behavior",
            ["parameter", "option", "flag", "인수", "옵션"]),
    Concept("shell_pipeline", "Pipeline", "shell_operation",
            "A chain of commands where output of one feeds into the next",
            ["pipeline", "pipe chain", "파이프라인"]),
    Concept("shell_variable", "Variable", "shell_operation",
            "A named storage location for data in shell scripts",
            ["variable", "var", "변수"]),
    Concept("shell_environment", "Environment", "shell_operation",
            "Runtime configuration and system-level variables",
            ["environment", "env", "환경변수", "환경"]),
    Concept("shell_filesystem", "Filesystem", "shell_operation",
            "Organized hierarchy of files and directories",
            ["filesystem", "file system", "directory structure", "파일시스템",
             "폴더", "디렉토리", "파일"]),
    Concept("shell_process", "Process", "shell_operation",
            "A running instance of a program managed by the OS",
            ["process", "task", "프로세스", "실행"]),
    Concept("shell_permission", "Permission", "shell_operation",
            "Access control rules for files and directories",
            ["permission", "access", "mode", "권한", "읽기", "쓰기"]),
    Concept("shell_glob", "Glob", "shell_operation",
            "Wildcard pattern matching for file names",
            ["glob", "wildcard", "pattern match", "와일드카드"]),
    Concept("shell_function", "Function", "shell_operation",
            "A reusable block of shell code",
            ["function", "script block", "함수"]),
    Concept("shell_script", "Script", "shell_operation",
            "A file containing a sequence of shell commands",
            ["script", "script file", "스크립트"]),
    Concept("shell_redirection", "Redirection", "shell_operation",
            "Redirecting input/output streams between commands and files",
            ["redirection", "redirect", "stream", "리다이렉션"]),
    Concept("shell_object_pipeline", "Object Pipeline", "shell_operation",
            "PowerShell's mechanism of passing .NET objects between commands",
            ["object pipeline", "object-based pipeline", "객체 파이프라인"]),
    Concept("shell_stream_pipeline", "Stream Pipeline", "shell_operation",
            "Bash/POSIX mechanism of passing text streams between commands",
            ["stream pipeline", "text pipeline", "byte stream", "스트림 파이프라인"]),
    Concept("shell_recursion", "Recursive Traversal", "shell_operation",
            "Traversing directory trees including all subdirectories",
            ["recursive", "recursion", "subdirectory", "하위폴더", "재귀",
             "하위 디렉토리", "재귀적"]),
    Concept("shell_search", "File Search", "shell_operation",
            "Finding files by name, content, or attributes",
            ["search", "find", "lookup", "검색", "찾기", "파일 검색"]),
    Concept("shell_listing", "Directory Listing", "shell_operation",
            "Displaying contents of a directory",
            ["list", "listing", "ls", "dir", "목록", "폴더 내용",
             "디렉토리 목록", "내용 확인", "확인"]),
    Concept("shell_package_management", "Package Management", "shell_operation",
            "Installing, updating, and removing software packages",
            ["package manager", "apt", "yum", "choco", "패키지"]),
]

# ═══════════════════════════════════════════════════════════════════════
# Shell Relation Definitions
# ═══════════════════════════════════════════════════════════════════════

_SHELL_RELATIONS = [
    # ── Platform → Shell Language ──
    Relation("r_plat_ps", "platform_windows", "shell_powershell",
             RelationType.USES, {"context": "default_shell"}),
    Relation("r_plat_cmd", "platform_windows", "shell_cmd",
             RelationType.USES, {"context": "legacy_shell"}),
    Relation("r_plat_bash_linux", "platform_linux", "shell_bash",
             RelationType.USES, {"context": "default_shell"}),
    Relation("r_plat_bash_ubuntu", "platform_ubuntu", "shell_bash",
             RelationType.USES, {"context": "default_shell"}),
    Relation("r_ubuntu_is_linux", "platform_ubuntu", "platform_linux",
             RelationType.IS_A),

    # ── Shell Language → Shell Operations (capabilities) ──
    Relation("r_ps_has_cmd", "shell_powershell", "shell_command",
             RelationType.CONTAINS),
    Relation("r_ps_has_pipe", "shell_powershell", "shell_pipeline",
             RelationType.CONTAINS),
    Relation("r_ps_has_obj_pipe", "shell_powershell", "shell_object_pipeline",
             RelationType.CONTAINS),
    Relation("r_ps_has_var", "shell_powershell", "shell_variable",
             RelationType.CONTAINS),
    Relation("r_ps_has_env", "shell_powershell", "shell_environment",
             RelationType.CONTAINS),
    Relation("r_ps_has_fs", "shell_powershell", "shell_filesystem",
             RelationType.CONTAINS),
    Relation("r_ps_has_proc", "shell_powershell", "shell_process",
             RelationType.CONTAINS),
    Relation("r_ps_has_perm", "shell_powershell", "shell_permission",
             RelationType.CONTAINS),
    Relation("r_ps_has_redir", "shell_powershell", "shell_redirection",
             RelationType.CONTAINS),
    Relation("r_ps_has_func", "shell_powershell", "shell_function",
             RelationType.CONTAINS),
    Relation("r_ps_has_script", "shell_powershell", "shell_script",
             RelationType.CONTAINS),
    Relation("r_ps_has_search", "shell_powershell", "shell_search",
             RelationType.CONTAINS),
    Relation("r_ps_has_list", "shell_powershell", "shell_listing",
             RelationType.CONTAINS),
    Relation("r_ps_has_recurse", "shell_powershell", "shell_recursion",
             RelationType.CONTAINS),

    Relation("r_bash_has_cmd", "shell_bash", "shell_command",
             RelationType.CONTAINS),
    Relation("r_bash_has_pipe", "shell_bash", "shell_pipeline",
             RelationType.CONTAINS),
    Relation("r_bash_has_stream", "shell_bash", "shell_stream_pipeline",
             RelationType.CONTAINS),
    Relation("r_bash_has_var", "shell_bash", "shell_variable",
             RelationType.CONTAINS),
    Relation("r_bash_has_env", "shell_bash", "shell_environment",
             RelationType.CONTAINS),
    Relation("r_bash_has_fs", "shell_bash", "shell_filesystem",
             RelationType.CONTAINS),
    Relation("r_bash_has_proc", "shell_bash", "shell_process",
             RelationType.CONTAINS),
    Relation("r_bash_has_perm", "shell_bash", "shell_permission",
             RelationType.CONTAINS),
    Relation("r_bash_has_glob", "shell_bash", "shell_glob",
             RelationType.CONTAINS),
    Relation("r_bash_has_func", "shell_bash", "shell_function",
             RelationType.CONTAINS),
    Relation("r_bash_has_script", "shell_bash", "shell_script",
             RelationType.CONTAINS),
    Relation("r_bash_has_redir", "shell_bash", "shell_redirection",
             RelationType.CONTAINS),
    Relation("r_bash_has_search", "shell_bash", "shell_search",
             RelationType.CONTAINS),
    Relation("r_bash_has_list", "shell_bash", "shell_listing",
             RelationType.CONTAINS),
    Relation("r_bash_has_recurse", "shell_bash", "shell_recursion",
             RelationType.CONTAINS),
    Relation("r_bash_has_pkg", "shell_bash", "shell_package_management",
             RelationType.CONTAINS),

    # ── Pipeline types: similarity and distinction ──
    Relation("r_pipe_sim", "shell_object_pipeline", "shell_stream_pipeline",
             RelationType.SIMILAR_TO,
             {"difference": "objects vs text streams"}),
    Relation("r_obj_pipe_kind", "shell_object_pipeline", "shell_pipeline",
             RelationType.IS_A),
    Relation("r_stream_pipe_kind", "shell_stream_pipeline", "shell_pipeline",
             RelationType.IS_A),

    # ── Operation → Command mapping (semantic, not keyword) ──
    Relation("r_list_cmd_ps", "shell_listing", "shell_powershell",
             RelationType.IMPLEMENTS,
             {"approach": "Get-ChildItem or dir or ls"}),
    Relation("r_list_cmd_bash", "shell_listing", "shell_bash",
             RelationType.IMPLEMENTS,
             {"approach": "ls"}),
    Relation("r_search_cmd_ps", "shell_search", "shell_powershell",
             RelationType.IMPLEMENTS,
             {"approach": "Get-ChildItem -Recurse -Filter or Find-String"}),
    Relation("r_search_cmd_bash", "shell_search", "shell_bash",
             RelationType.IMPLEMENTS,
             {"approach": "find or grep"}),
    Relation("r_pipe_cmd_ps", "shell_pipeline", "shell_powershell",
             RelationType.IMPLEMENTS,
             {"approach": "Pipe (|) passes .NET objects between cmdlets"}),
    Relation("r_pipe_cmd_bash", "shell_pipeline", "shell_bash",
             RelationType.IMPLEMENTS,
             {"approach": "Pipe (|) passes text streams (stdout→stdin) between commands"}),
    Relation("r_obj_pipe_cmd_ps", "shell_object_pipeline", "shell_powershell",
             RelationType.IMPLEMENTS,
             {"approach": "Cmdlet | Cmdlet — objects with properties, not text"}),
    Relation("r_stream_pipe_cmd_bash", "shell_stream_pipeline", "shell_bash",
             RelationType.IMPLEMENTS,
             {"approach": "command1 | command2 — text streams, line-oriented"}),
    Relation("r_perm_cmd_ps", "shell_permission", "shell_powershell",
             RelationType.IMPLEMENTS,
             {"approach": "Get-Acl / Set-Acl"}),
    Relation("r_perm_cmd_bash", "shell_permission", "shell_bash",
             RelationType.IMPLEMENTS,
             {"approach": "chmod / chown / stat"}),
    Relation("r_env_cmd_ps", "shell_environment", "shell_powershell",
             RelationType.IMPLEMENTS,
             {"approach": "$env:VAR or Get-ChildItem Env:"}),
    Relation("r_env_cmd_bash", "shell_environment", "shell_bash",
             RelationType.IMPLEMENTS,
             {"approach": "echo $VAR or env or printenv"}),
    Relation("r_proc_cmd_ps", "shell_process", "shell_powershell",
             RelationType.IMPLEMENTS,
             {"approach": "Get-Process / Stop-Process"}),
    Relation("r_proc_cmd_bash", "shell_process", "shell_bash",
             RelationType.IMPLEMENTS,
             {"approach": "ps / top / kill / pgrep"}),
    Relation("r_recurse_cmd_ps", "shell_recursion", "shell_powershell",
             RelationType.IMPLEMENTS,
             {"approach": "Get-ChildItem -Recurse"}),
    Relation("r_recurse_cmd_bash", "shell_recursion", "shell_bash",
             RelationType.IMPLEMENTS,
             {"approach": "find / -type d or tree"}),
    Relation("r_recurseRequires_search", "shell_recursion", "shell_search",
             RelationType.REQUIRES),
    Relation("r_recurseRequires_fs", "shell_recursion", "shell_filesystem",
             RelationType.REQUIRES),
    Relation("r_searchRequires_fs", "shell_search", "shell_filesystem",
             RelationType.REQUIRES),
    Relation("r_listRequires_fs", "shell_listing", "shell_filesystem",
             RelationType.REQUIRES),
    Relation("r_permRequires_fs", "shell_permission", "shell_filesystem",
             RelationType.REQUIRES),
    Relation("r_procUses_env", "shell_process", "shell_environment",
             RelationType.USES),
    Relation("r_env_part_fs", "shell_environment", "shell_filesystem",
             RelationType.PART_OF),
    Relation("r_cmd_uses_param", "shell_command", "shell_parameter",
             RelationType.USES),
    Relation("r_cmd_uses_redir", "shell_command", "shell_redirection",
             RelationType.USES),
    Relation("r_func_uses_cmd", "shell_function", "shell_command",
             RelationType.CONTAINS),
    Relation("r_script_uses_func", "shell_script", "shell_function",
             RelationType.CONTAINS),
    Relation("r_script_uses_cmd", "shell_script", "shell_command",
             RelationType.CONTAINS),
]


def seed_shell_concepts(store: ConceptStore) -> None:
    """Add all shell knowledge concepts to the concept store."""
    for concept in _SHELL_CONCEPTS:
        store.add(concept)


def seed_shell_relations(store: RelationStore) -> None:
    """Add all shell knowledge relations to the relation store."""
    for relation in _SHELL_RELATIONS:
        store.add(relation)


def seed_shell_knowledge(
    concept_store: ConceptStore,
    relation_store: RelationStore,
) -> None:
    """Convenience: seed both shell concepts and relations."""
    seed_shell_concepts(concept_store)
    seed_shell_relations(relation_store)


# ═══════════════════════════════════════════════════════════════════════
# Shell Domain Query Expansion
# ═══════════════════════════════════════════════════════════════════════

# Korean shell term → English equivalents (only truly ambiguous terms)
_SHELL_EXPANSIONS = {
    "폴더": ["directory"],
    "디렉토리": ["folder"],
    "환경변수": ["environment variable", "env"],
    "환경": ["environment"],
    "권한": ["permission"],
    "프로세스": ["process"],
    "재귀": ["recursive"],
    "재귀적": ["recursive"],
    "파이프라인": ["pipeline"],
    "명령어": ["command"],
    "패키지": ["package"],
}


def expand_shell_query(query: str) -> str:
    """Expand Korean shell terms in a query with English equivalents.

    Adds English equivalents for Korean shell-domain terms found in the query.
    Uses word-boundary matching to avoid false expansions (e.g., "폴더" in
    "폴더 내용" expands correctly, but doesn't affect unrelated words).
    """
    expanded_parts = [query]
    text_lower = query.lower()
    for ko_term, en_equivalents in _SHELL_EXPANSIONS.items():
        if ko_term in text_lower:
            # Only add English equivalents that aren't already in the query
            for eq in en_equivalents:
                if eq not in text_lower:
                    expanded_parts.append(eq)
    return " ".join(expanded_parts)


# ═══════════════════════════════════════════════════════════════════════
# Shell Reasoning
# ═══════════════════════════════════════════════════════════════════════

import re

# Query type patterns (Korean + English)
_HOWTO_PATTERNS = re.compile(
    r'(방법|어떻게|어캐|하는|사용|이용|how\s+to|command|명령)', re.IGNORECASE
)
_WHATIS_PATTERNS = re.compile(
    r'(무엇|什麽|뭔|뭐|설명|개념|이란|what\s+is|explain|concept|개념)', re.IGNORECASE
)
_DIFFERENCE_PATTERNS = re.compile(
    r'(차이|다른|비교|틀린|difference|compare|vs|versus)', re.IGNORECASE
)
_CONCEPT_PATTERNS = re.compile(
    r'(개념|의미|뜻|하는\s*일|purpose|meaning|role|기능)', re.IGNORECASE
)


def _detect_query_type(query: str) -> str:
    """Detect the intent type of a shell query."""
    if _DIFFERENCE_PATTERNS.search(query):
        return "difference"
    if _HOWTO_PATTERNS.search(query):
        return "howto"
    if _WHATIS_PATTERNS.search(query):
        return "what_is"
    if _CONCEPT_PATTERNS.search(query):
        return "concept"
    return "general"


def _find_approach(op_id: str, shell_id: str, concepts, relations) -> str | None:
    """Find the specific command/approach for an operation on a given shell."""
    rels = relations.find_relations(
        source_id=op_id,
        target_id=shell_id,
        relation_type=RelationType.IMPLEMENTS,
    )
    for rel in rels:
        return rel.properties.get("approach")
    # Fallback: find any IMPLEMENTS relation for this operation
    rels = relations.find_relations(
        source_id=op_id,
        relation_type=RelationType.IMPLEMENTS,
    )
    for rel in rels:
        if shell_id in rel.target_id or rel.target_id in shell_id:
            return rel.properties.get("approach")
    return None


def _gather_operation_context(op_id: str, concepts, relations) -> dict:
    """Gather full context for an operation: description, requires, approaches."""
    ctx = {"description": "", "requires": [], "approaches": {}}
    concept = concepts.get(op_id)
    if concept:
        ctx["description"] = concept.description

    # Find REQUIRES relations
    req_rels = relations.find_relations(
        source_id=op_id, relation_type=RelationType.REQUIRES
    )
    for rel in req_rels:
        req_concept = concepts.get(rel.target_id)
        if req_concept:
            ctx["requires"].append(req_concept.name)

    # Find IMPLEMENTS relations (approaches per shell)
    impl_rels = relations.find_relations(
        source_id=op_id, relation_type=RelationType.IMPLEMENTS
    )
    for rel in impl_rels:
        approach = rel.properties.get("approach", "")
        shell_concept = concepts.get(rel.target_id)
        if shell_concept:
            ctx["approaches"][shell_concept.name] = approach

    return ctx


def _generate_response(query: str, query_type: str, detected: dict,
                       concepts, relations) -> str:
    """Generate a natural language response from reasoning results."""
    platform = detected.get("platform")
    shell = detected.get("shell")
    operations = detected.get("operations", [])

    plat_name = ""
    if platform:
        pc = concepts.get(platform)
        plat_name = pc.name if pc else platform

    shell_name = ""
    if shell:
        sc = concepts.get(shell)
        shell_name = sc.name if sc else shell

    op_names = []
    op_ctxs = []
    for op_id in operations:
        oc = concepts.get(op_id)
        name = oc.name if oc else op_id
        op_names.append(name)
        op_ctxs.append(_gather_operation_context(op_id, concepts, relations))

    # ── Difference query ──
    if query_type == "difference" and len(operations) >= 1:
        parts = []
        if shell_name:
            parts.append(f"{shell_name}에서")
        parts.append(f"**{op_names[0]}** 관련 개념:")
        if op_ctxs[0]["description"]:
            parts.append(f"  - {op_ctxs[0]['description']}")

        # Compare approaches across shells
        all_approaches = {}
        for ctx in op_ctxs:
            for shell_n, approach in ctx["approaches"].items():
                all_approaches[shell_n] = approach

        if len(all_approaches) >= 2:
            parts.append("  차이점:")
            for shell_n, approach in all_approaches.items():
                parts.append(f"    - {shell_n}: {approach}")
        elif all_approaches:
            for shell_n, approach in all_approaches.items():
                parts.append(f"  {shell_n}에서: {approach}")

        if op_ctxs[0]["requires"]:
            parts.append(f"  필요 개념: {', '.join(op_ctxs[0]['requires'])}")

        return "\n".join(parts)

    # ── How-to query ──
    if query_type == "howto" and operations:
        parts = []
        context = f"{shell_name}에서 " if shell_name else ""
        context += f"{plat_name} 환경에서 " if plat_name and not shell_name else ""

        for i, (name, ctx) in enumerate(zip(op_names, op_ctxs)):
            parts.append(f"**{context}{name}**:")
            if ctx["description"]:
                parts.append(f"  개념: {ctx['description']}")

            # Find the approach for the detected shell
            approach = None
            if shell:
                approach = _find_approach(operations[i], shell, concepts, relations)
            if approach:
                parts.append(f"  방법: {approach}")
            elif ctx["approaches"]:
                for shell_n, appr in ctx["approaches"].items():
                    parts.append(f"  {shell_n}: {appr}")

            if ctx["requires"]:
                parts.append(f"  필요: {', '.join(ctx['requires'])}")

        return "\n".join(parts)

    # ── What-is / concept query ──
    if query_type in ("what_is", "concept") and operations:
        parts = []
        for name, ctx in zip(op_names, op_ctxs):
            parts.append(f"**{name}**: {ctx['description']}")
            if ctx["requires"]:
                parts.append(f"  관련 개념: {', '.join(ctx['requires'])}")
            if ctx["approaches"]:
                apps = "; ".join(f"{s}: {a}" for s, a in ctx["approaches"].items())
                parts.append(f"  구현 예시: {apps}")
        return "\n".join(parts)

    # ── General / fallback ──
    parts = []
    if plat_name or shell_name:
        env = f"{plat_name} {shell_name}".strip()
        parts.append(f"환경: {env}")

    if operations:
        for name, ctx in zip(op_names, op_ctxs):
            parts.append(f"**{name}**: {ctx['description']}")
            if ctx["approaches"]:
                apps = "; ".join(f"{s}: {a}" for s, a in ctx["approaches"].items())
                parts.append(f"  방법: {apps}")
    else:
        parts.append("요청에서 특정 셸 작업을 찾지 못했습니다.")

    return "\n".join(parts)


def reason_about_shell(
    query: str,
    concepts: ConceptStore,
    relations: RelationStore,
) -> dict:
    """Reason about a natural language shell query using the knowledge graph.

    Returns a structured dict:
    {
        "platform": str | None,
        "shell": str | None,
        "operations": list[str],
        "query_type": str,           # "howto" | "what_is" | "difference" | "concept" | "general"
        "concepts": list[str],
        "relations": list[str],
        "explanation": str,          # human-readable reasoning trace
        "response": str,             # natural language answer
    }
    """
    result = {
        "platform": None,
        "shell": None,
        "operations": [],
        "query_type": _detect_query_type(query),
        "concepts": [],
        "relations": [],
        "explanation": "",
        "response": "",
    }

    # Step 0: Expand query with English equivalents for better scoring
    expanded_query = expand_shell_query(query)

    # Step 1: Search shell concepts from the expanded query
    shell_concepts = concepts.search_with_scores(expanded_query, category="shell")
    if not shell_concepts:
        shell_concepts = concepts.search_with_scores(expanded_query, category="shell_platform")
        shell_concepts += concepts.search_with_scores(expanded_query, category="shell_language")
        shell_concepts += concepts.search_with_scores(expanded_query, category="shell_operation")
        seen = set()
        deduped = []
        for c, s in shell_concepts:
            if c.id not in seen:
                seen.add(c.id)
                deduped.append((c, s))
        shell_concepts = deduped

    # Step 2: Classify detected concepts
    platforms = []
    shells = []
    operations = []
    for concept, score in shell_concepts:
        if concept.category == "shell_platform":
            platforms.append(concept)
        elif concept.category == "shell_language":
            shells.append(concept)
        elif concept.category == "shell_operation":
            operations.append(concept)

    if platforms:
        result["platform"] = platforms[0].id
    if shells:
        shell_pick = shells[0].id
        if platforms:
            # The shell must be CONSISTENT with the detected platform
            # (Linux → bash/sh/zsh, Windows → powershell/cmd). The
            # platform's declared languages decide; a shell contradicting
            # the platform (e.g. "Command" in a Linux question) is
            # rejected.
            pc = concepts.get(platforms[0].id)
            plat_langs = set(pc.languages or []) if pc else set()
            if plat_langs:
                consistent = [sh for sh in shells
                              if set(sh.languages or []) & plat_langs]
                if consistent:
                    shell_pick = consistent[0].id
        result["shell"] = shell_pick
    result["operations"] = [op.id for op in operations]

    # Step 2b: Infer operations from shell/platform via CONTAINS relations
    if not result["operations"] and (result["shell"] or result["platform"]):
        source_id = result["shell"] or result["platform"]
        contains_rels = relations.find_relations(
            source_id=source_id,
            relation_type=RelationType.CONTAINS,
        )
        text_lower = query.lower()
        scored_ops = []
        for rel in contains_rels:
            op_concept = concepts.get(rel.target_id)
            if not op_concept or op_concept.category != "shell_operation":
                continue
            op_score = 0
            for tag in op_concept.tags:
                if tag.lower() in text_lower:
                    op_score += 4.0
                elif any(w in text_lower for w in tag.split() if len(w) >= 3):
                    op_score += 1.5
            for word in text_lower.split():
                if len(word) >= 3 and word in op_concept.description.lower():
                    op_score += 0.5
            if op_score > 0:
                scored_ops.append((op_concept.id, op_score))

        implements_rels = relations.find_relations(
            target_id=source_id,
            relation_type=RelationType.IMPLEMENTS,
        )
        for rel in implements_rels:
            op_concept = concepts.get(rel.source_id)
            if not op_concept or op_concept.category != "shell_operation":
                continue
            op_score = 0
            for tag in op_concept.tags:
                if tag.lower() in text_lower:
                    op_score += 4.0
                elif any(w in text_lower for w in tag.split() if len(w) >= 3):
                    op_score += 1.5
            if op_score > 0:
                scored_ops.append((op_concept.id, op_score))

        scored_ops.sort(key=lambda x: -x[1])
        result["operations"] = [op_id for op_id, _ in scored_ops[:5]]

    # Step 2c: For difference queries, find operations from BOTH shells
    if result["query_type"] == "difference" and result["shell"]:
        # Find all operations the detected shell supports
        all_ops_rels = relations.find_relations(
            source_id=result["shell"],
            relation_type=RelationType.CONTAINS,
        )
        for rel in all_ops_rels:
            if rel.target_id not in result["operations"]:
                op_concept = concepts.get(rel.target_id)
                if op_concept and op_concept.category == "shell_operation":
                    text_lower = expanded_query.lower()
                    for tag in op_concept.tags:
                        if tag.lower() in text_lower:
                            result["operations"].append(rel.target_id)
                            break

    # Step 3: Traverse relations for each detected concept
    for concept, _ in shell_concepts:
        related = relations.find_related(concept.id)
        for rel in related:
            other_id = rel.target_id if rel.source_id == concept.id else rel.source_id
            other_concept = concepts.get(other_id)
            rel_desc = f"{rel.source_id} --{rel.relation_type.value}--> {rel.target_id}"
            result["relations"].append(rel_desc)
            if other_id not in result["concepts"] and other_concept:
                result["concepts"].append(other_id)

    # Step 4: Infer platform from shell
    if result["shell"] and not result["platform"]:
        platform_rels = relations.find_relations(
            target_id=result["shell"],
            relation_type=RelationType.USES,
        )
        for rel in platform_rels:
            plat_concept = concepts.get(rel.source_id)
            if plat_concept and plat_concept.category == "shell_platform":
                result["platform"] = rel.source_id
                break

    # Step 5: Infer shell from platform
    if result["platform"] and not result["shell"]:
        shell_rels = relations.find_relations(
            source_id=result["platform"],
            relation_type=RelationType.USES,
        )
        for rel in shell_rels:
            shell_concept = concepts.get(rel.target_id)
            if shell_concept and shell_concept.category == "shell_language":
                result["shell"] = rel.target_id
                break

    # Step 6: Generate explanation trace
    parts = []
    if result["platform"]:
        p = concepts.get(result["platform"])
        parts.append(f"Platform: {p.name if p else result['platform']}")
    if result["shell"]:
        s = concepts.get(result["shell"])
        parts.append(f"Shell: {s.name if s else result['shell']}")
    if result["operations"]:
        op_names = []
        for op_id in result["operations"]:
            oc = concepts.get(op_id)
            op_names.append(oc.name if oc else op_id)
        parts.append(f"Operations: {', '.join(op_names)}")
    parts.append(f"Query type: {result['query_type']}")
    result["explanation"] = " | ".join(parts) if parts else "No shell knowledge matched"

    # Step 7: Generate natural language response
    result["response"] = _generate_response(
        query, result["query_type"], result, concepts, relations
    )

    return result
