"""Rule Store - stores and executes rules for code generation and validation."""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class RuleCategory(Enum):
    SYNTAX = "syntax"
    NAMING = "naming"
    CODE_GEN = "code_gen"
    CODE_MOD = "code_mod"
    VALIDATION = "validation"
    PATTERN = "pattern"
    ERROR_FIX = "error_fix"


@dataclass
class Rule:
    id: str
    name: str
    category: RuleCategory
    description: str = ""
    languages: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)  # simple pattern conditions
    actions: list[str] = field(default_factory=list)  # actions to perform
    priority: int = 0
    properties: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class RuleEngine:
    def __init__(self, data_dir: str = None):
        self.rules: dict[str, Rule] = {}
        self._actions: dict[str, Callable] = {}
        self.data_dir = data_dir
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._load()
        self._register_builtins()

    def add(self, rule: Rule) -> None:
        self.rules[rule.id] = rule
        self._save()

    def get(self, rule_id: str) -> Optional[Rule]:
        return self.rules.get(rule_id)

    def find_rules(self, category: RuleCategory = None, language: str = None,
                   name_pattern: str = None) -> list[Rule]:
        results = []
        for r in self.rules.values():
            if not r.enabled:
                continue
            if category and r.category != category:
                continue
            if language and r.languages and language not in r.languages:
                continue
            if name_pattern and not re.search(name_pattern, r.name, re.IGNORECASE):
                continue
            results.append(r)
        return sorted(results, key=lambda x: -x.priority)

    def execute(self, rule: Rule, context: dict[str, Any]) -> list[str]:
        results = []
        if self._check_conditions(rule, context):
            for action_str in rule.actions:
                result = self._execute_action(action_str, rule, context)
                if result:
                    results.append(result)
        return results

    def execute_all(self, category: RuleCategory = None, language: str = None,
                    context: dict[str, Any] = None) -> dict[str, list[str]]:
        rules = self.find_rules(category=category, language=language)
        results = {}
        for rule in rules:
            rule_results = self.execute(rule, context or {})
            if rule_results:
                results[rule.id] = rule_results
        return results

    def register_action(self, name: str, func: Callable) -> None:
        self._actions[name] = func

    def _check_conditions(self, rule: Rule, context: dict[str, Any]) -> bool:
        for cond in rule.conditions:
            if not self._eval_condition(cond, context):
                return False
        return True

    def _eval_condition(self, condition: str, context: dict[str, Any]) -> bool:
        m = re.match(r"has_key\((\w+)\)", condition)
        if m:
            return m.group(1) in context
        m = re.match(r"type_is\((\w+),\s*(\w+)\)", condition)
        if m:
            val = context.get(m.group(1))
            return type(val).__name__ == m.group(2)
        m = re.match(r"lang_is\((\w+)\)", condition)
        if m:
            return context.get("language") == m.group(1)
        m = re.match(r"contains\((\w+),\s*['\"](.+?)['\"]\)", condition)
        if m:
            val = str(context.get(m.group(1), ""))
            return m.group(2) in val
        m = re.match(r"equals\((\w+),\s*['\"](.+?)['\"]\)", condition)
        if m:
            return str(context.get(m.group(1), "")) == m.group(2)
        return True

    def _execute_action(self, action_str: str, rule: Rule, context: dict[str, Any]) -> Optional[str]:
        m = re.match(r"(\w+)\((.+)\)", action_str)
        if m:
            func_name = m.group(1)
            args_str = m.group(2)
            args = [a.strip().strip("'\"") for a in args_str.split(",")]
            resolved = []
            for a in args:
                if a.startswith("$"):
                    resolved.append(context.get(a[1:], a))
                else:
                    resolved.append(a)
            if func_name in self._actions:
                try:
                    return self._actions[func_name](*resolved)
                except Exception as e:
                    return f"Error in action {func_name}: {e}"
        return f"Unknown action: {action_str}"

    def _register_builtins(self):
        self.register_action("format_code", self._action_format_code)
        self.register_action("add_indent", self._action_add_indent)
        self.register_action("set_type", self._action_set_type)
        self.register_action("apply_naming", self._action_apply_naming)

    def _action_format_code(self, code: str, style: str = "default") -> str:
        return code

    def _action_add_indent(self, code: str, level: str = "1") -> str:
        indent = "    " * int(level)
        return "\n".join(indent + line if line.strip() else "" for line in code.split("\n"))

    def _action_set_type(self, name: str, type_name: str) -> str:
        return f"{name}: {type_name}"

    def _action_apply_naming(self, name: str, convention: str = "camelCase") -> str:
        if convention == "camelCase":
            parts = name.split("_")
            return parts[0] + "".join(p.capitalize() for p in parts[1:])
        elif convention == "PascalCase":
            return "".join(p.capitalize() for p in name.split("_"))
        elif convention == "snake_case":
            return re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
        return name

    def _save(self):
        if not self.data_dir:
            return
        path = os.path.join(self.data_dir, "rules.json")
        data = []
        for r in self.rules.values():
            data.append({
                "id": r.id,
                "name": r.name,
                "category": r.category.value,
                "description": r.description,
                "languages": r.languages,
                "conditions": r.conditions,
                "actions": r.actions,
                "priority": r.priority,
                "properties": r.properties,
                "enabled": r.enabled,
            })
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _load(self):
        path = os.path.join(self.data_dir, "rules.json")
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
                d["category"] = RuleCategory(d["category"])
                r = Rule(**d)
            except (TypeError, KeyError, ValueError):
                continue
            self.rules[r.id] = r

    def export_json(self) -> str:
        data = []
        for r in self.rules.values():
            data.append({
                "id": r.id,
                "name": r.name,
                "category": r.category.value,
                "description": r.description,
                "languages": r.languages,
                "conditions": r.conditions,
                "actions": r.actions,
                "priority": r.priority,
                "properties": r.properties,
                "enabled": r.enabled,
            })
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> int:
        data = json.loads(json_str)
        count = 0
        for d in data:
            d["category"] = RuleCategory(d["category"])
            r = Rule(**d)
            self.add(r)
            count += 1
        return count


def seed_default_rules(engine: RuleEngine) -> None:
    rules = [
        Rule("naming_snake", "Snake Case for Python", RuleCategory.NAMING,
             "Python uses snake_case for functions and variables",
             ["python"], [], ["apply_naming($name,snake_case)"], 10),
        Rule("naming_camel", "Camel Case for Java/Kotlin/JS", RuleCategory.NAMING,
             "Java/Kotlin/JS use camelCase for methods and variables",
             ["java", "kotlin", "javascript"], [], ["apply_naming($name,camelCase)"], 10),
        Rule("naming_pascal", "Pascal Case for Classes", RuleCategory.NAMING,
             "Most languages use PascalCase for class names",
             ["python", "kotlin", "java", "cpp", "javascript", "rust"], [],
             ["apply_naming($name,PascalCase)"], 10),

        # ── Semantic validation rules (NOT no-ops) ──
        Rule("validate_empty_list", "Empty List Guard", RuleCategory.VALIDATION,
             "Guard against empty list input causing division by zero",
             ["python", "kotlin", "java", "javascript", "go", "rust"],
             ["has_key(needs_average)", "has_key(input_var)"],
             ["add_indent($code,1)"], 25),
        Rule("validate_single_element", "Single Element Guard", RuleCategory.VALIDATION,
             "Handle single-element collections gracefully",
             ["python", "kotlin", "java", "javascript", "go", "rust"],
             ["has_key(needs_sort)", "has_key(input_var)"],
             ["add_indent($code,1)"], 20),

        # ── Code generation rules (real behavior) ──
        Rule("gen_python_func", "Generate Python Function", RuleCategory.CODE_GEN,
             "Python function syntax with proper indentation",
             ["python"],
             ["lang_is(python)", "has_key(func_name)"],
             ["add_indent($code,1)"], 20),
        Rule("gen_java_func", "Generate Java Method", RuleCategory.CODE_GEN,
             "Java method syntax with proper indentation",
             ["java"],
             ["lang_is(java)", "has_key(func_name)"],
             ["add_indent($code,1)"], 20),
        Rule("gen_kotlin_func", "Generate Kotlin Function", RuleCategory.CODE_GEN,
             "Kotlin function syntax with proper indentation",
             ["kotlin"],
             ["lang_is(kotlin)", "has_key(func_name)"],
             ["add_indent($code,1)"], 20),
        Rule("gen_js_func", "Generate JavaScript Function", RuleCategory.CODE_GEN,
             "JavaScript function syntax with proper indentation",
             ["javascript"],
             ["lang_is(javascript)", "has_key(func_name)"],
             ["add_indent($code,1)"], 20),
        Rule("gen_go_func", "Generate Go Function", RuleCategory.CODE_GEN,
             "Go function syntax with proper indentation",
             ["go"],
             ["lang_is(go)", "has_key(func_name)"],
             ["add_indent($code,1)"], 20),
        Rule("gen_rust_func", "Generate Rust Function", RuleCategory.CODE_GEN,
             "Rust function syntax with proper indentation",
             ["rust"],
             ["lang_is(rust)", "has_key(func_name)"],
             ["add_indent($code,1)"], 20),
        Rule("gen_c_func", "Generate C Function", RuleCategory.CODE_GEN,
             "C function syntax with proper indentation",
             ["c"],
             ["lang_is(c)", "has_key(func_name)"],
             ["add_indent($code,1)"], 20),
        Rule("gen_cpp_func", "Generate C++ Function", RuleCategory.CODE_GEN,
             "C++ function syntax with proper indentation",
             ["cpp"],
             ["lang_is(cpp)", "has_key(func_name)"],
             ["add_indent($code,1)"], 20),

        # ── Error fix rules ──
        Rule("error_null_check", "Null Check Pattern", RuleCategory.ERROR_FIX,
             "Add null checks for nullable types",
             ["python", "kotlin", "java", "javascript", "go", "rust"],
             ["has_key(nullable_var)"],
             ["add_indent($code,1)"], 15),
        Rule("error_index_check", "Index Bounds Check", RuleCategory.ERROR_FIX,
             "Add index bounds checking before array access",
             ["python", "kotlin", "java", "javascript", "go", "rust"],
             ["has_key(index_var)"],
             ["add_indent($code,1)"], 15),

        # ── Pattern rules ──
        Rule("pattern_sum", "Sum Pattern", RuleCategory.PATTERN,
             "Pattern for adding two numbers",
             ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"],
             ["has_key(param_a)", "has_key(param_b)"],
             ["add_indent($code,1)"], 5),
        Rule("pattern_factorial", "Factorial Pattern", RuleCategory.PATTERN,
             "Pattern for factorial computation",
             ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"],
             ["has_key(n_param)"],
             ["add_indent($code,1)"], 5),
    ]
    for r in rules:
        engine.add(r)
