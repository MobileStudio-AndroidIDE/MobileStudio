"""Planner - converts reasoning results into structured IR for all intent types."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from brain.reasoner import ReasoningResult, StructuredIntent, ClassInfo
from code.ir.ir_types import (
    IRNode, IRNodeType, create_function_ir, create_return_ir, create_binary_op_ir,
    create_reference_ir, create_literal_ir, create_if_ir, create_call_ir,
    create_variable_ir, create_for_each_ir, create_for_ir, create_index_ir,
    create_length_ir, create_assignment_ir, create_class_ir,
)


@dataclass
class PlanAction:
    action_type: str
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[int] = field(default_factory=list)
    description: str = ""


@dataclass
class Plan:
    id: str
    description: str
    actions: list[PlanAction]
    language: str = ""
    reasoning_trace: ReasoningResult = None
    status: str = "pending"


# Canonical algorithm signatures (single registry; mirrored by
# Planner._structured_to_func_info and asserted by tests/test_all.py).
ALGORITHM_PATTERNS: dict[str, dict] = {
    "add": {"params": [("a", "int"), ("b", "int")], "return_type": "int"},
    "multiply": {"params": [("a", "int"), ("b", "int")], "return_type": "int"},
    "concat_strings": {"params": [("a", "string"), ("b", "string")],
                       "return_type": "string"},
    "find_max": {"params": [("items", "List<int>")], "return_type": "int"},
    "find_min": {"params": [("items", "List<int>")], "return_type": "int"},
    "sum_list": {"params": [("items", "List<int>")], "return_type": "int"},
    "average_list": {"params": [("items", "List<int>")], "return_type": "float"},
    "search_in_list": {"params": [("items", "List<int>"), ("target", "int")],
                       "return_type": "bool"},
    "search_in_string": {"params": [("text", "string"), ("pattern", "string")],
                         "return_type": "bool"},
    "factorial": {"params": [("n", "int")], "return_type": "int"},
    "fibonacci": {"params": [("n", "int")], "return_type": "int"},
    "is_prime": {"params": [("n", "int")], "return_type": "bool"},
    "sort_list": {"params": [("items", "List<int>")],
                  "return_type": "List<int>"},
}


class Planner:
    def __init__(self):
        self._plan_counter = 0

    def create_plan(self, reasoning: ReasoningResult, request: str,
                    language: str = "python") -> Plan:
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter}"
        actions = []

        si = reasoning.structured_intent
        if si and si.compound_steps and "class_and_function" in si.compound_steps:
            actions.extend(self._plan_function(request, reasoning, language))
        elif reasoning.intent == "generate_class":
            actions.extend(self._plan_class(request, reasoning, language))
        elif reasoning.intent == "generate_function":
            actions.extend(self._plan_function(request, reasoning, language))
        elif reasoning.intent == "generate_interface":
            actions.extend(self._plan_interface(request, reasoning, language))
        elif reasoning.intent == "generate_enum":
            actions.extend(self._plan_enum(request, reasoning, language))
        elif reasoning.intent == "generate_project":
            actions.extend(self._plan_project(request, reasoning, language))
        elif reasoning.intent == "generate_ui":
            actions.extend(self._plan_ui(request, reasoning, language))
        elif reasoning.intent == "modify_code":
            actions.extend(self._plan_modification(request, reasoning, language))
        elif reasoning.intent == "fix_error":
            actions.extend(self._plan_error_fix(request, reasoning, language))
        elif reasoning.intent == "explain_code":
            actions.extend(self._plan_explain(request, reasoning, language))
        elif reasoning.intent == "analyze_code":
            actions.extend(self._plan_analyze(request, reasoning, language))
        else:
            actions.extend(self._plan_generic(request, reasoning, language))

        return Plan(
            id=plan_id,
            description=f"Plan for: {request}",
            actions=actions,
            language=language,
            reasoning_trace=reasoning,
        )

    def _plan_class(self, request: str, reasoning: ReasoningResult,
                    language: str) -> list[PlanAction]:
        si = reasoning.structured_intent
        if si and si.class_info:
            ci = si.class_info
        else:
            ci = self._extract_class_info(request)
        return [PlanAction(
            action_type="create_class",
            target=ci.name,
            parameters={"class_info": ci.__dict__},
            description=f"Create class '{ci.name}' with properties: {[p[0] for p in ci.properties]}",
        )]

    def _extract_class_info(self, request: str) -> ClassInfo:
        name = "User"
        m = re.search(r'(\w+)\s*클래스', request)
        if m:
            name = m.group(1)
        m = re.search(r'class\s+(\w+)', request, re.IGNORECASE)
        if m:
            name = m.group(1)
        # LEGACY: entity-to-class-name mapping
        # source: legacy, usage: fallback+compatibility, confidence: 0.4
        name_keywords = {
            "사용자": "User", "유저": "User", "product": "Product", "상품": "Product",
            "item": "Item", "아이템": "Item", "order": "Order", "주문": "Order",
            "account": "Account", "계정": "Account", "student": "Student", "학생": "Student",
            "person": "Person", "사람": "Person", "car": "Car", "자동차": "Car",
            "book": "Book", "책": "Book", "message": "Message", "메시지": "Message",
        }
        for kw, cn in name_keywords.items():
            if kw in request.lower() or kw in request:
                name = cn
                break
        properties = []
        constructor_params = []
        prop_patterns = [
            (r'이름', "name", "string"), (r'나이', "age", "int"),
            (r'주소', "address", "string"), (r'전화번호', "phone", "string"),
            (r'이메일', "email", "string"), (r'가격', "price", "float"),
            (r'수량', "quantity", "int"), (r'제목', "title", "string"),
            (r'내용', "content", "string"), (r'설명', "description", "string"),
            (r'아이디', "id", "int"), (r'번호', "number", "int"),
            (r'날짜', "date", "string"), (r'상태', "status", "string"),
            (r'타입', "type", "string"), (r'값', "value", "int"),
        ]
        for pattern, prop_name, prop_type in prop_patterns:
            if pattern in request:
                properties.append((prop_name, prop_type))
                constructor_params.append((prop_name, prop_type))
        if not properties:
            properties = [("name", "string"), ("age", "int")]
            constructor_params = [("name", "string"), ("age", "int")]
        return ClassInfo(name=name, properties=properties, constructor_params=constructor_params)

    def _plan_function(self, request: str, reasoning: ReasoningResult,
                       language: str) -> list[PlanAction]:
        actions = []
        if reasoning.structured_intent:
            si = reasoning.structured_intent
            if si.compound_steps:
                # Handle class_and_function compound
                if "class_and_function" in si.compound_steps:
                    # Create class first
                    class_info = self._extract_class_info(request)
                    actions.append(PlanAction(
                        action_type="create_class",
                        target=class_info.name,
                        parameters={"class_info": class_info.__dict__},
                        description=f"Create class '{class_info.name}'",
                    ))
                    # Then create search function
                    func_info = {
                        "name": f"search_{class_info.name.lower()}",
                        "params": [(f"items", f"List<{class_info.name}>"), ("target", "string")],
                        "return_type": f"{class_info.name}?",
                        "ir_builder": "search_in_list",
                    }
                    actions.append(PlanAction(
                        action_type="create_function",
                        target=func_info["name"],
                        parameters=func_info,
                        description=f"Create search function '{func_info['name']}'",
                    ))
                    return actions
                # Handle other compound algorithms
                for step_algo in si.compound_steps:
                    func_info = self._build_algo_func_info(step_algo, request)
                    actions.append(PlanAction(
                        action_type="create_function",
                        target=func_info["name"],
                        parameters=func_info,
                        description=f"Create function '{func_info['name']}' ({step_algo})",
                    ))
                return actions
            func_info = self._structured_to_func_info(si, request)
        else:
            func_info = self._extract_function_info(request, reasoning)
        actions.append(PlanAction(
            action_type="create_function",
            target=func_info["name"],
            parameters=func_info,
            description=f"Create function '{func_info['name']}'",
        ))
        return actions

    def _build_algo_func_info(self, algo_id: str, request: str) -> dict:
        algo_map = {
            "remove_duplicates": ("remove_duplicates", [("items", "List<int>")], "List<int>"),
            "sort_list": ("sort_list", [("items", "List<int>")], "List<int>"),
            "filter_even": ("filter_even", [("items", "List<int>")], "List<int>"),
            "filter_positive": ("filter_positive", [("items", "List<int>")], "List<int>"),
            "sum_list": ("sum_list", [("items", "List<int>")], "int"),
            "average_list": ("average_list", [("items", "List<int>")], "float"),
            "find_max": ("find_max", [("items", "List<int>")], "int"),
            "find_min": ("find_min", [("items", "List<int>")], "int"),
            "add": ("add", [("a", "int"), ("b", "int")], "int"),
            "multiply": ("multiply", [("a", "int"), ("b", "int")], "int"),
            "concat_strings": ("concat_strings", [("a", "string"), ("b", "string")], "string"),
            "search_in_list": ("search_in_list", [("items", "List<int>"), ("target", "int")], "bool"),
            "search_in_string": ("search_in_string", [("text", "string"), ("pattern", "string")], "bool"),
            "factorial": ("factorial", [("n", "int")], "int"),
            "fibonacci": ("fibonacci", [("n", "int")], "int"),
            "is_prime": ("is_prime", [("n", "int")], "bool"),
        }
        if algo_id in algo_map:
            name, params, ret = algo_map[algo_id]
            return {"name": name, "params": params, "return_type": ret, "ir_builder": algo_id}
        return {"name": algo_id, "params": [("items", "List<int>")], "return_type": "List<int>", "ir_builder": algo_id}

    def _structured_to_func_info(self, si: StructuredIntent, request: str) -> dict:
        algo_builders = {
            "add": self._build_add, "multiply": self._build_multiply,
            "concat_strings": self._build_concat_strings, "find_max": self._build_find_max,
            "find_min": self._build_find_min, "sum_list": self._build_sum_list,
            "average_list": self._build_average_list, "search_in_list": self._build_search_in_list,
            "search_in_string": self._build_search_in_string, "factorial": self._build_factorial,
            "fibonacci": self._build_fibonacci, "is_prime": self._build_is_prime,
            "sort_list": self._build_sort_list,
        }
        builder = algo_builders.get(si.algorithm)
        if builder:
            return builder(si, request)
        return {
            "name": si.algorithm, "params": si.params,
            "return_type": si.return_type, "ir_builder": "generic",
        }

    def _build_add(self, si, request):
        return {"name": self._extract_name(request, "add"), "params": [("a", "int"), ("b", "int")], "return_type": "int", "ir_builder": "add"}

    def _build_multiply(self, si, request):
        return {"name": self._extract_name(request, "multiply"), "params": [("a", "int"), ("b", "int")], "return_type": "int", "ir_builder": "multiply"}

    def _build_concat_strings(self, si, request):
        return {"name": self._extract_name(request, "concat_strings"), "params": [("a", "string"), ("b", "string")], "return_type": "string", "ir_builder": "concat_strings"}

    def _build_find_max(self, si, request):
        return {"name": self._extract_name(request, "find_max"), "params": [("items", "List<int>")], "return_type": "int", "ir_builder": "find_max"}

    def _build_find_min(self, si, request):
        return {"name": self._extract_name(request, "find_min"), "params": [("items", "List<int>")], "return_type": "int", "ir_builder": "find_min"}

    def _build_sum_list(self, si, request):
        return {"name": self._extract_name(request, "sum_list"), "params": [("items", "List<int>")], "return_type": "int", "ir_builder": "sum_list"}

    def _build_average_list(self, si, request):
        return {"name": self._extract_name(request, "average_list"), "params": [("items", "List<int>")], "return_type": "float", "ir_builder": "average_list"}

    def _build_search_in_list(self, si, request):
        return {"name": self._extract_name(request, "search_in_list"), "params": [("items", "List<int>"), ("target", "int")], "return_type": "bool", "ir_builder": "search_in_list"}

    def _build_search_in_string(self, si, request):
        return {"name": self._extract_name(request, "search_in_string"), "params": [("text", "string"), ("pattern", "string")], "return_type": "bool", "ir_builder": "search_in_string"}

    def _build_factorial(self, si, request):
        return {"name": self._extract_name(request, "factorial"), "params": [("n", "int")], "return_type": "int", "ir_builder": "factorial"}

    def _build_fibonacci(self, si, request):
        return {"name": self._extract_name(request, "fibonacci"), "params": [("n", "int")], "return_type": "int", "ir_builder": "fibonacci"}

    def _build_is_prime(self, si, request):
        return {"name": self._extract_name(request, "is_prime"), "params": [("n", "int")], "return_type": "bool", "ir_builder": "is_prime"}

    def _build_sort_list(self, si, request):
        return {"name": self._extract_name(request, "sort_list"), "params": [("items", "List<int>")], "return_type": "List<int>", "ir_builder": "sort_list"}

    def _extract_name(self, request: str, default: str) -> str:
        patterns = [
            (r'함수\s*[\'"]?([a-zA-Z_]\w*)', 1),
            (r'function\s+([a-zA-Z_]\w*)', 1),
            (r'([a-zA-Z_]\w*)\s*함수', 1),
            (r'([a-zA-Z_]\w*)\s*function', 1),
            (r'func\s+([a-zA-Z_]\w*)', 1),
        ]
        for pat, grp in patterns:
            m = re.search(pat, request, re.IGNORECASE)
            if m:
                return m.group(grp)
        return default

    def _extract_function_info(self, request: str, reasoning: ReasoningResult) -> dict:
        name = self._extract_name(request, "unknown_func")
        params = []
        return_type = "void"
        body_desc = ""
        request_lower = request.lower()
        if "더하는" in request or "더하기" in request or "add" in request_lower:
            name = name if name != "unknown_func" else "add"
            params = [("a", "int"), ("b", "int")]
            return_type = "int"
            body_desc = "return a + b"
        elif "곱하는" in request or "곱하기" in request or "multiply" in request_lower:
            name = name if name != "unknown_func" else "multiply"
            params = [("a", "int"), ("b", "int")]
            return_type = "int"
            body_desc = "return a * b"
        elif "팩토리얼" in request or "factorial" in request_lower:
            name = name if name != "unknown_func" else "factorial"
            params = [("n", "int")]
            return_type = "int"
            body_desc = "factorial"
        elif "문자열" in request and ("결합" in request or "합치" in request):
            name = name if name != "unknown_func" else "concat_strings"
            params = [("a", "string"), ("b", "string")]
            return_type = "string"
            body_desc = "return a + b"
        return {"name": name, "params": params, "return_type": return_type, "body_description": body_desc, "ir_builder": "generic"}

    def _plan_interface(self, request, reasoning, language):
        name = "MyInterface"
        m = re.search(r'class\s+(\w+)', request, re.IGNORECASE)
        if m:
            name = m.group(1)
        return [PlanAction(action_type="create_interface", target=name,
                           parameters={"name": name}, description=f"Create interface '{name}'")]

    def _plan_enum(self, request, reasoning, language):
        name = "MyEnum"
        m = re.search(r'class\s+(\w+)', request, re.IGNORECASE)
        if m:
            name = m.group(1)
        return [PlanAction(action_type="create_enum", target=name,
                           parameters={"name": name}, description=f"Create enum '{name}'")]

    def _plan_project(self, request, reasoning, language):
        name = "MyProject"
        return [PlanAction(action_type="create_project", target=name,
                           parameters={"name": name}, description=f"Create project '{name}'")]

    def _plan_ui(self, request, reasoning, language):
        ui_info = self._extract_ui_info(request)
        return [PlanAction(action_type="create_ui_screen", target=ui_info["screen_name"],
                           parameters=ui_info, description=f"Create UI screen '{ui_info['screen_name']}'")]

    def _plan_modification(self, request, reasoning, language):
        return [PlanAction(action_type="modify_code", target="existing_code",
                           parameters={"request": request}, description="Modify code")]

    def _plan_error_fix(self, request, reasoning, language):
        return [PlanAction(action_type="fix_error", target="error_site",
                           parameters={"request": request}, description="Fix error")]

    def _plan_explain(self, request, reasoning, language):
        return [PlanAction(action_type="explain", target="code_or_concept",
                           parameters={"request": request}, description="Explain")]

    def _plan_analyze(self, request, reasoning, language):
        return [PlanAction(action_type="analyze", target="code",
                           parameters={"request": request}, description="Analyze")]

    def _plan_generic(self, request, reasoning, language):
        return [PlanAction(action_type="process", target="request",
                           parameters={"request": request}, description="Process")]

    def _extract_ui_info(self, request: str) -> dict:
        request_lower = request.lower()
        screen_name = "MainScreen"
        components = []
        m = re.search(r'[\'"]?(\w+)[\'"]?\s*화면', request)
        if m:
            screen_name = m.group(1)
        if "버튼" in request or "button" in request_lower:
            components.append({"type": "BUTTON", "name": "actionButton", "text": "Click Me"})
        if "텍스트" in request or "text" in request_lower:
            components.append({"type": "TEXT", "name": "label", "text": "Hello"})
        if "입력" in request or "input" in request_lower:
            components.append({"type": "INPUT", "name": "inputField", "placeholder": "Enter text"})
        if "리스트" in request or "list" in request_lower:
            components.append({"type": "LIST", "name": "itemList"})
        if not components:
            components = [{"type": "TEXT", "name": "title", "text": screen_name},
                          {"type": "BUTTON", "name": "submit", "text": "Submit"}]
        return {"screen_name": screen_name, "components": components}

    def execute_plan(self, plan: Plan) -> list[IRNode]:
        plan.status = "in_progress"
        results = []
        for action in plan.actions:
            if action.action_type == "create_function":
                results.append(self._execute_create_function(action))
            elif action.action_type == "create_class":
                results.append(self._execute_create_class(action))
            elif action.action_type == "create_ui_screen":
                results.append(self._execute_create_ui_screen(action))
        plan.status = "completed"
        return results

    def _execute_create_class(self, action: PlanAction) -> IRNode:
        ci_dict = action.parameters.get("class_info", {})
        class_name = ci_dict.get("name", action.target)
        properties = ci_dict.get("properties", [])
        constructor_params = ci_dict.get("constructor_params", properties)
        cls = create_class_ir(class_name)
        for pname, ptype in properties:
            prop_node = IRNode(node_type=IRNodeType.PROPERTY, name=pname, type_ref=ptype)
            cls.add_child(prop_node)
        ctor = IRNode(node_type=IRNodeType.CONSTRUCTOR)
        ctor.properties["class_name"] = class_name
        for pname, ptype in constructor_params:
            ctor.add_child(IRNode(node_type=IRNodeType.PARAMETER, name=pname, type_ref=ptype))
        ctor_block = IRNode(node_type=IRNodeType.BLOCK)
        for pname, ptype in constructor_params:
            ctor_block.add_child(IRNode(
                node_type=IRNodeType.ASSIGNMENT,
                name=f"self.{pname}",
                children=[IRNode(node_type=IRNodeType.REFERENCE, name=pname)],
            ))
        ctor.add_child(ctor_block)
        cls.add_child(ctor)
        return cls

    def _ir_builder_registry(self) -> dict:
        """The algorithm's semantic spec → IR builder (the structured
        knowledge of what each algorithm IS — the IR is the source of
        truth, never a code template)."""
        return {
            "add": self._ir_add, "multiply": self._ir_multiply,
            "concat_strings": self._ir_concat_strings, "find_max": self._ir_find_max,
            "find_min": self._ir_find_min, "sum_list": self._ir_sum_list,
            "average_list": self._ir_average_list, "search_in_list": self._ir_search_in_list,
            "search_in_string": self._ir_search_in_string, "factorial": self._ir_factorial,
            "fibonacci": self._ir_fibonacci, "is_prime": self._ir_is_prime,
            "sort_list": self._ir_sort_list,
            "remove_duplicates": self._ir_remove_duplicates,
            "filter_even": self._ir_filter_even,
            "filter_positive": self._ir_filter_positive,
        }

    def build_algorithm_ir(self, algorithm: str, name: str = None):
        """Structured IR for a known algorithm.  Returns None when the
        registry has no builder for it (the caller falls back)."""
        builder = self._ir_builder_registry().get(algorithm)
        if not builder:
            return None
        return builder({"name": name or algorithm})

    def _execute_create_function(self, action: PlanAction) -> IRNode:
        builder_name = action.parameters.get("ir_builder", "generic")
        builder = self._ir_builder_registry().get(builder_name)
        if builder:
            return builder(action.parameters)
        return self._ir_generic(action.parameters)

    def _ir_add(self, params):
        return create_function_ir(params["name"], [("a", "int"), ("b", "int")], "int",
            [create_return_ir(create_binary_op_ir("+", create_reference_ir("a"), create_reference_ir("b")))])

    def _ir_multiply(self, params):
        return create_function_ir(params["name"], [("a", "int"), ("b", "int")], "int",
            [create_return_ir(create_binary_op_ir("*", create_reference_ir("a"), create_reference_ir("b")))])

    def _ir_concat_strings(self, params):
        return create_function_ir(params["name"], [("a", "string"), ("b", "string")], "string",
            [create_return_ir(create_binary_op_ir("+", create_reference_ir("a"), create_reference_ir("b")))])

    def _ir_find_max(self, params):
        fname = params["name"]
        body = [
            create_variable_ir("max_val", "int",
                create_call_ir("get_element", [create_reference_ir("items"), create_literal_ir("0", "int")])),
            create_for_each_ir("item", "items", [
                create_if_ir(
                    create_binary_op_ir(">", create_reference_ir("item"), create_reference_ir("max_val")),
                    [create_assignment_ir("max_val", create_reference_ir("item"))]),
            ]),
            create_return_ir(create_reference_ir("max_val")),
        ]
        return create_function_ir(fname, [("items", "List<int>")], "int", body)

    def _ir_find_min(self, params):
        fname = params["name"]
        body = [
            create_variable_ir("min_val", "int",
                create_call_ir("get_element", [create_reference_ir("items"), create_literal_ir("0", "int")])),
            create_for_each_ir("item", "items", [
                create_if_ir(
                    create_binary_op_ir("<", create_reference_ir("item"), create_reference_ir("min_val")),
                    [create_assignment_ir("min_val", create_reference_ir("item"))]),
            ]),
            create_return_ir(create_reference_ir("min_val")),
        ]
        return create_function_ir(fname, [("items", "List<int>")], "int", body)

    def _ir_sum_list(self, params):
        fname = params["name"]
        body = [
            create_variable_ir("total", "int", create_literal_ir("0", "int")),
            create_for_each_ir("item", "items", [
                create_assignment_ir("total",
                    create_binary_op_ir("+", create_reference_ir("total"), create_reference_ir("item"))),
            ]),
            create_return_ir(create_reference_ir("total")),
        ]
        return create_function_ir(fname, [("items", "List<int>")], "int", body)

    def _ir_average_list(self, params):
        fname = params["name"]
        body = [
            create_variable_ir("total", "int", create_literal_ir("0", "int")),
            create_for_each_ir("item", "items", [
                create_assignment_ir("total",
                    create_binary_op_ir("+", create_reference_ir("total"), create_reference_ir("item"))),
            ]),
            create_return_ir(create_binary_op_ir("/",
                create_reference_ir("total"),
                create_length_ir(create_reference_ir("items")))),
        ]
        return create_function_ir(fname, [("items", "List<int>")], "float", body)

    def _ir_search_in_list(self, params):
        fname = params["name"]
        body = [
            create_for_each_ir("item", "items", [
                create_if_ir(
                    create_binary_op_ir("==", create_reference_ir("item"), create_reference_ir("target")),
                    [create_return_ir(create_literal_ir("true", "bool"))]),
            ]),
            create_return_ir(create_literal_ir("false", "bool")),
        ]
        return create_function_ir(fname, [("items", "List<int>"), ("target", "int")], "bool", body)

    def _ir_search_in_string(self, params):
        fname = params["name"]
        body = [
            create_return_ir(create_call_ir("contains", [
                create_reference_ir("text"), create_reference_ir("pattern")])),
        ]
        return create_function_ir(fname, [("text", "string"), ("pattern", "string")], "bool", body)

    def _ir_factorial(self, params):
        fname = params["name"]
        body = [
            create_if_ir(
                create_binary_op_ir("<=", create_reference_ir("n"), create_literal_ir("1", "int")),
                [create_return_ir(create_literal_ir("1", "int"))],
                [create_return_ir(create_binary_op_ir("*",
                    create_reference_ir("n"),
                    create_call_ir(fname, [
                        create_binary_op_ir("-", create_reference_ir("n"), create_literal_ir("1", "int"))])))],
            ),
        ]
        return create_function_ir(fname, [("n", "int")], "int", body)

    def _ir_fibonacci(self, params):
        fname = params["name"]
        body = [
            create_if_ir(
                create_binary_op_ir("<=", create_reference_ir("n"), create_literal_ir("0", "int")),
                [create_return_ir(create_literal_ir("0", "int"))]),
            create_if_ir(
                create_binary_op_ir("==", create_reference_ir("n"), create_literal_ir("1", "int")),
                [create_return_ir(create_literal_ir("1", "int"))]),
            create_return_ir(create_binary_op_ir("+",
                create_call_ir(fname, [create_binary_op_ir("-", create_reference_ir("n"), create_literal_ir("1", "int"))]),
                create_call_ir(fname, [create_binary_op_ir("-", create_reference_ir("n"), create_literal_ir("2", "int"))]))),
        ]
        return create_function_ir(fname, [("n", "int")], "int", body)

    def _ir_is_prime(self, params):
        fname = params["name"]
        body = [
            create_if_ir(
                create_binary_op_ir("<=", create_reference_ir("n"), create_literal_ir("1", "int")),
                [create_return_ir(create_literal_ir("false", "bool"))]),
            create_for_ir(
                create_variable_ir("i", "int", create_literal_ir("2", "int")),
                create_binary_op_ir("<=", create_reference_ir("i"),
                    create_call_ir("sqrt", [create_reference_ir("n")])),
                create_assignment_ir("i", create_binary_op_ir("+", create_reference_ir("i"), create_literal_ir("1", "int"))),
                [
                    create_if_ir(
                        create_binary_op_ir("==",
                            create_binary_op_ir("%", create_reference_ir("n"), create_reference_ir("i")),
                            create_literal_ir("0", "int")),
                        [create_return_ir(create_literal_ir("false", "bool"))]),
                ]),
            create_return_ir(create_literal_ir("true", "bool")),
        ]
        return create_function_ir(fname, [("n", "int")], "bool", body)

    def _ir_sort_list(self, params):
        fname = params["name"]
        body = [
            create_variable_ir("sorted_list", "List<int>",
                create_call_ir("copy_list", [create_reference_ir("items")])),
            create_for_ir(
                create_variable_ir("i", "int", create_literal_ir("0", "int")),
                create_binary_op_ir("<", create_reference_ir("i"),
                    create_binary_op_ir("-", create_length_ir(create_reference_ir("sorted_list")), create_literal_ir("1", "int"))),
                create_assignment_ir("i", create_binary_op_ir("+", create_reference_ir("i"), create_literal_ir("1", "int"))),
                [
                    create_for_ir(
                        create_variable_ir("j", "int",
                            create_binary_op_ir("+", create_reference_ir("i"), create_literal_ir("1", "int"))),
                        create_binary_op_ir("<", create_reference_ir("j"),
                            create_length_ir(create_reference_ir("sorted_list"))),
                        create_assignment_ir("j", create_binary_op_ir("+", create_reference_ir("j"), create_literal_ir("1", "int"))),
                        [
                            create_if_ir(
                                # Ascending: swap when the LATER element
                                # is smaller, so the smaller one moves to
                                # position i (the artifact's own semantics
                                # map sort_list → sort_ascending).
                                create_binary_op_ir("<",
                                    create_index_ir(create_reference_ir("sorted_list"), create_reference_ir("j")),
                                    create_index_ir(create_reference_ir("sorted_list"), create_reference_ir("i"))),
                                [
                                    create_variable_ir("temp", "int",
                                        create_index_ir(create_reference_ir("sorted_list"), create_reference_ir("j"))),
                                    create_assignment_ir("sorted_list[j]",
                                        create_index_ir(create_reference_ir("sorted_list"), create_reference_ir("i"))),
                                    create_assignment_ir("sorted_list[i]", create_reference_ir("temp")),
                                ]),
                        ]),
                ]),
            create_return_ir(create_reference_ir("sorted_list")),
        ]
        return create_function_ir(fname, [("items", "List<int>")], "List<int>", body)

    def _ir_remove_duplicates(self, params):
        fname = params["name"]
        body = [
            create_variable_ir("result", "List<int>",
                create_call_ir("new_list", [])),
            create_for_each_ir("item", "items", [
                create_if_ir(
                    create_binary_op_ir("==",
                        create_call_ir("contains", [
                            create_reference_ir("result"), create_reference_ir("item")]),
                        create_literal_ir("false", "bool")),
                    [create_call_ir("append", [
                        create_reference_ir("result"), create_reference_ir("item")])]),
            ]),
            create_return_ir(create_reference_ir("result")),
        ]
        return create_function_ir(fname, [("items", "List<int>")], "List<int>", body)

    def _ir_filter_even(self, params):
        fname = params["name"]
        body = [
            create_variable_ir("result", "List<int>",
                create_call_ir("new_list", [])),
            create_for_each_ir("item", "items", [
                create_if_ir(
                    create_binary_op_ir("==",
                        create_binary_op_ir("%", create_reference_ir("item"), create_literal_ir("2", "int")),
                        create_literal_ir("0", "int")),
                    [create_call_ir("append", [
                        create_reference_ir("result"), create_reference_ir("item")])]),
            ]),
            create_return_ir(create_reference_ir("result")),
        ]
        return create_function_ir(fname, [("items", "List<int>")], "List<int>", body)

    def _ir_filter_positive(self, params):
        fname = params["name"]
        body = [
            create_variable_ir("result", "List<int>",
                create_call_ir("new_list", [])),
            create_for_each_ir("item", "items", [
                create_if_ir(
                    create_binary_op_ir(">", create_reference_ir("item"), create_literal_ir("0", "int")),
                    [create_call_ir("append", [
                        create_reference_ir("result"), create_reference_ir("item")])]),
            ]),
            create_return_ir(create_reference_ir("result")),
        ]
        return create_function_ir(fname, [("items", "List<int>")], "List<int>", body)

    def _ir_generic(self, params):
        name = params.get("name", "unknown_func")
        func_params = params.get("params", [])
        return_type = params.get("return_type", "void")
        body_desc = params.get("body_description", "")
        body_stmts = []
        if "return a + b" in body_desc:
            body_stmts.append(create_return_ir(
                create_binary_op_ir("+", create_reference_ir("a"), create_reference_ir("b"))))
        elif "return a * b" in body_desc:
            body_stmts.append(create_return_ir(
                create_binary_op_ir("*", create_reference_ir("a"), create_reference_ir("b"))))
        elif body_desc == "factorial":
            body_stmts.append(create_if_ir(
                create_binary_op_ir("<=", create_reference_ir("n"), create_literal_ir("1", "int")),
                [create_return_ir(create_literal_ir("1", "int"))],
                [create_return_ir(create_binary_op_ir("*", create_reference_ir("n"),
                    create_call_ir("factorial", [
                        create_binary_op_ir("-", create_reference_ir("n"), create_literal_ir("1", "int"))])))]))
        else:
            body_stmts.append(create_return_ir(create_literal_ir("None")))
        return create_function_ir(name, func_params, return_type, body_stmts)

    def _execute_create_ui_screen(self, action):
        screen = IRNode(node_type=IRNodeType.UI_SCREEN, name=action.parameters.get("screen_name", "Screen"))
        for comp in action.parameters.get("components", []):
            ui_comp = IRNode(node_type=IRNodeType.UI_COMPONENT, name=comp.get("name", "component"),
                             properties={"component_type": comp.get("type", "TEXT")})
            for key in ("text", "placeholder", "source"):
                if key in comp:
                    ui_comp.add_child(IRNode(node_type=IRNodeType.UI_PROPERTY, name=key, value=comp[key]))
            screen.add_child(ui_comp)
        return screen

    def plan_to_dict(self, plan: Plan) -> dict:
        return {
            "id": plan.id,
            "description": plan.description,
            "language": plan.language,
            "status": plan.status,
            "actions": [{"action_type": a.action_type, "target": a.target,
                         "parameters": a.parameters, "description": a.description}
                        for a in plan.actions],
        }
