"""Structural IR - Common intermediate representation for all languages."""
from __future__ import annotations
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class IRNodeType(Enum):
    PROJECT = "project"
    PACKAGE = "package"
    IMPORT = "import"
    CLASS = "class"
    INTERFACE = "interface"
    OBJECT = "object"
    ENUM = "enum"
    PROPERTY = "property"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    FUNCTION = "function"
    CONSTRUCTOR = "constructor"
    IF = "if"
    ELSE = "else"
    WHEN = "when"
    FOR = "for"
    FOR_EACH = "for_each"
    WHILE = "while"
    TRY = "try"
    CATCH = "catch"
    FINALLY = "finally"
    CALL = "call"
    RETURN = "return"
    ASSIGNMENT = "assignment"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    LITERAL = "literal"
    REFERENCE = "reference"
    TYPE = "type"
    BLOCK = "block"
    STRING_TEMPLATE = "string_template"
    LAMBDA = "lambda"
    ANNOTATION = "annotation"
    COMMENT = "comment"
    UI_SCREEN = "ui_screen"
    UI_COMPONENT = "ui_component"
    UI_PROPERTY = "ui_property"
    UI_ACTION = "ui_action"
    TUPLE = "tuple"
    ARRAY = "array"
    MAP = "map"
    NULL = "null"
    THIS = "this"
    SUPER = "super"
    SELF = "self"
    INDEX = "index"
    LENGTH = "length"
    MEMBER_ACCESS = "member_access"
    INIT = "init"


class NodeTypeAction(Enum):
    ADD = "add"
    MODIFY = "modify"
    REMOVE = "remove"
    MOVE = "move"
    REPLACE = "replace"
    SET_PROPERTY = "set_property"


@dataclass
class SourceLocation:
    file: str = ""
    line: int = 0
    column: int = 0
    end_line: int = 0
    end_column: int = 0


@dataclass
class IRNode:
    node_type: IRNodeType
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    value: Optional[str] = None
    type_ref: Optional[str] = None
    children: list[IRNode] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    location: SourceLocation = field(default_factory=SourceLocation)
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def add_child(self, child: IRNode) -> IRNode:
        child.parent_id = self.id
        self.children.append(child)
        return child

    def find_child_by_name(self, name: str) -> Optional[IRNode]:
        for c in self.children:
            if c.name == name:
                return c
            found = c.find_child_by_name(name)
            if found:
                return found
        return None

    def find_children_by_type(self, node_type: IRNodeType) -> list[IRNode]:
        result = []
        if self.node_type == node_type:
            result.append(self)
        for c in self.children:
            result.extend(c.find_children_by_type(node_type))
        return result

    def remove_child(self, child_id: str) -> bool:
        for i, c in enumerate(self.children):
            if c.id == child_id:
                self.children.pop(i)
                return True
            if c.remove_child(child_id):
                return True
        return False

    def replace_child(self, child_id: str, new_node: IRNode) -> bool:
        for i, c in enumerate(self.children):
            if c.id == child_id:
                new_node.parent_id = self.id
                self.children[i] = new_node
                return True
            if c.replace_child(child_id, new_node):
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type.value,
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "type_ref": self.type_ref,
            "children": [c.to_dict() for c in self.children],
            "properties": self.properties,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IRNode:
        node = cls(
            node_type=IRNodeType(d["node_type"]),
            id=d.get("id", str(uuid.uuid4())[:8]),
            name=d.get("name", ""),
            value=d.get("value"),
            type_ref=d.get("type_ref"),
            properties=d.get("properties", {}),
            parent_id=d.get("parent_id"),
            metadata=d.get("metadata", {}),
        )
        for cd in d.get("children", []):
            child = cls.from_dict(cd)
            child.parent_id = node.id
            node.children.append(child)
        return node

    def clone(self) -> IRNode:
        return IRNode.from_dict(self.to_dict())

    def pretty_print(self, indent: int = 0) -> str:
        prefix = "  " * indent
        parts = [f"{prefix}{self.node_type.value}"]
        if self.name:
            parts.append(f" name={self.name}")
        if self.value:
            parts.append(f" value={self.value}")
        if self.type_ref:
            parts.append(f" type={self.type_ref}")
        if self.properties:
            parts.append(f" props={self.properties}")
        line = "".join(parts)
        for child in self.children:
            line += "\n" + child.pretty_print(indent + 1)
        return line


def create_function_ir(name: str, params: list[tuple[str, str]], return_type: str, body: list[IRNode]) -> IRNode:
    func = IRNode(node_type=IRNodeType.FUNCTION, name=name, type_ref=return_type)
    for pname, ptype in params:
        func.add_child(IRNode(node_type=IRNodeType.PARAMETER, name=pname, type_ref=ptype))
    block = IRNode(node_type=IRNodeType.BLOCK)
    for stmt in body:
        block.add_child(stmt)
    func.add_child(block)
    return func


def create_class_ir(name: str, parent_type: str = None, interfaces: list[str] = None) -> IRNode:
    cls = IRNode(node_type=IRNodeType.CLASS, name=name)
    if parent_type:
        cls.properties["extends"] = parent_type
    if interfaces:
        cls.properties["implements"] = ",".join(interfaces)
    return cls


def create_variable_ir(name: str, type_ref: str = None, init_value: IRNode = None) -> IRNode:
    var = IRNode(node_type=IRNodeType.VARIABLE, name=name, type_ref=type_ref)
    if init_value:
        var.add_child(init_value)
    return var


def create_return_ir(value: IRNode = None) -> IRNode:
    ret = IRNode(node_type=IRNodeType.RETURN)
    if value:
        ret.add_child(value)
    return ret


def create_call_ir(func_name: str, args: list[IRNode] = None) -> IRNode:
    call = IRNode(node_type=IRNodeType.CALL, name=func_name)
    if args:
        for a in args:
            call.add_child(a)
    return call


def create_literal_ir(value: str, type_ref: str = "string") -> IRNode:
    return IRNode(node_type=IRNodeType.LITERAL, value=value, type_ref=type_ref)


def create_reference_ir(name: str) -> IRNode:
    return IRNode(node_type=IRNodeType.REFERENCE, name=name)


def create_binary_op_ir(op: str, left: IRNode, right: IRNode) -> IRNode:
    binop = IRNode(node_type=IRNodeType.BINARY_OP, value=op)
    binop.add_child(left)
    binop.add_child(right)
    return binop


def create_if_ir(condition: IRNode, then_body: list[IRNode], else_body: list[IRNode] = None) -> IRNode:
    if_node = IRNode(node_type=IRNodeType.IF)
    if_node.add_child(condition)
    then_block = IRNode(node_type=IRNodeType.BLOCK)
    for s in then_body:
        then_block.add_child(s)
    if_node.add_child(then_block)
    if else_body:
        else_block = IRNode(node_type=IRNodeType.BLOCK)
        for s in else_body:
            else_block.add_child(s)
        if_node.add_child(else_block)
    return if_node


def create_for_each_ir(iter_var: str, collection: str, body: list[IRNode]) -> IRNode:
    for_node = IRNode(node_type=IRNodeType.FOR_EACH, name=iter_var, value=collection)
    block = IRNode(node_type=IRNodeType.BLOCK)
    for s in body:
        block.add_child(s)
    for_node.add_child(block)
    return for_node


def create_for_ir(init: IRNode, condition: IRNode, update: IRNode, body: list[IRNode]) -> IRNode:
    for_node = IRNode(node_type=IRNodeType.FOR)
    for_node.add_child(init)
    for_node.add_child(condition)
    for_node.add_child(update)
    block = IRNode(node_type=IRNodeType.BLOCK)
    for s in body:
        block.add_child(s)
    for_node.add_child(block)
    return for_node


def create_index_ir(collection: IRNode, index: IRNode) -> IRNode:
    idx = IRNode(node_type=IRNodeType.INDEX)
    idx.add_child(collection)
    idx.add_child(index)
    return idx


def create_length_ir(collection: IRNode) -> IRNode:
    ln = IRNode(node_type=IRNodeType.LENGTH)
    ln.add_child(collection)
    return ln


def create_assignment_ir(target: str, value: IRNode) -> IRNode:
    a = IRNode(node_type=IRNodeType.ASSIGNMENT, name=target)
    a.add_child(value)
    return a
