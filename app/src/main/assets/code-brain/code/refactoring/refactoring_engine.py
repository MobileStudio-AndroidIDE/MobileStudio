"""Refactoring engine - performs AST-level code modifications."""
from __future__ import annotations
from typing import Optional
from code.ir.ir_types import IRNode, IRNodeType, NodeTypeAction


class RefactoringEngine:
    def __init__(self):
        self._operations = {
            NodeTypeAction.ADD: self._add_node,
            NodeTypeAction.MODIFY: self._modify_node,
            NodeTypeAction.REMOVE: self._remove_node,
            NodeTypeAction.MOVE: self._move_node,
            NodeTypeAction.REPLACE: self._replace_node,
            NodeTypeAction.SET_PROPERTY: self._set_property,
        }

    def apply(self, root: IRNode, action: NodeTypeAction, target_id: str,
              node: IRNode = None, **kwargs) -> bool:
        op = self._operations.get(action)
        if not op:
            return False
        return op(root, target_id, node, **kwargs)

    def find_and_apply(self, root: IRNode, action: NodeTypeAction,
                       name: str = None, node_type: IRNodeType = None,
                       node: IRNode = None, **kwargs) -> bool:
        targets = self._find_targets(root, name, node_type)
        if not targets:
            return False
        for target in targets:
            self._operations[action](root, target.id, node, **kwargs)
        return True

    def _add_node(self, root: IRNode, target_id: str, node: IRNode, **kwargs) -> bool:
        if not node:
            return False
        target = self._find_by_id(root, target_id)
        if target:
            target.add_child(node)
            return True
        return False

    def _modify_node(self, root: IRNode, target_id: str, node: IRNode, **kwargs) -> bool:
        target = self._find_by_id(root, target_id)
        if not target:
            return False
        if node:
            target.name = node.name
            target.value = node.value
            target.type_ref = node.type_ref
            target.properties = node.properties
            target.children = node.children
        for k, v in kwargs.items():
            if hasattr(target, k):
                setattr(target, k, v)
        return True

    def _remove_node(self, root: IRNode, target_id: str, node: IRNode = None, **kwargs) -> bool:
        return root.remove_child(target_id)

    def _move_node(self, root: IRNode, target_id: str, node: IRNode = None, **kwargs) -> bool:
        new_parent_id = kwargs.get("new_parent_id")
        if not new_parent_id:
            return False
        target = self._find_by_id(root, target_id)
        new_parent = self._find_by_id(root, new_parent_id)
        if not target or not new_parent:
            return False
        if target.parent_id:
            old_parent = self._find_by_id(root, target.parent_id)
            if old_parent:
                old_parent.remove_child(target_id)
        target.parent_id = new_parent_id
        new_parent.add_child(target)
        return True

    def _replace_node(self, root: IRNode, target_id: str, node: IRNode, **kwargs) -> bool:
        if not node:
            return False
        target = self._find_by_id(root, target_id)
        if not target or not target.parent_id:
            return False
        parent = self._find_by_id(root, target.parent_id)
        if parent:
            return parent.replace_child(target_id, node)
        return False

    def _set_property(self, root: IRNode, target_id: str, node: IRNode = None, **kwargs) -> bool:
        target = self._find_by_id(root, target_id)
        if not target:
            return False
        for k, v in kwargs.items():
            target.properties[k] = v
        return True

    def _find_by_id(self, node: IRNode, target_id: str) -> Optional[IRNode]:
        if node.id == target_id:
            return node
        for child in node.children:
            found = self._find_by_id(child, target_id)
            if found:
                return found
        return None

    def _find_targets(self, root: IRNode, name: str = None,
                      node_type: IRNodeType = None) -> list[IRNode]:
        results = []
        if name and root.name == name and (node_type is None or root.node_type == node_type):
            results.append(root)
        for child in root.children:
            results.extend(self._find_targets(child, name, node_type))
        return results

    def list_modifications(self, root: IRNode, node_type: IRNodeType = None) -> list[dict]:
        results = []
        self._collect_nodes(root, node_type, results)
        return results

    def _collect_nodes(self, node: IRNode, node_type: IRNodeType, results: list):
        if node_type is None or node.node_type == node_type:
            results.append({
                "id": node.id,
                "type": node.node_type.value,
                "name": node.name,
            })
        for child in node.children:
            self._collect_nodes(child, node_type, results)
