"""
AST based dependency collection for Nadata expressions.

Provides the collect_dependencies helper walking a parsed Expression AST and
returning the free identifiers it references. Member names accessed after a
dot (e.g. 'now' in 'DateTime.now()') are NOT dependencies: only root
identifiers are collected, unlike the textual IdentifierExtractor.
"""

# Import Modules
from __future__ import annotations

from src.core.expressions.expression import (
    ArrayLiteralExpr,
    BinaryOpExpr,
    CallExpr,
    DictLiteralExpr,
    Expression,
    IdentifierExpr,
    LiteralExpr,
    MemberAccessExpr,
    UnaryOpExpr,
)


def collect_dependencies(expression: Expression) -> set[str]:
    """
    Collects the free identifier names referenced by an expression AST.

    Args:
        expression (Expression): Parsed expression to analyze.

    Returns:
        set[str]: Names of the identifiers the expression depends on.
    """

    dependencies: set[str] = set()

    _collect_from(node=expression, dependencies=dependencies)

    return dependencies


def _collect_from(
    node: Expression,
    dependencies: set[str]
) -> None:
    """
    Recursively collects dependencies from a single AST node into the set.

    Args:
        node (Expression): Current AST node to process.
        dependencies (set[str]): Dependency set to update in place.
    """

    # Literals reference nothing
    if isinstance(node, LiteralExpr):
        return

    # Plain identifiers are dependencies (root of member chains included)
    if isinstance(node, IdentifierExpr):
        dependencies.add(node.name)
        return

    # Member access: recurse into the target only, the member is not free
    if isinstance(node, MemberAccessExpr):
        _collect_from(node=node.target, dependencies=dependencies)
        return

    # Calls: the callee and every argument (positional and named)
    if isinstance(node, CallExpr):
        _collect_from(node=node.callee, dependencies=dependencies)

        for argument in node.args:
            _collect_from(node=argument, dependencies=dependencies)

        for argument in node.kwargs.values():
            _collect_from(node=argument, dependencies=dependencies)

        return

    # Unary operations: the operand only
    if isinstance(node, UnaryOpExpr):
        _collect_from(node=node.operand, dependencies=dependencies)
        return

    # Binary operations: both operands
    if isinstance(node, BinaryOpExpr):
        _collect_from(node=node.left, dependencies=dependencies)
        _collect_from(node=node.right, dependencies=dependencies)
        return

    # Array literals: every element
    if isinstance(node, ArrayLiteralExpr):
        for element in node.elements:
            _collect_from(node=element, dependencies=dependencies)

        return

    # Dict literals: every key and value
    if isinstance(node, DictLiteralExpr):
        for key, value in node.entries:
            _collect_from(node=key, dependencies=dependencies)
            _collect_from(node=value, dependencies=dependencies)

        return
