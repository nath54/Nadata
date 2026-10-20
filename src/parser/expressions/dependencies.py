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
    IndexExpr,
    LiteralExpr,
    MemberAccessExpr,
    UnaryOpExpr,
)
from src.core.expressions.statement import (
    AssignStmt,
    BlockStmt,
    ExprStmt,
    IfStmt,
    ReturnStmt,
    Statement,
    VarDeclStmt,
)
from src.parser.utils.identifiers import IdentifierExtractor


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

    # Indexing: recurse into the target and the index expression
    if isinstance(node, IndexExpr):
        _collect_from(node=node.target, dependencies=dependencies)
        _collect_from(node=node.index, dependencies=dependencies)
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


def collect_statement_dependencies(statements: list[Statement]) -> set[str]:
    """
    Collects the free identifiers referenced by a statement list.

    Parameter and local variable names are NOT excluded here; callers should
    subtract the locally bound names to obtain the true external dependencies.

    Args:
        statements (list[Statement]): Statements to analyze.

    Returns:
        set[str]: Union of the referenced identifiers, including local
            variable type annotation names.
    """

    dependencies: set[str] = set()

    for statement in statements:
        _collect_statement_from(statement=statement, dependencies=dependencies)

    return dependencies


def collect_declared_names(statements: list[Statement]) -> set[str]:
    """
    Collects the names of all variables declared in a statement list.

    Args:
        statements (list[Statement]): Statements to analyze (recursed into
            nested blocks).

    Returns:
        set[str]: Declared variable names.
    """

    names: set[str] = set()

    for statement in statements:

        # Variable declarations bind a name
        if isinstance(statement, VarDeclStmt):
            names.add(statement.name)

        # If statements may declare in both branches
        elif isinstance(statement, IfStmt):
            names |= collect_declared_names(
                statements=statement.then_block.statements
            )

            if statement.else_block is not None:
                names |= collect_declared_names(
                    statements=statement.else_block.statements
                )

        # Nested blocks bind their own names
        elif isinstance(statement, BlockStmt):
            names |= collect_declared_names(statements=statement.statements)

    return names


def _collect_statement_from(
    statement: Statement,
    dependencies: set[str]
) -> None:
    """
    Recursively collects dependencies from a single statement node.

    Args:
        statement (Statement): Current statement node to process.
        dependencies (set[str]): Dependency set to update in place.
    """

    # Variable declarations: the initializer plus the raw type name
    if isinstance(statement, VarDeclStmt):
        _collect_from(node=statement.init_expr, dependencies=dependencies)

        if statement.var_type_name is not None:
            dependencies |= IdentifierExtractor.extract_identifiers(
                text=statement.var_type_name
            )

        return

    # Assignments: the value expression only
    if isinstance(statement, AssignStmt):
        _collect_from(node=statement.value_expr, dependencies=dependencies)
        return

    # Expression statements: the expression only
    if isinstance(statement, ExprStmt):
        _collect_from(node=statement.expression, dependencies=dependencies)
        return

    # Returns: the optional value expression
    if isinstance(statement, ReturnStmt):
        if statement.value_expr is not None:
            _collect_from(node=statement.value_expr, dependencies=dependencies)

        return

    # If statements: the condition plus both branches
    if isinstance(statement, IfStmt):
        _collect_from(node=statement.condition, dependencies=dependencies)

        for inner in statement.then_block.statements:
            _collect_statement_from(statement=inner, dependencies=dependencies)

        if statement.else_block is not None:
            for inner in statement.else_block.statements:
                _collect_statement_from(statement=inner, dependencies=dependencies)

        return

    # Nested blocks
    if isinstance(statement, BlockStmt):
        for inner in statement.statements:
            _collect_statement_from(statement=inner, dependencies=dependencies)

        return
