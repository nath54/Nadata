"""
Statement AST nodes and execution engine for Nadata method bodies.

Provides AST representations for variable declarations, assignments,
conditional if blocks, return statements, and statement blocks managing typed Value objects.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.core.types.types import Type
from src.core.values.values import Value
from src.core.expressions.expression import Expression
from src.core.expressions.scope import (
    ReturnSignal,
    ExecutionScope,
)


class Statement:
    """
    Abstract base class for all AST statement nodes.
    """

    def execute(self, scope: ExecutionScope) -> None:
        """
        Executes the statement within the given scope.

        Args:
            scope (ExecutionScope): Current execution environment.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError("execute must be implemented by subclasses.")


class VarDeclStmt(Statement):
    """
    AST statement declaring a new typed Value variable with an initial value.

    Attributes:
        name (str): Variable identifier.
        init_expr (Expression): Initializer expression returning a Value.
        var_type (Optional[Type]): Expected schema type for the variable.
    """

    def __init__(
        self,
        name: str,
        init_expr: Expression,
        var_type: Optional[Type] = None
    ) -> None:
        """
        Initializes the VarDeclStmt instance.

        Args:
            name (str): Variable identifier.
            init_expr (Expression): Initializer expression.
            var_type (Optional[Type], optional): Optional type annotation. Defaults to None.
        """

        self.name: str = name
        self.init_expr: Expression = init_expr
        self.var_type: Optional[Type] = var_type

    def execute(self, scope: ExecutionScope) -> None:
        """
        Evaluates the initializer and declares the typed Value variable in the scope.

        Args:
            scope (ExecutionScope): Current execution scope.

        Raises:
            TypeError: If the evaluated result is not a Value or violates var_type.
        """

        val: Value = self.init_expr.evaluate(scope=scope)

        # Validate against type annotation if provided
        if self.var_type is not None:
            self.var_type.check_value(value=val.get())

        # Declare variable in the local scope
        scope.define(name=self.name, value=val)

    def __repr__(self) -> str:
        """
        Returns string representation of the VarDeclStmt.

        Returns:
            str: Variable declaration representation.
        """

        if self.var_type is not None:
            return f"{self.name}: {repr(self.var_type)} = {repr(self.init_expr)};"

        return f"{self.name} = {repr(self.init_expr)};"


class AssignStmt(Statement):
    """
    AST statement reassigning a Value to an existing variable.

    Attributes:
        target_name (str): Name of the variable to reassign.
        value_expr (Expression): Expression producing the new Value.
    """

    def __init__(self, target_name: str, value_expr: Expression) -> None:
        """
        Initializes the AssignStmt instance.

        Args:
            target_name (str): Target variable identifier.
            value_expr (Expression): Value expression to assign.
        """

        self.target_name: str = target_name
        self.value_expr: Expression = value_expr

    def execute(self, scope: ExecutionScope) -> None:
        """
        Evaluates the expression and reassigns the target in the scope.

        Args:
            scope (ExecutionScope): Current execution scope.
        """

        val: Value = self.value_expr.evaluate(scope=scope)

        # Assign in the existing scope hierarchy
        scope.assign(name=self.target_name, value=val)

    def __repr__(self) -> str:
        """
        Returns string representation of the AssignStmt.

        Returns:
            str: Assignment statement string.
        """

        return f"{self.target_name} = {repr(self.value_expr)};"


class ReturnStmt(Statement):
    """
    AST statement returning a typed Value from a method.

    Attributes:
        value_expr (Optional[Expression]): Expression producing the return Value.
    """

    def __init__(self, value_expr: Optional[Expression] = None) -> None:
        """
        Initializes the ReturnStmt instance.

        Args:
            value_expr (Optional[Expression], optional): Return value expression. Defaults to None.
        """

        self.value_expr: Optional[Expression] = value_expr

    def execute(self, scope: ExecutionScope) -> None:
        """
        Evaluates the return value and raises a ReturnSignal with the typed Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Raises:
            ReturnSignal: Propagates the return Value.
        """

        ret_val: Optional[Value] = (
            self.value_expr.evaluate(scope=scope)
            if self.value_expr is not None
            else None
        )

        raise ReturnSignal(value=ret_val)

    def __repr__(self) -> str:
        """
        Returns string representation of the ReturnStmt.

        Returns:
            str: Return statement string.
        """

        if self.value_expr is not None:
            return f"return {repr(self.value_expr)};"

        return "return;"


class BlockStmt(Statement):
    """
    AST node representing a sequential list of statements inside braces.

    Attributes:
        statements (list[Statement]): List of statements contained in the block.
    """

    def __init__(self, statements: Optional[list[Statement]] = None) -> None:
        """
        Initializes the BlockStmt instance.

        Args:
            statements (Optional[list[Statement]], optional): Initial statements. Defaults to None.
        """

        self.statements: list[Statement] = statements if statements is not None else []

    def add_statement(self, stmt: Statement) -> None:
        """
        Appends a statement to the block.

        Args:
            stmt (Statement): Statement to add.
        """

        self.statements.append(stmt)

    def execute(self, scope: ExecutionScope) -> None:
        """
        Executes each statement sequentially in the given scope.

        Args:
            scope (ExecutionScope): Current execution scope.
        """

        for stmt in self.statements:
            stmt.execute(scope=scope)

    def execute_as_function(self, scope: ExecutionScope) -> Optional[Value]:
        """
        Executes the block as a complete function body, intercepting ReturnSignal.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Optional[Value]: Value returned by ReturnStmt, or None if flow falls through.
        """

        try:
            self.execute(scope=scope)
        except ReturnSignal as signal:
            return signal.value

        return None

    def __repr__(self) -> str:
        """
        Returns string representation of the BlockStmt.

        Returns:
            str: Block representation with statement count.
        """

        return f"Block({len(self.statements)} stmts)"


class IfStmt(Statement):
    """
    AST statement representing a conditional if-else branch operating on a Value condition.

    Attributes:
        condition (Expression): Boolean condition expression evaluating to a Value.
        then_block (BlockStmt): Block executed when condition is truthy.
        else_block (Optional[BlockStmt]): Optional block executed when condition is falsy.
    """

    def __init__(
        self,
        condition: Expression,
        then_block: BlockStmt,
        else_block: Optional[BlockStmt] = None
    ) -> None:
        """
        Initializes the IfStmt instance.

        Args:
            condition (Expression): Condition expression.
            then_block (BlockStmt): Statements executed if condition is true.
            else_block (Optional[BlockStmt], optional): Optional else branch. Defaults to None.
        """

        self.condition: Expression = condition
        self.then_block: BlockStmt = then_block
        self.else_block: Optional[BlockStmt] = else_block

    def execute(self, scope: ExecutionScope) -> None:
        """
        Evaluates condition and executes the appropriate branch in a child scope.

        Args:
            scope (ExecutionScope): Current execution scope.
        """

        cond_val: Value = self.condition.evaluate(scope=scope)

        # Extract boolean truth value from Value
        raw_bool: bool = bool(cond_val.get())

        # Execute in a new nested child scope
        child_scope: ExecutionScope = scope.create_child_scope()

        if raw_bool:
            self.then_block.execute(scope=child_scope)
        elif self.else_block is not None:
            self.else_block.execute(scope=child_scope)

    def __repr__(self) -> str:
        """
        Returns string representation of the IfStmt.

        Returns:
            str: If statement representation.
        """

        if self.else_block is not None:
            return f"if ({repr(self.condition)}) {{ ... }} else {{ ... }}"

        return f"if ({repr(self.condition)}) {{ ... }}"
