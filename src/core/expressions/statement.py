"""
Statement AST nodes and execution engine for Nadata method bodies.

Provides AST representations for variable declarations, assignments,
conditional if blocks, return statements, and statement blocks managing typed Value objects.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.types.types import Type
from src.core.values.values import Value
from src.core.expressions.expression import Expression, _wrap_indexed_element
from src.core.expressions.scope import (
    BreakSignal,
    ContinueSignal,
    LoopLimitError,
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
        var_type_name (Optional[str]): Raw type annotation name, awaiting
            resolution into var_type by the resolution phase.
    """

    def __init__(
        self,
        name: str,
        init_expr: Expression,
        var_type: Optional[Type] = None,
        var_type_name: Optional[str] = None
    ) -> None:
        """
        Initializes the VarDeclStmt instance.

        Args:
            name (str): Variable identifier.
            init_expr (Expression): Initializer expression.
            var_type (Optional[Type], optional): Optional type annotation. Defaults to None.
            var_type_name (Optional[str], optional): Raw type annotation name. Defaults to None.
        """

        self.name: str = name
        self.init_expr: Expression = init_expr
        self.var_type: Optional[Type] = var_type
        self.var_type_name: Optional[str] = var_type_name

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

        if self.var_type_name is not None:
            return f"{self.name}: {self.var_type_name} = {repr(self.init_expr)};"

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


class ExprStmt(Statement):
    """
    AST statement evaluating an expression for its side effects.

    Attributes:
        expression (Expression): Expression to evaluate (e.g. a method call).
    """

    def __init__(self, expression: Expression) -> None:
        """
        Initializes the ExprStmt instance.

        Args:
            expression (Expression): Expression to evaluate.
        """

        self.expression: Expression = expression

    def execute(self, scope: ExecutionScope) -> None:
        """
        Evaluates the expression and discards the resulting Value.

        Args:
            scope (ExecutionScope): Current execution scope.
        """

        self.expression.evaluate(scope=scope)

    def __repr__(self) -> str:
        """
        Returns string representation of the ExprStmt.

        Returns:
            str: Expression statement string.
        """

        return f"{repr(self.expression)};"


def _check_fuel(scope: ExecutionScope, iteration: int, loop_repr: str) -> None:
    """
    Validates that a loop iteration is still within the fuel budget.

    Args:
        scope (ExecutionScope): Scope the loop executes in.
        iteration (int): Zero-based index of the iteration about to run.
        loop_repr (str): Representation of the loop for the error message.

    Raises:
        LoopLimitError: If the shared fuel limit is exhausted.
    """

    if iteration >= scope.fuel.limit:
        raise LoopLimitError(
            f"Loop '{loop_repr}' exceeded the fuel limit of {scope.fuel.limit} "
            f"iterations"
        )


class ForInStmt(Statement):
    """
    AST statement iterating a typed loop variable over an array's elements.

    Attributes:
        var_name (str): Loop variable name (scoped to the loop body).
        var_type (Optional[Type]): Resolved declared type of the loop variable.
        var_type_name (Optional[str]): Raw type annotation, awaiting resolution.
        iterable (Expression): Expression evaluating to the iterated array.
        body (BlockStmt): Block executed once per element.
    """

    def __init__(
        self,
        var_name: str,
        var_type: Optional[Type],
        var_type_name: Optional[str],
        iterable: Expression,
        body: BlockStmt
    ) -> None:
        """
        Initializes the ForInStmt instance.

        Args:
            var_name (str): Loop variable name.
            var_type (Optional[Type]): Resolved loop variable type.
            var_type_name (Optional[str]): Raw type annotation name.
            iterable (Expression): Iterable expression.
            body (BlockStmt): Loop body block.
        """

        self.var_name: str = var_name
        self.var_type: Optional[Type] = var_type
        self.var_type_name: Optional[str] = var_type_name
        self.iterable: Expression = iterable
        self.body: BlockStmt = body

    def execute(self, scope: ExecutionScope) -> None:
        """
        Iterates over the array elements, checking each against the declared
        loop variable type.

        Args:
            scope (ExecutionScope): Current execution scope.

        Raises:
            RuntimeError: If the loop variable type was not resolved.
            TypeError: If the iterable is not an array or an element violates
                the declared loop variable type.
            LoopLimitError: If the fuel limit is exhausted.
        """

        if self.var_type is None:
            raise RuntimeError(
                f"Loop variable '{self.var_name}' has no resolved type"
            )

        iterable_value: Value = self.iterable.evaluate(scope=scope)
        raw_iterable: Any = iterable_value.get()

        # Query-like containers (e.g. QuerySet) expose their elements
        if hasattr(raw_iterable, "to_list"):
            raw_iterable = raw_iterable.to_list()

        # Sets iterate their (unordered) elements
        if isinstance(raw_iterable, (set, frozenset)):
            raw_iterable = list(raw_iterable)

        if not isinstance(raw_iterable, list):
            raise TypeError(
                f"Cannot iterate over '{repr(iterable_value)}': "
                f"only arrays are iterable (use .keys() / .values() for dicts)"
            )

        iteration: int = 0

        for element in raw_iterable:
            _check_fuel(scope=scope, iteration=iteration, loop_repr=repr(self))

            # Strict per-element check against the declared type
            self.var_type.check_value(value=element)

            child_scope: ExecutionScope = scope.create_child_scope()
            child_scope.define(
                name=self.var_name,
                value=_wrap_indexed_element(raw=element, element_type=self.var_type),
            )

            try:
                self.body.execute(scope=child_scope)
            except BreakSignal:
                break
            except ContinueSignal:
                pass

            iteration += 1

    def __repr__(self) -> str:
        """
        Returns string representation of the ForInStmt.

        Returns:
            str: For-in loop representation.
        """

        type_name: str = (
            self.var_type_name
            if self.var_type_name is not None
            else repr(self.var_type)
        )

        return f"for ({self.var_name}: {type_name} in {repr(self.iterable)}) {{ ... }}"


class ForStmt(Statement):
    """
    AST statement executing a C-style loop with typed init declarations, a
    condition, and update statements.

    Attributes:
        init_statements (list[Statement]): Typed declarations run once in
            the loop scope (all with initializers).
        condition (Expression): Boolean condition evaluated before each
            iteration.
        update_statements (list[Statement]): Statements run after each
            iteration (assignments, compound assignments, or increments).
        body (BlockStmt): Block executed once per iteration.
    """

    def __init__(
        self,
        init_statements: list[Statement],
        condition: Expression,
        update_statements: list[Statement],
        body: BlockStmt
    ) -> None:
        """
        Initializes the ForStmt instance.

        Args:
            init_statements (list[Statement]): Init declarations.
            condition (Expression): Loop condition.
            update_statements (list[Statement]): Update statements.
            body (BlockStmt): Loop body block.
        """

        self.init_statements: list[Statement] = init_statements
        self.condition: Expression = condition
        self.update_statements: list[Statement] = update_statements
        self.body: BlockStmt = body

    def execute(self, scope: ExecutionScope) -> None:
        """
        Executes the C-style loop with per-iteration child scopes.

        Args:
            scope (ExecutionScope): Current execution scope.

        Raises:
            LoopLimitError: If the fuel limit is exhausted.
        """

        loop_scope: ExecutionScope = scope.create_child_scope()

        for init_statement in self.init_statements:
            init_statement.execute(scope=loop_scope)

        iteration: int = 0

        while True:
            _check_fuel(scope=scope, iteration=iteration, loop_repr=repr(self))

            condition_value: Value = self.condition.evaluate(scope=loop_scope)

            if not bool(condition_value.get()):
                break

            child_scope: ExecutionScope = loop_scope.create_child_scope()

            try:
                self.body.execute(scope=child_scope)
            except BreakSignal:
                break
            except ContinueSignal:
                pass

            for update_statement in self.update_statements:
                update_statement.execute(scope=loop_scope)

            iteration += 1

    def __repr__(self) -> str:
        """
        Returns string representation of the ForStmt.

        Returns:
            str: C-style loop representation.
        """

        inits: str = ", ".join(repr(init) for init in self.init_statements)
        updates: str = ", ".join(repr(update) for update in self.update_statements)

        return f"for ({inits} :: {repr(self.condition)} :: {updates}) {{ ... }}"


class WhileStmt(Statement):
    """
    AST statement executing a block while a condition holds.

    Attributes:
        condition (Expression): Boolean condition evaluated before each
            iteration.
        body (BlockStmt): Block executed once per iteration.
    """

    def __init__(self, condition: Expression, body: BlockStmt) -> None:
        """
        Initializes the WhileStmt instance.

        Args:
            condition (Expression): Loop condition.
            body (BlockStmt): Loop body block.
        """

        self.condition: Expression = condition
        self.body: BlockStmt = body

    def execute(self, scope: ExecutionScope) -> None:
        """
        Executes the loop body while the condition holds.

        Args:
            scope (ExecutionScope): Current execution scope.

        Raises:
            LoopLimitError: If the fuel limit is exhausted.
        """

        iteration: int = 0

        while True:
            _check_fuel(scope=scope, iteration=iteration, loop_repr=repr(self))

            condition_value: Value = self.condition.evaluate(scope=scope)

            if not bool(condition_value.get()):
                break

            child_scope: ExecutionScope = scope.create_child_scope()

            try:
                self.body.execute(scope=child_scope)
            except BreakSignal:
                break
            except ContinueSignal:
                pass

            iteration += 1

    def __repr__(self) -> str:
        """
        Returns string representation of the WhileStmt.

        Returns:
            str: While loop representation.
        """

        return f"while ({repr(self.condition)}) {{ ... }}"


class BreakStmt(Statement):
    """
    AST statement exiting the innermost enclosing loop.
    """

    def execute(self, scope: ExecutionScope) -> None:
        """
        Raises a BreakSignal to unwind to the innermost loop.

        Args:
            scope (ExecutionScope): Current execution scope.

        Raises:
            BreakSignal: Always, to exit the innermost loop.
        """

        raise BreakSignal()

    def __repr__(self) -> str:
        """
        Returns string representation of the BreakStmt.

        Returns:
            str: Break statement representation.
        """

        return "break;"


class ContinueStmt(Statement):
    """
    AST statement skipping to the next iteration of the innermost loop.
    """

    def execute(self, scope: ExecutionScope) -> None:
        """
        Raises a ContinueSignal to unwind to the innermost loop.

        Args:
            scope (ExecutionScope): Current execution scope.

        Raises:
            ContinueSignal: Always, to skip the iteration.
        """

        raise ContinueSignal()

    def __repr__(self) -> str:
        """
        Returns string representation of the ContinueStmt.

        Returns:
            str: Continue statement representation.
        """

        return "continue;"


class TargetAssignStmt(Statement):
    """
    AST statement assigning a Value through a member or index target.

    Covers targets such as 'self.count = x', 'self.readings[0] = y', and
    nested chains like 'self.child.name = z'. Compound forms ('+=', '++',
    ...) are desugared at parse time into a plain TargetAssignStmt whose
    value expression re-reads the target.

    Attributes:
        target (Expression): MemberAccessExpr or IndexExpr to assign through.
        value_expr (Expression): Expression producing the new Value.
    """

    def __init__(self, target: Expression, value_expr: Expression) -> None:
        """
        Initializes the TargetAssignStmt instance.

        Args:
            target (Expression): Member or index target to assign through.
            value_expr (Expression): Expression producing the new Value.
        """

        self.target: Expression = target
        self.value_expr: Expression = value_expr

    def execute(self, scope: ExecutionScope) -> None:
        """
        Evaluates the value and assigns it through the target expression.

        Args:
            scope (ExecutionScope): Current execution scope.
        """

        val: Value = self.value_expr.evaluate(scope=scope)

        self.target.assign_to(scope=scope, value=val)

    def __repr__(self) -> str:
        """
        Returns string representation of the target assignment statement.

        Returns:
            str: Assignment statement string.
        """

        return f"{repr(self.target)} = {repr(self.value_expr)};"
