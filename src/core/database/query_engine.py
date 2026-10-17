"""
Query optimizer, loop invariant hoisting, and parallel execution engine for Nadata.

Optimizes predicate expressions by pre-evaluating invariant sub-expressions
(e.g., target.age(now) computed once instead of N times) and parallelizing
chunk execution across candidate instances.
"""

# Import Modules
from __future__ import annotations
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor

from src.core.values.values import Value
from src.core.database.instance import StructInstance
from src.core.expressions.scope import ExecutionScope

from src.core.expressions.expression import (
    CallExpr,
    Expression,
    LiteralExpr,
    BinaryOpExpr,
    IdentifierExpr,
    MemberAccessExpr,
)



class QueryOptimizer:
    """
    Analyzes AST expressions to identify and hoist loop-invariant sub-expressions.
    """

    @staticmethod
    def is_invariant(expr: Expression, loop_var: str = "self") -> bool:
        """
        Determines if an expression is independent of the loop variable.

        Args:
            expr (Expression): Expression node to analyze.
            loop_var (str, optional): Name of the candidate variable. Defaults to "self".

        Returns:
            bool: True if expression does not reference loop_var.
        """

        if isinstance(expr, LiteralExpr):
            return True

        if isinstance(expr, IdentifierExpr):
            return expr.name != loop_var

        if isinstance(expr, MemberAccessExpr):
            return QueryOptimizer.is_invariant(expr=expr.target, loop_var=loop_var)

        if isinstance(expr, CallExpr):
            callee_inv: bool = QueryOptimizer.is_invariant(expr=expr.callee, loop_var=loop_var)

            if not callee_inv:
                return False

            args_inv: bool = all(
                QueryOptimizer.is_invariant(expr=arg, loop_var=loop_var)
                for arg in expr.args
            )

            if not args_inv:
                return False

            kwargs_inv: bool = all(
                QueryOptimizer.is_invariant(expr=val, loop_var=loop_var)
                for val in expr.kwargs.values()
            )

            return kwargs_inv

        if isinstance(expr, BinaryOpExpr):
            return (
                QueryOptimizer.is_invariant(expr=expr.left, loop_var=loop_var)
                and QueryOptimizer.is_invariant(expr=expr.right, loop_var=loop_var)
            )

        return False

    @staticmethod
    def hoist_invariants(
        expr: Expression,
        scope: ExecutionScope,
        loop_var: str = "self"
    ) -> Expression:
        """
        Replaces invariant sub-expressions with pre-evaluated LiteralExpr nodes.

        Args:
            expr (Expression): Expression to optimize.
            scope (ExecutionScope): Query evaluation scope.
            loop_var (str, optional): Target loop variable name. Defaults to "self".

        Returns:
            Expression: Optimized AST with hoisted constant values.
        """

        # If the entire expression is invariant, evaluate it once immediately
        if QueryOptimizer.is_invariant(expr=expr, loop_var=loop_var):
            evaluated: Value = expr.evaluate(scope=scope)
            return LiteralExpr(value=evaluated)

        # Recursively optimize children in binary operations
        if isinstance(expr, BinaryOpExpr):
            new_left: Expression = QueryOptimizer.hoist_invariants(
                expr=expr.left,
                scope=scope,
                loop_var=loop_var
            )
            new_right: Expression = QueryOptimizer.hoist_invariants(
                expr=expr.right,
                scope=scope,
                loop_var=loop_var
            )

            return BinaryOpExpr(left=new_left, op=expr.op, right=new_right)

        # Recursively optimize call arguments
        if isinstance(expr, CallExpr):
            new_args: list[Expression] = [
                QueryOptimizer.hoist_invariants(expr=a, scope=scope, loop_var=loop_var)
                for a in expr.args
            ]
            new_kwargs: dict[str, Expression] = {
                k: QueryOptimizer.hoist_invariants(expr=v, scope=scope, loop_var=loop_var)
                for k, v in expr.kwargs.items()
            }

            return CallExpr(callee=expr.callee, args=new_args, kwargs=new_kwargs)

        return expr


class QueryExecutor:
    """
    Executes optimized queries across instance collections in parallel chunks.

    Attributes:
        chunk_size (int): Size of candidate batch chunks for parallel workers.
        max_workers (int): Number of parallel thread workers.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        max_workers: Optional[int] = None
    ) -> None:
        """
        Initializes the QueryExecutor.

        Args:
            chunk_size (int, optional): Instances per batch chunk. Defaults to 1024.
            max_workers (Optional[int], optional): Worker thread count. Defaults to None.
        """

        self.chunk_size: int = chunk_size
        self.max_workers: Optional[int] = max_workers

    def filter(
        self,
        candidates: list[StructInstance],
        predicate: Callable[[StructInstance], bool]
    ) -> list[StructInstance]:
        """
        Filters candidates using parallel chunk workers.

        Args:
            candidates (list[StructInstance]): Instances to filter.
            predicate (Callable[[StructInstance], bool]): Filtering predicate.

        Returns:
            list[StructInstance]: Matching instances.
        """

        if not candidates:
            return []

        # If candidates count is small, avoid thread pool overhead
        if len(candidates) <= self.chunk_size:
            return [inst for inst in candidates if predicate(inst)]

        # Partition into chunks
        chunks: list[list[StructInstance]] = [
            candidates[i:i + self.chunk_size]
            for i in range(0, len(candidates), self.chunk_size)
        ]

        results: list[StructInstance] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._filter_chunk, chunk=chunk, predicate=predicate)
                for chunk in chunks
            ]

            for future in futures:
                results.extend(future.result())

        return results

    @staticmethod
    def _filter_chunk(
        chunk: list[StructInstance],
        predicate: Callable[[StructInstance], bool]
    ) -> list[StructInstance]:
        """
        Worker function filtering a single chunk.

        Args:
            chunk (list[StructInstance]): Chunk of instances.
            predicate (Callable[[StructInstance], bool]): Predicate to apply.

        Returns:
            list[StructInstance]: Filtered chunk subset.
        """

        return [inst for inst in chunk if predicate(inst)]
