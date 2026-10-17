"""
QuerySet collection supporting fluent chained filtering for Nadata.

Provides the QuerySet class enabling chained `.where((x: Type) -> predicate)`
queries, parallel chunk execution, pagination, counting, and iteration.
"""

# Import Modules
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
)

from src.core.struct.struct import Struct
from src.core.database.instance import StructInstance
from src.core.database.query_engine import (
    QueryExecutor,
    QueryOptimizer,
)
from src.core.expressions.scope import ExecutionScope
from src.core.expressions.expression import Expression

if TYPE_CHECKING:
    from src.core.database.database import NaDataDatabase


class QuerySet:
    """
    Fluent query collection enabling chained filtering, pagination, and inspection.

    Attributes:
        struct (Struct): Schema type of candidate instances in this set.
        database (Optional[NaDataDatabase]): Associated database engine reference.
        instances (list[StructInstance]): Current candidate instances in memory.
        query_executor (QueryExecutor): Engine used to run parallel chunk queries.
    """

    def __init__(
        self,
        struct: Struct,
        instances: list[StructInstance],
        database: Optional[NaDataDatabase] = None,
        query_executor: Optional[QueryExecutor] = None
    ) -> None:
        """
        Initializes the QuerySet instance.

        Args:
            struct (Struct): Associated Struct schema.
            instances (list[StructInstance]): Initial list of candidate instances.
            database (Optional[NaDataDatabase], optional): Database engine. Defaults to None.
            query_executor (Optional[QueryExecutor], optional): Query executor. Defaults to None.
        """

        self.struct: Struct = struct
        self.instances: list[StructInstance] = instances
        self.database: Optional[NaDataDatabase] = database
        self.query_executor: QueryExecutor = (
            query_executor if query_executor is not None else QueryExecutor()
        )

    def where(
        self,
        predicate: Callable[[StructInstance], bool]
    ) -> QuerySet:
        """
        Filters candidates using a callable predicate, returning a new chained QuerySet.

        Args:
            predicate (Callable[[StructInstance], bool]): Filtering predicate.

        Returns:
            QuerySet: New filtered QuerySet.
        """

        filtered_instances: list[StructInstance] = self.query_executor.filter(
            candidates=self.instances,
            predicate=predicate
        )

        return QuerySet(
            struct=self.struct,
            instances=filtered_instances,
            database=self.database,
            query_executor=self.query_executor
        )

    def where_expr(
        self,
        candidate_var: str,
        expr: Expression,
        scope: ExecutionScope
    ) -> QuerySet:
        """
        Filters candidates using an AST Expression with automatic loop-invariant hoisting.

        Args:
            candidate_var (str): Name of candidate variable (e.g. 'h').
            expr (Expression): Filter predicate expression AST.
            scope (ExecutionScope): Ambient execution scope.

        Returns:
            QuerySet: New filtered QuerySet.
        """

        # Pre-evaluate and hoist all loop-invariant sub-expressions
        optimized_expr: Expression = QueryOptimizer.hoist_invariants(
            expr=expr,
            scope=scope,
            loop_var=candidate_var
        )

        # Build evaluated predicate runner
        def eval_predicate(inst: StructInstance) -> bool:
            # Create child scope binding candidate instance to candidate_var
            child_scope: ExecutionScope = scope.create_child_scope()
            child_scope.globals_dict[candidate_var] = inst

            result = optimized_expr.evaluate(scope=child_scope)
            return bool(result.get())

        return self.where(predicate=eval_predicate)

    def count(self) -> int:
        """
        Returns total number of matching instances.

        Returns:
            int: Candidate count.
        """

        return len(self.instances)

    def first(self) -> Optional[StructInstance]:
        """
        Returns the first matching instance, or None if empty.

        Returns:
            Optional[StructInstance]: First instance or None.
        """

        if self.instances:
            return self.instances[0]

        return None

    def last(self) -> Optional[StructInstance]:
        """
        Returns the last matching instance, or None if empty.

        Returns:
            Optional[StructInstance]: Last instance or None.
        """

        if self.instances:
            return self.instances[-1]

        return None

    def limit(self, count: int) -> QuerySet:
        """
        Restricts the queryset to the first count instances.

        Args:
            count (int): Maximum number of instances.

        Returns:
            QuerySet: Sliced QuerySet.
        """

        return QuerySet(
            struct=self.struct,
            instances=self.instances[:count],
            database=self.database,
            query_executor=self.query_executor
        )

    def to_list(self) -> list[StructInstance]:
        """
        Returns raw list of contained instances.

        Returns:
            list[StructInstance]: Contained instances.
        """

        return list(self.instances)

    def __getitem__(self, index: int | slice) -> Any:
        """
        Retrieves an instance by index or returns a sliced QuerySet.

        Args:
            index (int | slice): Integer index or slice.

        Returns:
            Any: StructInstance or sliced QuerySet.
        """

        if isinstance(index, slice):
            return QuerySet(
                struct=self.struct,
                instances=self.instances[index],
                database=self.database,
                query_executor=self.query_executor
            )

        return self.instances[index]

    def __iter__(self) -> Iterator[StructInstance]:
        """
        Iterates over matching instances.

        Returns:
            Iterator[StructInstance]: Iterator over instances.
        """

        return iter(self.instances)


    def __len__(self) -> int:
        """
        Returns count of instances.

        Returns:
            int: Instance count.
        """

        return len(self.instances)

    def __repr__(self) -> str:
        """
        Returns string representation of the QuerySet.

        Returns:
            str: Representation showing struct and count.
        """

        return f"<QuerySet[{self.struct.name}] count={len(self.instances)}>"
