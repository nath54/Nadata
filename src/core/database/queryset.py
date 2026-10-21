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
from src.core.values.values import Value
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
            predicate=(
                predicate.get()
                if isinstance(predicate, Value) and callable(predicate.get())
                else predicate
            )
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

    def limit(self, count: int | Value) -> QuerySet:
        """
        Restricts the queryset to the first count instances.

        Args:
            count (int | Value): Maximum number of instances (language-side
                calls pass a wrapped integer Value).

        Returns:
            QuerySet: Sliced QuerySet.

        Raises:
            TypeError: If the count is not an integer.
        """

        raw_count: Any = count.get() if isinstance(count, Value) else count

        if not isinstance(raw_count, int) or isinstance(raw_count, bool):
            raise TypeError(f"Limit must be an integer, got '{repr(raw_count)}'")

        return QuerySet(
            struct=self.struct,
            instances=self.instances[:raw_count],
            database=self.database,
            query_executor=self.query_executor
        )

    def order_by(self, attribute: str | Value, descending: bool = False) -> QuerySet:
        """
        Returns a new QuerySet sorted by an attribute's raw payload.

        The sort is stable. Instances whose struct (or a parent struct)
        does not declare the attribute raise a lookup error.

        Args:
            attribute (str): Name of the attribute to sort by.
            descending (bool, optional): Sort descending. Defaults to False.

        Returns:
            QuerySet: New sorted QuerySet.

        Raises:
            AttributeError: If the attribute is not declared in the struct
                schema.
        """

        attribute = self._attr_name(attribute=attribute)

        self._validate_attribute(name=attribute)

        def sort_key(inst: StructInstance) -> Any:
            return self._payload_of(inst=inst, attribute=attribute)

        return QuerySet(
            struct=self.struct,
            instances=sorted(
                self.instances,
                key=sort_key,
                reverse=descending,
            ),
            database=self.database,
            query_executor=self.query_executor,
        )

    def exists(self) -> bool:
        """
        Returns whether at least one instance matches.

        Returns:
            bool: True if the set is non-empty.
        """

        return bool(self.instances)

    def pluck(self, attribute: str | Value) -> list[Any]:
        """
        Extracts the raw payload of an attribute from every instance.

        Args:
            attribute (str): Name of the attribute to extract.

        Returns:
            list[Any]: Raw payloads in instance order.

        Raises:
            AttributeError: If the attribute is not declared in the struct
                schema.
        """

        attribute = self._attr_name(attribute=attribute)

        self._validate_attribute(name=attribute)

        return [self._payload_of(inst=inst, attribute=attribute) for inst in self.instances]

    def sum_of(self, attribute: str | Value) -> Any:
        """
        Sums a numeric attribute over every instance.

        Args:
            attribute (str): Name of the numeric attribute.

        Returns:
            Any: The sum payload (int or float), 0 for an empty set.

        Raises:
            AttributeError: If the attribute is not declared.
            TypeError: If a payload is not numeric.
        """

        attribute = self._attr_name(attribute=attribute)

        values: list[Any] = self.pluck(attribute=attribute)
        self._validate_numeric(attribute=attribute, values=values)

        return sum(values)

    def min_of(self, attribute: str | Value) -> Any:
        """
        Returns the minimum payload of a numeric attribute.

        Args:
            attribute (str): Name of the numeric attribute.

        Returns:
            Any: The minimum payload.

        Raises:
            AttributeError: If the attribute is not declared.
            TypeError: If a payload is not numeric.
            ValueError: If the set is empty.
        """

        attribute = self._attr_name(attribute=attribute)

        values: list[Any] = self.pluck(attribute=attribute)
        self._validate_numeric(attribute=attribute, values=values)

        return min(values)

    def max_of(self, attribute: str | Value) -> Any:
        """
        Returns the maximum payload of a numeric attribute.

        Args:
            attribute (str): Name of the numeric attribute.

        Returns:
            Any: The maximum payload.

        Raises:
            AttributeError: If the attribute is not declared.
            TypeError: If a payload is not numeric.
            ValueError: If the set is empty.
        """

        attribute = self._attr_name(attribute=attribute)

        values: list[Any] = self.pluck(attribute=attribute)
        self._validate_numeric(attribute=attribute, values=values)

        return max(values)

    def average_of(self, attribute: str | Value) -> float:
        """
        Returns the arithmetic mean of a numeric attribute.

        Args:
            attribute (str): Name of the numeric attribute.

        Returns:
            float: The mean payload.

        Raises:
            AttributeError: If the attribute is not declared.
            TypeError: If a payload is not numeric.
            ValueError: If the set is empty.
        """

        attribute = self._attr_name(attribute=attribute)

        values: list[Any] = self.pluck(attribute=attribute)
        self._validate_numeric(attribute=attribute, values=values)

        return float(sum(values)) / len(values)

    def _payload_of(self, inst: StructInstance, attribute: str) -> Any:
        """
        Resolves an attribute's raw payload from an instance.

        Falls back to the declared default when the instance does not
        carry the attribute explicitly (lazy defaults).

        Args:
            inst (StructInstance): Instance to read.
            attribute (str): Attribute name.

        Returns:
            Any: Raw payload.

        Raises:
            KeyError: If the instance has no value and no default exists.
        """

        if attribute in inst.attributes:
            return inst.attributes[attribute].get()

        attr_def: Any = self.struct.get_attribute(name=attribute)

        if attr_def is not None and attr_def.has_default:
            default: Any = attr_def.default_value

            return default.get() if isinstance(default, Value) else default

        raise KeyError(
            f"Instance of '{inst.struct.name}' has no value for attribute "
            f"'{attribute}'"
        )

    def _attr_name(self, attribute: str | Value) -> str:
        """
        Normalizes an attribute name argument from any caller.

        Language-side calls pass string literals as Value wrappers.

        Args:
            attribute (str | Value): Raw name or wrapped string Value.

        Returns:
            str: The plain attribute name.

        Raises:
            TypeError: If the argument is not a string.
        """

        raw: Any = attribute.get() if isinstance(attribute, Value) else attribute

        if not isinstance(raw, str):
            raise TypeError(
                f"Attribute name must be a string, got '{repr(raw)}'"
            )

        return raw

    def _validate_attribute(self, name: str) -> None:
        """
        Checks that an attribute is declared in the struct schema.

        Args:
            name (str): Attribute name.

        Raises:
            AttributeError: If the attribute is not declared.
        """

        if self.struct.get_attribute(name=name) is None:
            raise AttributeError(
                f"Attribute '{name}' is not declared in the schema of "
                f"'{self.struct.name}'"
            )

    def _validate_numeric(self, attribute: str, values: list[Any]) -> None:
        """
        Checks that every extracted payload is numeric.

        Args:
            attribute (str): Attribute name (for the error message).
            values (list[Any]): Extracted payloads.

        Raises:
            TypeError: If a payload is not an int or float.
        """

        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"Attribute '{attribute}' payload '{repr(value)}' is not numeric"
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
