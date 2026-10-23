"""
Expression AST nodes and evaluation engine for Nadata method bodies.

Provides AST representations for literals, identifiers, unary operations,
binary operations, member accesses, and function/method calls operating on Value objects.
"""

# Import Modules
from __future__ import annotations
from collections import deque
from itertools import product
from typing import Any, Optional, cast

from src.core.types.types import (
    Int64,
    TypeStr,
    Float32,
    TypeBool,
    Type,
    TypeInt,
    TypeFloat,
    TypeArray,
    TypeDict,
    TypeQueue,
    TypePriorityQueue,
    TypeSpatialGrid,
    TypeSet,
    Float64,
    UInt64,
    TypeTensor,
    flatten_nested_lists,
)
from src.core.values.values import (
    TYPE_EXTERNAL,
    Value,
    ValueStr,
    ValueInt,
    ValueBool,
    ValueColor,
    ValueFloat,
    ValueArray,
    ValueDict,
    ValueDateTime,
    ValueExternal,
    ValuePriorityQueue,
    ValueQueue,
    ValueSet,
    ValueSpatialGrid,
    ValueTensor,
    make_value,
)
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime
from src.core.expressions.autograd import (
    Optimizer,
    maybe_record_binary,
    maybe_record_member,
    maybe_record_slice,
)
from src.core.expressions.builtins import (
    DictCollectionCallable,
    GridCallable,
    LengthCallable,
    QueueCallable,
    SetCallable,
    TensorCallable,
    _is_struct_instance,
    _struct_from_object,
    resolve_virtual_callable,
)
from src.core.expressions.scope import ScopeAwareFunction, ExecutionScope



class Expression:
    """
    Abstract base class for all AST expression nodes that evaluate to a typed Value.
    """

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates the expression within the given scope, returning a typed Value.

        Args:
            scope (ExecutionScope): Current execution environment.

        Returns:
            Value: Typed runtime Value resulting from the evaluation.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError("evaluate must be implemented by subclasses.")

    def assign_to(self, scope: ExecutionScope, value: Value) -> None:
        """
        Assigns a Value through this expression as an assignment target.

        Args:
            scope (ExecutionScope): Current execution environment.
            value (Value): Value to assign.

        Raises:
            TypeError: If this expression cannot be an assignment target.
        """

        raise TypeError(f"Cannot assign to '{repr(self)}'")

    def ensure_materialized(self, scope: ExecutionScope) -> None:
        """
        Prepares the target chain for mutation by materializing lazy
        defaults along the access path.

        Args:
            scope (ExecutionScope): Current execution environment.
        """


class LiteralExpr(Expression):
    """
    AST node representing a literal constant typed Value.

    Attributes:
        value (Value): Typed constant runtime Value.
    """

    def __init__(self, value: Value | Any) -> None:
        """
        Initializes the LiteralExpr instance, wrapping native primitives if needed.

        Args:
            value (Value | Any): Typed Value or native Python literal.
        """

        if isinstance(value, Value):
            self.value: Value = value
        elif isinstance(value, bool):
            self.value = ValueBool(value_type=TypeBool(), value=value)
        elif isinstance(value, int):

            # Unsigned 64 bit max does not fit the default signed Int64
            int_type: TypeInt = Int64 if value <= 2**63 - 1 else UInt64

            self.value = ValueInt(value_type=int_type, value=value)
        elif isinstance(value, float):
            self.value = ValueFloat(value_type=Float64, value=value)
        elif isinstance(value, str):
            self.value = ValueStr(value_type=TypeStr(), value=value)
        elif isinstance(value, DateTime):
            self.value = ValueDateTime(value=value)
        elif isinstance(value, Color):
            self.value = ValueColor(value=value)
        else:
            raise TypeError(f"Cannot construct literal from type '{type(value).__name__}'")

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Returns the constant Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Stored constant Value.
        """

        return self.value

    def __repr__(self) -> str:
        """
        Returns string representation of the LiteralExpr.

        Returns:
            str: Representation showing the literal Value.
        """

        return repr(self.value)


class IdentifierExpr(Expression):
    """
    AST node representing an identifier lookup (variable, parameter, or member).

    Attributes:
        name (str): Identifier name to look up.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the IdentifierExpr instance.

        Args:
            name (str): Identifier name.
        """

        self.name: str = name

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Resolves the identifier in the given scope.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Value bound to the identifier.

        Raises:
            TypeError: If the resolved identifier is not a Value instance.
        """

        resolved: Any = scope.get(name=self.name)

        if isinstance(resolved, Value):
            return resolved

        # Callables (standalone functions, builtin constructors) pass as
        # opaque values so they can be handed to higher-order members
        # like Tensor.map(); call resolution goes through evaluate_symbol
        # and never reaches this path
        if callable(resolved) or hasattr(resolved, "execute"):
            return ValueExternal(payload=resolved)

        raise TypeError(
            f"Identifier '{self.name}' resolved to non-Value object '{type(resolved).__name__}'"
        )

    def evaluate_symbol(self, scope: ExecutionScope) -> Any:
        """
        Resolves the identifier to any object (including types and non-Value callables).

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Any: Resolved symbol object.
        """

        return scope.get(name=self.name)

    def __repr__(self) -> str:
        """
        Returns string representation of the IdentifierExpr.

        Returns:
            str: Identifier name.
        """

        return self.name


class UnaryOpExpr(Expression):
    """
    AST node representing a unary operation on a Value.

    Attributes:
        op (str): Unary operator symbol ('-', 'not', '!').
        operand (Expression): Operand expression.
    """

    def __init__(self, op: str, operand: Expression) -> None:
        """
        Initializes the UnaryOpExpr instance.

        Args:
            op (str): Operator string ('-', 'not', '!').
            operand (Expression): Operand expression.
        """

        self.op: str = op
        self.operand: Expression = operand

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates the unary operation on its operand, returning a typed Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Resulting typed Value.

        Raises:
            ValueError: If operator is unsupported for the operand type.
        """

        val: Value = self.operand.evaluate(scope=scope)

        if self.op == "-":
            if isinstance(val, ValueInt):
                return ValueInt(value_type=val.value_type, value=-val.get())

            if isinstance(val, ValueFloat):
                return ValueFloat(value_type=val.value_type, value=-val.get())

        if self.op in ("not", "!"):
            if isinstance(val, ValueBool):
                return ValueBool(value_type=TypeBool(), value=not val.get())

            return ValueBool(value_type=TypeBool(), value=not bool(val.get()))

        raise ValueError(f"Unsupported unary operator '{self.op}' for {type(val).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of the UnaryOpExpr.

        Returns:
            str: Operator and operand representation.
        """

        return f"({self.op}{repr(self.operand)})"


class BinaryOpExpr(Expression):
    """
    AST node representing a binary operation between two Value expressions.

    Attributes:
        left (Expression): Left-hand side operand.
        op (str): Binary operator symbol.
        right (Expression): Right-hand side operand.
    """

    def __init__(
        self,
        left: Expression,
        op: str,
        right: Expression
    ) -> None:
        """
        Initializes the BinaryOpExpr instance.

        Args:
            left (Expression): Left operand.
            op (str): Binary operator symbol.
            right (Expression): Right operand.
        """

        self.left: Expression = left
        self.op: str = op
        self.right: Expression = right

    @staticmethod
    def _eval_tensor_arithmetic(
        left: Value,
        op: str,
        right: Value,
    ) -> Optional[Value]:
        """
        Evaluates elementwise tensor arithmetic.

        Tensor-tensor operands must share the same shape; scalars
        broadcast across every cell. The result keeps the base tensor's
        shape and element type.

        Args:
            left (Value): Left operand Value.
            op (str): Arithmetic operator ('+', '-', '*', '/').
            right (Value): Right operand Value.

        Returns:
            Optional[Value]: Resulting ValueTensor, or None if the
                operator is unsupported for tensors.

        Raises:
            ValueError: On a tensor-tensor shape mismatch.
            TypeError: On a non-numeric scalar operand, integer tensor
                division, or unbound symbolic dimensions.
        """

        if op not in ("+", "-", "*", "/"):
            return None

        base: ValueTensor
        other: Value
        scalar_first: bool

        if isinstance(left, ValueTensor):
            base, other, scalar_first = left, right, False

        elif isinstance(right, ValueTensor):
            base, other, scalar_first = right, left, True

        else:
            return None

        extents: list[int] = BinaryOpExpr._tensor_extents(tensor=base)
        element_type: Type = base.value_type.value_type

        def apply(a: Any, b: Any) -> Any:
            """Applies the arithmetic operator to one cell pair."""

            if op == "+":
                return a + b

            if op == "-":
                return a - b

            if op == "*":
                return a * b

            return a / b

        if isinstance(other, ValueTensor):

            if BinaryOpExpr._tensor_extents(tensor=other) != extents:
                raise ValueError(
                    f"Tensor shape mismatch for '{op}': "
                    f"{extents} vs "
                    f"{BinaryOpExpr._tensor_extents(tensor=other)}"
                )

            if scalar_first:
                cells: list[Any] = [
                    apply(other_cell, base_cell)
                    for base_cell, other_cell in zip(base.cells, other.cells)
                ]

            else:
                cells = [
                    apply(base_cell, other_cell)
                    for base_cell, other_cell in zip(base.cells, other.cells)
                ]

        else:
            raw_scalar: Any = other.get()

            if isinstance(raw_scalar, bool) or not isinstance(
                raw_scalar, (int, float)
            ):
                raise TypeError(
                    f"Tensor '{op}' requires a numeric scalar, got "
                    f"'{repr(raw_scalar)}'"
                )

            if scalar_first:
                cells = [apply(raw_scalar, cell) for cell in base.cells]

            else:
                cells = [apply(cell, raw_scalar) for cell in base.cells]

        if op == "/" and isinstance(element_type, TypeInt):
            raise TypeError(
                "Tensor division on an integer element type produces "
                "floats; use a float element type"
            )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=element_type,
                dims=cast("list[int | str]", extents),
            ),
            cells=cells,
        )

    @staticmethod
    def _tensor_extents(tensor: ValueTensor) -> list[int]:
        """
        Returns the concrete per-axis extents of a tensor.

        Args:
            tensor (ValueTensor): Tensor to inspect.

        Returns:
            list[int]: Extents in axis order.

        Raises:
            TypeError: If the tensor has unbound symbolic dimensions.
        """

        if tensor.concrete_dims is not None:
            return list(tensor.concrete_dims)

        if tensor.value_type.is_symbolic:
            raise TypeError(
                "Cannot operate on a tensor with unbound symbolic dimensions"
            )

        return [dim for dim in tensor.value_type.dims if isinstance(dim, int)]

    @staticmethod
    def _eval_int_arithmetic(left: ValueInt, op: str, right: ValueInt) -> Optional[Value]:
        """
        Evaluates integer arithmetic operations.

        Args:
            left (ValueInt): Left integer operand.
            op (str): Arithmetic operator.
            right (ValueInt): Right integer operand.

        Returns:
            Optional[Value]: Resulting typed Value, or None if unknown op.
        """

        raw_left: int = left.get()
        raw_right: int = right.get()

        if op == "+":
            return ValueInt(value_type=left.value_type, value=raw_left + raw_right)
        if op == "-":
            return ValueInt(value_type=left.value_type, value=raw_left - raw_right)
        if op == "*":
            return ValueInt(value_type=left.value_type, value=raw_left * raw_right)
        if op == "/":
            return ValueFloat(value_type=Float32, value=float(raw_left / raw_right))
        if op == "%":
            return ValueInt(value_type=left.value_type, value=raw_left % raw_right)

        return None

    @staticmethod
    def _eval_float_arithmetic(left: Value, op: str, right: Value) -> Optional[Value]:
        """
        Evaluates floating point arithmetic operations.

        Args:
            left (Value): Left numeric operand.
            op (str): Arithmetic operator.
            right (Value): Right numeric operand.

        Returns:
            Optional[Value]: Resulting typed ValueFloat, or None if unknown op.
        """

        raw_left: float = float(left.get())
        raw_right: float = float(right.get())
        flt_type: TypeFloat = (
            left.value_type if isinstance(left, ValueFloat)
            else (right.value_type if isinstance(right, ValueFloat) else Float32)
        )

        if op == "+":
            return ValueFloat(value_type=flt_type, value=raw_left + raw_right)
        if op == "-":
            return ValueFloat(value_type=flt_type, value=raw_left - raw_right)
        if op == "*":
            return ValueFloat(value_type=flt_type, value=raw_left * raw_right)
        if op == "/":
            return ValueFloat(value_type=flt_type, value=raw_left / raw_right)

        return None

    @staticmethod
    def _eval_arithmetic(left: Value, op: str, right: Value) -> Optional[Value]:
        """
        Evaluates arithmetic binary operations between two Value instances.

        Args:
            left (Value): Left operand Value.
            op (str): Arithmetic operator.
            right (Value): Right operand Value.

        Returns:
            Optional[Value]: Resulting typed Value, or None if not an arithmetic op.
        """

        if op not in ("+", "-", "*", "/", "%"):
            return None

        # String concatenation: strictly Str + Str, nothing else
        if op == "+" and (
            isinstance(left, ValueStr) or isinstance(right, ValueStr)
        ):
            if not (isinstance(left, ValueStr) and isinstance(right, ValueStr)):
                raise ValueError(
                    f"Cannot concatenate {type(left).__name__} and "
                    f"{type(right).__name__}: '+' on Str requires two Str operands"
                )

            return ValueStr(
                value_type=left.value_type,
                value=str(left.get()) + str(right.get()),
            )

        # Check if left value has custom operator method (e.g. DateTime - DateTime)
        raw_left: Any = left.get()
        raw_right: Any = right.get()

        if hasattr(raw_left, "__sub__") and op == "-":
            try:
                custom_diff: Any = raw_left - raw_right
            except TypeError:
                custom_diff = None

            if isinstance(custom_diff, Value):
                return custom_diff

            if isinstance(custom_diff, DateTime):
                return ValueDateTime(value=custom_diff)

        if hasattr(raw_left, "__add__") and op == "+":
            try:
                custom_sum: Any = raw_left + raw_right
            except TypeError:
                custom_sum = None

            if isinstance(custom_sum, Value):
                return custom_sum

            if isinstance(custom_sum, DateTime):
                return ValueDateTime(value=custom_sum)

        # Tensor arithmetic: elementwise tensor-tensor (same shape) or
        # broadcast tensor-scalar
        if isinstance(left, ValueTensor) or isinstance(right, ValueTensor):
            return BinaryOpExpr._eval_tensor_arithmetic(
                left=left, op=op, right=right
            )

        # Integer arithmetic
        if isinstance(left, ValueInt) and isinstance(right, ValueInt):
            return BinaryOpExpr._eval_int_arithmetic(left=left, op=op, right=right)

        # Floating point arithmetic
        if isinstance(left, (ValueInt, ValueFloat)) and isinstance(right, (ValueInt, ValueFloat)):
            return BinaryOpExpr._eval_float_arithmetic(left=left, op=op, right=right)

        return None

    @staticmethod
    def _eval_comparison(left: Value, op: str, right: Value) -> Optional[ValueBool]:
        """
        Evaluates comparison binary operations between two Value instances.

        Args:
            left (Value): Left operand Value.
            op (str): Comparison operator.
            right (Value): Right operand Value.

        Returns:
            Optional[ValueBool]: ValueBool comparison result, or None if not a comparison op.
        """

        if op not in ("==", "!=", "<=", "<", ">=", ">"):
            return None

        raw_left: Any = left.get()
        raw_right: Any = right.get()

        # None payloads (missing get_by_nid / get_setting results) only
        # support equality: None == None is True, None ordered against
        # anything is False — ordering a missing value is meaningless
        left_missing: bool = raw_left is None or (
            isinstance(left, ValueExternal) and left.payload is None
        )
        right_missing: bool = raw_right is None or (
            isinstance(right, ValueExternal) and right.payload is None
        )

        if left_missing or right_missing:

            if op == "==":
                result: bool = left_missing and right_missing

            elif op == "!=":
                result = not (left_missing and right_missing)

            else:
                result = False

            return ValueBool(value_type=TypeBool(), value=result)

        # Evaluate only the requested operator (a table of all six would
        # eagerly compute e.g. '<=' on strings and crash on valid '!=')
        if op == "==":
            return ValueBool(value_type=TypeBool(), value=bool(raw_left == raw_right))

        if op == "!=":
            return ValueBool(value_type=TypeBool(), value=bool(raw_left != raw_right))

        if op == "<=":
            return ValueBool(value_type=TypeBool(), value=bool(raw_left <= raw_right))

        if op == "<":
            return ValueBool(value_type=TypeBool(), value=bool(raw_left < raw_right))

        if op == ">=":
            return ValueBool(value_type=TypeBool(), value=bool(raw_left >= raw_right))

        return ValueBool(value_type=TypeBool(), value=bool(raw_left > raw_right))

    @staticmethod
    def _eval_membership(left: Value, op: str, right: Value) -> Optional[ValueBool]:
        """
        Evaluates membership binary operations between two Value instances.

        Supported containers: Array (element scan), Dict (key probe), and
        Str (substring search) — all delegated to the native Python
        containment check on the raw payload, so no interpreted loop and
        no loop fuel is involved.

        Args:
            left (Value): Needle operand Value.
            op (str): Membership operator ('in' or 'not in').
            right (Value): Container operand Value.

        Returns:
            Optional[ValueBool]: ValueBool membership result, or None if
            not a membership op.

        Raises:
            TypeError: If the container kind is unsupported or the needle
                type does not fit a Str container.
        """

        if op not in ("in", "not in"):
            return None

        raw_left: Any = left.get()
        raw_right: Any = right.get()

        found: bool

        # Sparse grids probe occupied-cell values (raw payloads are
        # coordinate keyed dicts)
        if isinstance(right, ValueSpatialGrid):
            found = any(cell == raw_left for cell in right.cells.values())

        # Tensors probe flat row-major elements
        elif isinstance(right, ValueTensor):
            found = any(cell == raw_left for cell in right.cells)

        # Priority queues probe elements (raw payloads are priority pairs)
        elif isinstance(right, ValuePriorityQueue):
            found = any(
                element == raw_left for _, element in right.entries
            )

        elif isinstance(raw_right, str):
            if not isinstance(raw_left, str):
                raise TypeError(
                    f"'in' on a Str requires a Str needle, got "
                    f"{type(raw_left).__name__}"
                )

            found = raw_left in raw_right

        elif isinstance(raw_right, dict):
            found = raw_left in raw_right

        elif isinstance(raw_right, (list, tuple, set, deque)):
            found = raw_left in raw_right

        else:
            raise TypeError(
                f"'in' requires an Array, Dict, Set, Queue, or Str "
                f"container, got {type(raw_right).__name__}"
            )

        return ValueBool(
            value_type=TypeBool(),
            value=(found if op == "in" else not found),
        )

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates the binary expression, returning a typed Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Resulting typed Value.

        Raises:
            ValueError: If the operator is unsupported for the operands.
        """

        left_val: Value = self.left.evaluate(scope=scope)
        right_val: Value = self.right.evaluate(scope=scope)

        # 1. Attempt arithmetic evaluation
        arith_res: Optional[Value] = self._eval_arithmetic(left_val, self.op, right_val)

        if arith_res is not None:

            # Autograd: record tensor arithmetic when any Float input tracks
            if isinstance(left_val, ValueTensor) or isinstance(
                right_val, ValueTensor
            ):
                maybe_record_binary(
                    op=self.op,
                    left=left_val,
                    right=right_val,
                    result=arith_res,
                    tape=scope.tape,
                )

            # Autograd: record scalar float chains rooted at recorded ops
            elif (
                isinstance(left_val, (ValueFloat, ValueInt))
                and isinstance(right_val, (ValueFloat, ValueInt))
                and scope.tape is not None
            ):
                scope.tape.maybe_record_scalar(
                    op=self.op,
                    inputs=[left_val, right_val],
                    output=arith_res,
                )

            return arith_res

        # 2. Attempt comparison evaluation
        comp_res: Optional[ValueBool] = self._eval_comparison(left_val, self.op, right_val)

        if comp_res is not None:
            return comp_res

        # 2b. Attempt membership evaluation ('in' / 'not in')
        member_res: Optional[ValueBool] = self._eval_membership(left_val, self.op, right_val)

        if member_res is not None:
            return member_res

        # 3. Logical operations (keyword only: 'and', 'or', 'xor', 'nand')
        if self.op == "and":
            bool_and: bool = bool(left_val.get()) and bool(right_val.get())
            return ValueBool(value_type=TypeBool(), value=bool_and)

        if self.op == "or":
            bool_or: bool = bool(left_val.get()) or bool(right_val.get())
            return ValueBool(value_type=TypeBool(), value=bool_or)

        if self.op == "xor":
            bool_xor: bool = bool(left_val.get()) != bool(right_val.get())
            return ValueBool(value_type=TypeBool(), value=bool_xor)

        if self.op == "nand":
            bool_nand: bool = not (
                bool(left_val.get()) and bool(right_val.get())
            )
            return ValueBool(value_type=TypeBool(), value=bool_nand)

        raise ValueError(
            f"Unsupported operator '{self.op}' for {type(left_val).__name__} "
            f"and {type(right_val).__name__}"
        )

    def __repr__(self) -> str:
        """
        Returns string representation of the BinaryOpExpr.

        Returns:
            str: Infix expression representation.
        """

        return f"({repr(self.left)} {self.op} {repr(self.right)})"


class MemberAccessExpr(Expression):
    """
    AST node representing member access on a Value (e.g. target.member).

    Attributes:
        target (Expression): Target expression whose member is accessed.
        member (str): Attribute or method identifier name.
    """

    def __init__(self, target: Expression, member: str) -> None:
        """
        Initializes the MemberAccessExpr instance.

        Args:
            target (Expression): Target expression.
            member (str): Member name string.
        """

        self.target: Expression = target
        self.member: str = member

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates target and retrieves the member attribute as a Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Attribute Value.

        Raises:
            AttributeError: If member does not exist or is not a Value.
        """

        target_obj: Any = self._resolve_target(scope=scope)

        # Unwrap Value containers to their raw object for member access
        # (e.g. a ValueDateTime wrapping a DateTime exposes 'year')
        raw_target: Any = target_obj.get() if isinstance(target_obj, Value) else target_obj

        # Virtual member on an instance: the entity id (per-struct index)
        if self.member == "id" and hasattr(raw_target, "entity_id"):
            return LiteralExpr(value=raw_target.entity_id.local_id).evaluate(scope=scope)

        # Virtual member on an instance: the named id (None if anonymous)
        if self.member == "nid" and hasattr(raw_target, "entity_id"):

            nid_value: Any = getattr(raw_target, "nid", None)

            if nid_value is None:
                return ValueExternal(payload=None)

            return LiteralExpr(value=nid_value).evaluate(scope=scope)

        # Member access on a struct type: the attribute's declared type
        # (instances are excluded: their member access resolves to values)
        struct_from_type: Optional[Any] = (
            None
            if _is_struct_instance(obj=raw_target)
            else _struct_from_object(obj=raw_target)
        )

        if struct_from_type is not None:
            attr_def: Any = struct_from_type.get_attribute(name=self.member)

            if attr_def is not None:
                return ValueExternal(payload=attr_def.attr_type)

            raise AttributeError(
                f"'{self.member}' is not declared in the schema of "
                f"'{struct_from_type.name}'"
            )

        # Check dictionary-backed attributes on struct instances
        if hasattr(raw_target, "attributes"):
            attrs: dict[str, Any] = getattr(raw_target, "attributes")

            # Materialize lazily-defaulted attributes so access (and any
            # mutation through it) never aliases the schema's shared default
            if (
                self.member not in attrs
                and hasattr(raw_target, "ensure_attribute")
            ):
                raw_target.ensure_attribute(name=self.member)

            if self.member in attrs:
                attr_val: Any = attrs[self.member]

                if isinstance(attr_val, Value):
                    return attr_val

        # Check native Python object attributes
        if hasattr(raw_target, self.member):
            native_val: Any = getattr(raw_target, self.member)

            if isinstance(native_val, Value):
                return native_val

            # Wrap raw native members (e.g. DateTime.year) into typed Values
            return LiteralExpr(value=native_val).evaluate(scope=scope)

        raise AttributeError(f"Target '{target_obj}' has no Value member '{self.member}'")

    def _resolve_target(self, scope: ExecutionScope) -> Any:
        """
        Resolves the target object expression, supporting type symbol lookup.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Any: Target object or symbol.
        """

        if isinstance(self.target, IdentifierExpr):
            return self.target.evaluate_symbol(scope=scope)

        return self.target.evaluate(scope=scope)

    def get_callable(self, scope: ExecutionScope) -> Any:
        """
        Retrieves a method callable from the target object.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Any: Method callable or StructMethod bound invocation.

        Raises:
            AttributeError: If callable member is not found.
        """

        target_obj: Any = self._resolve_target(scope=scope)

        # Typed collection helpers on dict values (before the Python attr
        # fallback, which would return untyped builtin dict methods)
        if isinstance(target_obj, ValueDict) and self.member in ("keys", "values"):
            return DictCollectionCallable(value_dict=target_obj, member=self.member)

        # Typed queue members (before the Python attr fallback, which would
        # expose untyped deque / list methods)
        if (
            isinstance(target_obj, (ValueQueue, ValuePriorityQueue))
            and self.member in ("push", "pop", "peek", "size")
        ):
            return QueueCallable(target=target_obj, member=self.member)

        # Typed set members (before the Python attr fallback, which would
        # expose untyped set methods)
        if (
            isinstance(target_obj, ValueSet)
            and self.member in ("add", "discard", "size")
        ):
            return SetCallable(target=target_obj, member=self.member)

        # Typed grid members (before the Python attr fallback, which would
        # expose untyped dict methods)
        if (
            isinstance(target_obj, ValueSpatialGrid)
            and self.member in (
                "at", "set_at", "has", "remove", "size", "coords",
                "neighbors", "occupied_neighbors",
            )
        ):
            return GridCallable(target=target_obj, member=self.member)

        # Typed tensor members (before the Python attr fallback)
        if (
            isinstance(target_obj, ValueTensor)
            and self.member in (
                "at", "set_at", "size", "shape",
                "sum", "min", "max", "transpose", "fill", "map", "mean",
                "argmax", "argmin", "matmul", "concat", "reshape",
                "flatten", "gather", "topk", "to_float", "to_int", "permute",
                "relu", "grad", "sigmoid", "gelu", "softmax", "exp",
                "conv1d", "conv2d", "max_pool", "min_pool", "avg_pool",
            )
        ):
            return TensorCallable(target=target_obj, member=self.member)

        # length() on arrays and strings (no typed fallback exists)
        if self.member == "length":

            is_array: bool = isinstance(target_obj, ValueArray)
            is_string: bool = (
                isinstance(target_obj, Value)
                and isinstance(target_obj.get(), str)
            )

            if is_array or is_string:
                return LengthCallable(target=target_obj)

        # Unwrap Value containers to expose the payload object
        if isinstance(target_obj, Value):
            target_obj = target_obj.get()

        # Check methods on struct definition first: StructInstance's
        # __getattr__ binds methods without the caller's globals, which
        # would break builtin resolution inside method bodies
        if hasattr(target_obj, "struct"):
            struct_def: Any = getattr(target_obj, "struct")

            if hasattr(struct_def, "get_method"):
                method: Any = struct_def.get_method(name=self.member)

                if method is not None:
                    return lambda *args, **kwargs: method.execute(
                        self_instance=target_obj,
                        args=list(args),
                        kwargs=kwargs,
                        globals_dict=scope.globals_dict
                    )

        # Check Python attributes / methods / staticmethods (e.g. DateTime.now)
        if hasattr(target_obj, self.member):
            return getattr(target_obj, self.member)

        # Virtual members: introspection and schema-constrained access
        virtual: Optional[Any] = resolve_virtual_callable(
            target_obj=target_obj,
            member=self.member,
        )

        if virtual is not None:
            return virtual

        raise AttributeError(f"Target '{target_obj}' has no callable member '{self.member}'")

    def assign_to(self, scope: ExecutionScope, value: Value) -> None:
        """
        Assigns a Value to the member of the resolved source object.

        Args:
            scope (ExecutionScope): Current execution scope.
            value (Value): Value to assign.

        Raises:
            TypeError: If the source object does not support member
                assignment.
        """

        target_obj: Any = self._resolve_target(scope=scope)

        # Unwrap Value containers to the raw object (e.g. an instance ref
        # stored in an attribute resolves to a ValueExternal wrapper)
        raw_target: Any = target_obj.get() if isinstance(target_obj, Value) else target_obj

        # Struct instances validate the value against the declared type
        if hasattr(raw_target, "attributes") and hasattr(raw_target, "set_attribute"):
            raw_target.set_attribute(name=self.member, value=value)

            return None

        raise TypeError(
            f"Cannot assign to member '{self.member}' of '{repr(raw_target)}'"
        )

    def ensure_materialized(self, scope: ExecutionScope) -> None:
        """
        Materializes the member on the resolved source instance.

        Args:
            scope (ExecutionScope): Current execution scope.
        """

        self.target.ensure_materialized(scope=scope)

        target_obj: Any = self._resolve_target(scope=scope)

        raw_target: Any = target_obj.get() if isinstance(target_obj, Value) else target_obj

        if hasattr(raw_target, "ensure_attribute"):
            raw_target.ensure_attribute(name=self.member)

    def __repr__(self) -> str:
        """
        Returns string representation of the MemberAccessExpr.

        Returns:
            str: Dotted member representation.
        """

        return f"{repr(self.target)}.{self.member}"


def _wrap_indexed_element(raw: Any, element_type: Type) -> Value:
    """
    Wraps an indexed element back into a typed Value.

    Args:
        raw (Any): The raw element extracted from the container.
        element_type (Type): Declared element type of the container.

    Returns:
        Value: The typed Value for the element.
    """

    # Instances and opaque payloads have no make_value constructor
    if element_type is TYPE_EXTERNAL or hasattr(raw, "entity_id"):
        return ValueExternal(payload=raw)

    return make_value(value_type=element_type, raw=raw)


class SliceOperand(Expression):
    """
    AST node representing a slice inside index brackets: 'start:stop'.

    Both bounds are optional and default to the dimension extent (0 and
    dim). Slices are only valid inside tensor index brackets; evaluating
    one anywhere else is an error.

    Attributes:
        start (Optional[Expression]): Inclusive lower bound expression,
            or None for 'from the beginning'.
        stop (Optional[Expression]): Exclusive upper bound expression,
            or None for 'through the end'.
    """

    def __init__(
        self,
        start: Optional[Expression],
        stop: Optional[Expression]
    ) -> None:
        """
        Initializes the SliceOperand instance.

        Args:
            start (Optional[Expression]): Inclusive lower bound, or None.
            stop (Optional[Expression]): Exclusive upper bound, or None.
        """

        self.start: Optional[Expression] = start
        self.stop: Optional[Expression] = stop

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Raises: slices are not first-class values.

        Args:
            scope (ExecutionScope): Current execution scope.

        Raises:
            TypeError: Always; slices are only valid inside index
                brackets.
        """

        raise TypeError(
            "Slices are only valid inside tensor index brackets"
        )

    def __repr__(self) -> str:
        """
        Returns the slice representation.

        Returns:
            str: 'start:stop' style representation.
        """

        start: str = repr(self.start) if self.start is not None else ""
        stop: str = repr(self.stop) if self.stop is not None else ""

        return f"{start}:{stop}"


class IndexExpr(Expression):
    """
    AST node representing indexing: target[index] or target[i0, i1, ...].

    Works on arrays (integer index, Python semantics including negatives),
    dictionaries (any hashable key), and tensors (multi-index resolved to
    a flat row-major offset, with optional 'start:stop' slices per
    dimension).

    Attributes:
        target (Expression): Expression resolving to the indexed container.
        index (Expression): Expression resolving to the index or key.
        extra_indices (list[Expression]): Additional comma-separated index
            expressions (multi-index tensors); empty for single indexing.
    """

    def __init__(
        self,
        target: Expression,
        index: Expression,
        extra_indices: Optional[list[Expression]] = None
    ) -> None:
        """
        Initializes the IndexExpr instance.

        Args:
            target (Expression): Expression resolving to the container.
            index (Expression): Expression resolving to the index or key.
            extra_indices (Optional[list[Expression]], optional): Further
                index expressions for multi-index tensors. Defaults to None.
        """

        self.target: Expression = target
        self.index: Expression = index
        self.extra_indices: list[Expression] = list(extra_indices or [])

    @property
    def indices(self) -> list[Expression]:
        """
        Returns every index expression in source order.

        Returns:
            list[Expression]: The index expressions.
        """

        return [self.index, *self.extra_indices]

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates target and index, returning the indexed element as a Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: The indexed element wrapped in its declared type.

        Raises:
            IndexError: If an array index is out of range.
            KeyError: If a dictionary key is missing.
            TypeError: If the target is not an indexable container or the
                index has an invalid kind.
        """

        target_obj: Any = self.target.evaluate(scope=scope)

        # Multi-index and sliced tensors resolve through the flat
        # row-major payload (slices are not first-class values, so the
        # dispatch must happen before evaluating the first index)
        if self._has_slices() or self.extra_indices:
            return self._index_tensor(target_obj=target_obj, scope=scope)

        index_val: Value = self.index.evaluate(scope=scope)

        if isinstance(target_obj, ValueArray):
            return self._index_array(array=target_obj, index=index_val)

        if isinstance(target_obj, ValueDict):
            return self._index_dict(mapping=target_obj, index=index_val)

        # Single flat index on a tensor: row-major payload access with a
        # typed element result
        if isinstance(target_obj, ValueTensor):
            return self._index_tensor_flat(target_obj=target_obj, index=index_val)

        # Raw Python containers inside external payloads
        raw_target: Any = (
            target_obj.get() if isinstance(target_obj, Value) else target_obj
        )

        if isinstance(raw_target, list):
            element: Any = self._index_raw_sequence(sequence=raw_target, index=index_val)

            if isinstance(element, Value):
                return element

            return ValueExternal(payload=element)

        if isinstance(raw_target, dict):
            element = self._index_raw_mapping(mapping=raw_target, index=index_val)

            if isinstance(element, Value):
                return element

            return ValueExternal(payload=element)

        raise TypeError(f"Cannot index into '{repr(target_obj)}'")

    def _index_tensor_flat(self, target_obj: ValueTensor, index: Value) -> Value:
        """
        Indexes a tensor with a single flat row-major index.

        Args:
            target_obj (ValueTensor): Tensor being indexed.
            index (Value): Integer index Value.

        Returns:
            Value: The typed element.

        Raises:
            TypeError: If the index is not an integer.
            IndexError: If the index is out of range.
        """

        raw_index: Any = index.get()

        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError(
                f"Tensor index must be an integer, got '{repr(raw_index)}'"
            )

        if not 0 <= raw_index < len(target_obj.cells):
            raise IndexError(
                f"Tensor index {raw_index} out of range for "
                f"{len(target_obj.cells)} element(s)"
            )

        return _wrap_indexed_element(
            raw=target_obj.cells[raw_index],
            element_type=target_obj.value_type.value_type,
        )

    def _index_tensor(self, target_obj: Any, scope: ExecutionScope) -> Value:
        """
        Indexes a tensor with one index or slice per dimension.

        Integer indices drop their dimension (element access); slices
        keep it (sub-tensor view copied row-major into a new tensor).

        Args:
            target_obj (Any): Evaluated target (expected ValueTensor).
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: The element wrapped in its declared type, or a new
                ValueTensor for sliced access.

        Raises:
            TypeError: If the target is not a tensor or an index is not
                an integer.
            IndexError: If any index is out of range.
            ValueError: If a slice produces an empty result.
        """

        if not isinstance(target_obj, ValueTensor):
            raise TypeError(
                f"Multi-index is only supported on tensors, got "
                f"'{repr(target_obj)}'"
            )

        if not self._has_slices():
            indices: list[int] = []

            for index_expr in self.indices:
                index_val: Value = index_expr.evaluate(scope=scope)
                raw_index: Any = index_val.get()

                if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                    raise TypeError(
                        f"Tensor index must be an integer, got '{repr(raw_index)}'"
                    )

                indices.append(raw_index)

            element: Any = target_obj.cells[
                target_obj.flat_index(indices=tuple(indices))
            ]

            return _wrap_indexed_element(
                raw=element,
                element_type=target_obj.value_type.value_type,
            )

        # Sliced access: gather the selected sub-block into a new tensor
        dims: list[int] = self._tensor_dims(tensor=target_obj)

        if len(self.indices) != len(dims):
            raise TypeError(
                f"Tensor access expects {len(dims)} index expression(s), "
                f"got {len(self.indices)}"
            )

        spec: list[tuple[bool, range]] = [
            (
                isinstance(index_expr, SliceOperand),
                self._index_range(index_expr=index_expr, dim=dim, scope=scope),
            )
            for index_expr, dim in zip(self.indices, dims)
        ]

        result_dims: list[int | str] = [
            len(dim_range) for is_slice, dim_range in spec if is_slice
        ]

        positions: list[int] = [
            target_obj.flat_index(indices=tuple(combo))
            for combo in product(*(dim_range for _, dim_range in spec))
        ]

        cells: list[Any] = [target_obj.cells[position] for position in positions]

        result_type: TypeTensor = TypeTensor(
            value_type=target_obj.value_type.value_type,
            dims=result_dims,
        )

        result: ValueTensor = ValueTensor(value_type=result_type, cells=cells)

        # Autograd: record the read so backward scatters into the source
        maybe_record_slice(
            target=target_obj,
            spec=[
                (is_slice, dim_range.start, dim_range.stop)
                for is_slice, dim_range in spec
            ],
            indices=[
                dim_range.start
                for is_slice, dim_range in spec
                if not is_slice
            ],
            result=result,
            tape=scope.tape,
        )

        return result

    def _has_slices(self) -> bool:
        """
        Returns whether any index in the bracket is a slice.

        Returns:
            bool: True if at least one index operand is a SliceOperand.
        """

        return any(
            isinstance(index_expr, SliceOperand)
            for index_expr in self.indices
        )

    @staticmethod
    def _tensor_dims(tensor: ValueTensor) -> list[int]:
        """
        Returns the concrete per-axis extents of a tensor.

        Args:
            tensor (ValueTensor): Tensor to inspect.

        Returns:
            list[int]: Extents in axis order.

        Raises:
            TypeError: If the tensor has unbound symbolic dimensions.
        """

        if tensor.concrete_dims is not None:
            return list(tensor.concrete_dims)

        if tensor.value_type.is_symbolic:
            raise TypeError(
                "Cannot index a tensor with unbound symbolic dimensions"
            )

        return [dim for dim in tensor.value_type.dims if isinstance(dim, int)]

    def _index_range(
        self,
        index_expr: Expression,
        dim: int,
        scope: ExecutionScope,
    ) -> range:
        """
        Resolves one index operand against a dimension extent.

        Args:
            index_expr (Expression): Integer index or SliceOperand.
            dim (int): Extent of the dimension being indexed.
            scope (ExecutionScope): Current execution scope.

        Returns:
            range: Selected positions along the dimension (a single
                position for integer indices).

        Raises:
            TypeError: If a bound is not an integer.
            IndexError: If an index is out of range.
            ValueError: If a slice is empty or reversed.
        """

        if isinstance(index_expr, SliceOperand):
            start: int = 0

            if index_expr.start is not None:
                start = self._index_int(
                    expression=index_expr.start, scope=scope
                )

            stop: int = dim

            if index_expr.stop is not None:
                stop = self._index_int(
                    expression=index_expr.stop, scope=scope
                )

            if not 0 <= start <= stop <= dim:
                raise ValueError(
                    f"Invalid slice {start}:{stop} for dimension of size "
                    f"{dim}"
                )

            if start == stop:
                raise ValueError(
                    f"Slice {start}:{stop} produces an empty result"
                )

            return range(start, stop)

        index_val: Value = index_expr.evaluate(scope=scope)
        raw_index: Any = index_val.get()

        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError(
                f"Tensor index must be an integer, got '{repr(raw_index)}'"
            )

        if not 0 <= raw_index < dim:
            raise IndexError(
                f"Tensor index {raw_index} out of range for dimension of "
                f"size {dim}"
            )

        return range(raw_index, raw_index + 1)

    @staticmethod
    def _index_int(expression: Expression, scope: ExecutionScope) -> int:
        """
        Evaluates an expression that must be an integer.

        Args:
            expression (Expression): Bound expression of a slice.
            scope (ExecutionScope): Current execution scope.

        Returns:
            int: The evaluated integer bound.

        Raises:
            TypeError: If the bound is not an integer.
        """

        bound: Value = expression.evaluate(scope=scope)
        raw_bound: Any = bound.get()

        if isinstance(raw_bound, bool) or not isinstance(raw_bound, int):
            raise TypeError(
                f"Slice bounds must be integers, got '{repr(raw_bound)}'"
            )

        return raw_bound

    def _index_array(self, array: ValueArray, index: Value) -> Value:
        """
        Indexes a typed array Value.

        Args:
            array (ValueArray): The array to index.
            index (Value): The index Value (must be an integer).

        Returns:
            Value: The wrapped element.

        Raises:
            TypeError: If the index is not an integer.
            IndexError: If the index is out of range.
        """

        raw_index: Any = index.get()

        if not isinstance(raw_index, int):
            raise TypeError(
                f"Array index must be an integer, got '{repr(raw_index)}'"
            )

        if not -len(array.elements) <= raw_index < len(array.elements):
            raise IndexError(
                f"Array index {raw_index} out of range for "
                f"{len(array.elements)} element(s)"
            )

        element: Any = array.elements[raw_index]

        return _wrap_indexed_element(
            raw=element,
            element_type=array.value_type.value_type,
        )

    def _index_dict(self, mapping: ValueDict, index: Value) -> Value:
        """
        Indexes a typed dictionary Value.

        Args:
            mapping (ValueDict): The dictionary to index.
            index (Value): The key Value.

        Returns:
            Value: The wrapped value.

        Raises:
            KeyError: If the key is missing.
        """

        raw_key: Any = index.get()

        if raw_key not in mapping.entries:
            raise KeyError(f"Key '{repr(raw_key)}' not found in dictionary")

        element: Any = mapping.entries[raw_key]

        return _wrap_indexed_element(
            raw=element,
            element_type=mapping.value_type.value_type,
        )

    @staticmethod
    def _index_raw_sequence(sequence: list[Any], index: Value) -> Any:
        """
        Indexes a raw Python list with bounds checking.

        Args:
            sequence (list[Any]): The list to index.
            index (Value): The index Value.

        Returns:
            Any: The element.

        Raises:
            TypeError: If the index is not an integer.
            IndexError: If the index is out of range.
        """

        raw_index: Any = index.get()

        if not isinstance(raw_index, int):
            raise TypeError(
                f"Array index must be an integer, got '{repr(raw_index)}'"
            )

        if not -len(sequence) <= raw_index < len(sequence):
            raise IndexError(
                f"Array index {raw_index} out of range for "
                f"{len(sequence)} element(s)"
            )

        return sequence[raw_index]

    @staticmethod
    def _index_raw_mapping(mapping: dict[Any, Any], index: Value) -> Any:
        """
        Indexes a raw Python dictionary.

        Args:
            mapping (dict[Any, Any]): The dictionary to index.
            index (Value): The key Value.

        Returns:
            Any: The value.

        Raises:
            KeyError: If the key is missing.
        """

        raw_key: Any = index.get()

        if raw_key not in mapping:
            raise KeyError(f"Key '{repr(raw_key)}' not found in dictionary")

        return mapping[raw_key]

    def assign_to(self, scope: ExecutionScope, value: Value) -> None:
        """
        Assigns a Value to the indexed slot of the resolved container.

        Supports typed arrays (element type validated, negative indices
        allowed), typed dictionaries, and raw Python lists/dicts inside
        opaque payloads.

        Args:
            scope (ExecutionScope): Current execution scope.
            value (Value): Value to assign.

        Raises:
            IndexError: If an array index is out of range.
            KeyError: If a dictionary key is missing.
            TypeError: If the container or index is invalid.
        """

        # Materialize lazy defaults along the access path before mutating
        self.target.ensure_materialized(scope=scope)

        container: Any = self.target.evaluate(scope=scope)

        # Multi-index and sliced tensors assign through the flat
        # row-major payload
        if self._has_slices() or self.extra_indices:
            self._assign_tensor(container=container, scope=scope, value=value)

            return None

        index_val: Value = self.index.evaluate(scope=scope)

        # Single flat index assignment on a tensor: row-major payload
        # access with element type validation
        if isinstance(container, ValueTensor):
            self._assign_tensor_flat(container=container, index=index_val, value=value)

            return None

        if isinstance(container, ValueArray):
            self._assign_array(array=container, index=index_val, value=value)

            return None

        if isinstance(container, ValueDict):
            self._assign_dict(mapping=container, index=index_val, value=value)

            return None

        # Raw Python containers inside external payloads
        raw_container: Any = (
            container.get() if isinstance(container, Value) else container
        )

        if isinstance(raw_container, list):
            normalized: int = self._raw_sequence_index(
                sequence=raw_container,
                index=index_val,
            )

            raw_container[normalized] = value.get()

            return None

        if isinstance(raw_container, dict):
            raw_container[index_val.get()] = value.get()

            return None

        raise TypeError(f"Cannot assign into '{repr(container)}'")

    def _assign_tensor_flat(
        self,
        container: ValueTensor,
        index: Value,
        value: Value,
    ) -> None:
        """
        Assigns into a tensor through a single flat row-major index.

        Args:
            container (ValueTensor): Tensor being written.
            index (Value): Integer index Value.
            value (Value): The new element Value.

        Raises:
            TypeError: If the index is not an integer or the element
                violates the declared element type.
            IndexError: If the index is out of range.
        """

        raw_index: Any = index.get()

        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError(
                f"Tensor index must be an integer, got '{repr(raw_index)}'"
            )

        if not 0 <= raw_index < len(container.cells):
            raise IndexError(
                f"Tensor index {raw_index} out of range for "
                f"{len(container.cells)} element(s)"
            )

        container.value_type.value_type.check_value(value=value.get())

        container.cells[raw_index] = value.get()
        container.bump_version()

    def _assign_tensor(
        self,
        container: Any,
        scope: ExecutionScope,
        value: Value,
    ) -> None:
        """
        Assigns into a tensor through one index or slice per dimension.

        Integer indices write a single element; slices write a block —
        the payload must be a (possibly nested) array literal matching
        the sliced shape in row-major order, or a scalar broadcast to
        every selected cell.

        Args:
            container (Any): Evaluated target (expected ValueTensor).
            scope (ExecutionScope): Current execution scope.
            value (Value): The new element Value, block payload, or
                broadcast scalar.

        Raises:
            TypeError: If the target is not a tensor or an element
                violates the declared element type.
            IndexError: If any index is out of range.
            ValueError: If a slice produces an empty result.
        """

        if not isinstance(container, ValueTensor):
            raise TypeError(
                f"Multi-index assignment is only supported on tensors, got "
                f"'{repr(container)}'"
            )

        if not self._has_slices():
            indices: list[int] = []

            for index_expr in self.indices:
                index_val: Value = index_expr.evaluate(scope=scope)
                raw_index: Any = index_val.get()

                if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                    raise TypeError(
                        f"Tensor index must be an integer, got '{repr(raw_index)}'"
                    )

                indices.append(raw_index)

            container.value_type.value_type.check_value(value=value.get())

            container.cells[container.flat_index(indices=tuple(indices))] = value.get()

            return None

        # Sliced assignment: validate the payload against the block shape
        dims: list[int] = self._tensor_dims(tensor=container)

        if len(self.indices) != len(dims):
            raise TypeError(
                f"Tensor assignment expects {len(dims)} index expression(s), "
                f"got {len(self.indices)}"
            )

        spec: list[tuple[bool, range]] = [
            (
                isinstance(index_expr, SliceOperand),
                self._index_range(index_expr=index_expr, dim=dim, scope=scope),
            )
            for index_expr, dim in zip(self.indices, dims)
        ]

        count: int = 1

        for is_slice, dim_range in spec:

            if is_slice:
                count *= len(dim_range)

        positions: list[int] = [
            container.flat_index(indices=tuple(combo))
            for combo in product(*(dim_range for _, dim_range in spec))
        ]

        element_type: Type = container.value_type.value_type
        raw_value: Any = value.get()

        if isinstance(raw_value, list):
            payload: list[Any] = flatten_nested_lists(raw=raw_value)

            if len(payload) != count:
                raise TypeError(
                    f"Slice assignment expects {count} element(s), got "
                    f"{len(payload)}"
                )

        else:
            element_type.check_value(value=raw_value)

            payload = [raw_value] * count

        for position, element in zip(positions, payload):

            element_type.check_value(value=element)

            container.cells[position] = element

        container.bump_version()

    def _assign_array(
        self,
        array: ValueArray,
        index: Value,
        value: Value,
    ) -> None:
        """
        Assigns into a typed array after validating index and element type.

        Args:
            array (ValueArray): The array to update.
            index (Value): The index Value (must be an integer).
            value (Value): The new element Value.

        Raises:
            TypeError: If the index is not an integer or the element
                violates the array element type.
            IndexError: If the index is out of range.
        """

        raw_index: Any = index.get()

        if not isinstance(raw_index, int):
            raise TypeError(f"Array index must be an integer, got '{repr(raw_index)}'")

        if not -len(array.elements) <= raw_index < len(array.elements):
            raise IndexError(
                f"Array index {raw_index} out of range for "
                f"{len(array.elements)} element(s)"
            )

        element_type: Type = array.value_type.value_type

        # Validate the raw payload against the declared element type
        element_type.check_value(value=value.get())

        array.elements[raw_index] = value.get()

    def _assign_dict(
        self,
        mapping: ValueDict,
        index: Value,
        value: Value,
    ) -> None:
        """
        Assigns into a typed dictionary after validating value type.

        Args:
            mapping (ValueDict): The dictionary to update.
            index (Value): The key Value.
            value (Value): The new value Value.

        Raises:
            KeyError: If the key is missing.
            TypeError: If the value violates the declared value type.
        """

        raw_key: Any = index.get()

        # Validate both key and value against the declared types; new keys
        # are inserted (Python dict semantics)
        mapping.value_type.key_type.check_value(value=raw_key)

        value_type: Type = mapping.value_type.value_type

        # Validate the raw payload against the declared value type
        value_type.check_value(value=value.get())

        mapping.entries[raw_key] = value.get()

    def _raw_sequence_index(self, sequence: list[Any], index: Value) -> int:
        """
        Normalizes an index for a raw Python sequence assignment.

        Args:
            sequence (list[Any]): The target sequence.
            index (Value): The index Value.

        Returns:
            int: Normalized (non-negative) index.

        Raises:
            TypeError: If the index is not an integer.
            IndexError: If the index is out of range.
        """

        raw_index: Any = index.get()

        if not isinstance(raw_index, int):
            raise TypeError(f"Sequence index must be an integer, got '{repr(raw_index)}'")

        if not -len(sequence) <= raw_index < len(sequence):
            raise IndexError(
                f"Sequence index {raw_index} out of range for "
                f"{len(sequence)} element(s)"
            )

        return raw_index % len(sequence)

    def __repr__(self) -> str:
        """
        Returns the indexed target representation.

        Returns:
            str: Bracketed index representation.
        """

        return f"{repr(self.target)}[{', '.join(repr(index_expr) for index_expr in self.indices)}]"


class CallExpr(Expression):
    """
    AST node representing a method, constructor, or function call returning a Value.

    Attributes:
        callee (Expression): Expression resolving to the callable.
        args (list[Expression]): Positional argument expressions.
        kwargs (dict[str, Expression]): Keyword argument expressions.
    """

    def __init__(
        self,
        callee: Expression,
        args: Optional[list[Expression]] = None,
        kwargs: Optional[dict[str, Expression]] = None
    ) -> None:
        """
        Initializes the CallExpr instance.

        Args:
            callee (Expression): Callee expression.
            args (Optional[list[Expression]], optional): Positional args. Defaults to None.
            kwargs (Optional[dict[str, Expression]], optional): Keyword args. Defaults to None.
        """

        self.callee: Expression = callee
        self.args: list[Expression] = args if args is not None else []
        self.kwargs: dict[str, Expression] = kwargs if kwargs is not None else {}

    def _resolve_callable(self, scope: ExecutionScope) -> Any:
        """
        Resolves the callee expression into a callable object.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Any: Callable object or StructMethod.
        """

        if isinstance(self.callee, MemberAccessExpr):
            return self.callee.get_callable(scope=scope)

        if isinstance(self.callee, IdentifierExpr):
            symbol: Any = self.callee.evaluate_symbol(scope=scope)

            # Check if symbol is a StructMethod or standalone Function
            if hasattr(symbol, "execute"):
                return lambda *args, **kwargs: symbol.execute(
                    self_instance=scope.self_instance,
                    args=list(args),
                    kwargs=kwargs,
                    globals_dict=scope.globals_dict
                )

            return symbol

        return self.callee.evaluate(scope=scope)


    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates the callee and arguments, invoking the callable and returning a Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Returned Value result.

        Raises:
            TypeError: If result is not a Value or callee is not callable.
        """

        callable_obj: Any = self._resolve_callable(scope=scope)

        # Evaluate arguments to Value objects
        eval_args: list[Value] = [arg.evaluate(scope=scope) for arg in self.args]
        eval_kwargs: dict[str, Value] = {
            k: v.evaluate(scope=scope) for k, v in self.kwargs.items()
        }

        # Scope-aware natives (e.g. set_loop_limit, backward) receive the
        # live scope alongside the evaluated argument Values
        if isinstance(callable_obj, ScopeAwareFunction):
            native_res: Any = callable_obj.invoke(
                scope=scope,
                args=eval_args,
                kwargs=eval_kwargs,
            )

            if isinstance(native_res, Value):
                return native_res

            if native_res is None:
                return ValueExternal(payload=None)

            return LiteralExpr(value=native_res).evaluate(scope=scope)

        # Invoke callable
        if callable(callable_obj):
            # Check if callable is a native Python class or function (e.g. DateTime)
            is_type: bool = isinstance(callable_obj, type)
            is_home_type: bool = (
                getattr(callable_obj, "__module__", None) or ""
            ).startswith("src.core.types.home_types")

            if is_type or is_home_type:
                raw_args: list[Any] = [a.get() for a in eval_args]
                raw_kwargs: dict[str, Any] = {k: v.get() for k, v in eval_kwargs.items()}
                res: Any = callable_obj(*raw_args, **raw_kwargs)
            else:
                res = callable_obj(*eval_args, **eval_kwargs)

                # Autograd: record differentiable tensor member calls
                if isinstance(callable_obj, TensorCallable):
                    maybe_record_member(
                        member=callable_obj.member,
                        receiver=callable_obj.target,
                        args=eval_args,
                        result=res,
                        tape=scope.tape,
                    )


            if isinstance(res, Value):
                return res

            # Wrap primitive results (e.g. QuerySet.count()) into typed Values
            if isinstance(res, (bool, int, float, str)):
                return LiteralExpr(value=res).evaluate(scope=scope)

            # Native callables returning None (e.g. print) yield an empty opaque Value
            if res is None:
                return ValueExternal(payload=None)

            # Raw lists (e.g. QuerySet.to_list()) stay iterable in for-in
            if isinstance(res, list):
                return ValueExternal(payload=res)

            # Wrap optimizer instances (autograd) as opaque Values so
            # opt.step(model) resolves through the attribute fallback
            if isinstance(res, Optimizer):
                return ValueExternal(payload=res)

            # Wrap query-like results (duck-typed QuerySet API) as opaque Values
            if hasattr(res, "to_list") and hasattr(res, "where"):
                return ValueExternal(payload=res)

            # Wrap struct-instance results (e.g. constructors) as opaque Values
            if hasattr(res, "struct") and hasattr(res, "attributes"):
                return ValueExternal(payload=res)

            if isinstance(res, DateTime):
                return ValueDateTime(value=res)

            if isinstance(res, Color):
                return ValueColor(value=res)

            raise TypeError(
                f"Call to '{repr(self.callee)}' returned '{type(res).__name__}' "
                f"instead of expected Value instance"
            )

        raise TypeError(f"Target '{callable_obj}' is not callable")

    def __repr__(self) -> str:
        """
        Returns string representation of the CallExpr.

        Returns:
            str: Function call representation.
        """

        arg_strs: list[str] = [repr(a) for a in self.args]
        kw_strs: list[str] = [f"{k}={repr(v)}" for k, v in self.kwargs.items()]
        combined_args: str = ", ".join(arg_strs + kw_strs)

        return f"{repr(self.callee)}({combined_args})"


class LambdaPredicate:
    """
    Callable predicate produced by LambdaExpr evaluation.

    Binds the candidate payload into a child scope under the lambda parameter
    name, evaluates the body expression, and returns its boolean result.

    Attributes:
        param_name (str): Candidate parameter identifier.
        body (Expression): Predicate body expression.
        scope (ExecutionScope): Scope the lambda was evaluated in.
    """

    def __init__(
        self,
        param_name: str,
        body: Expression,
        scope: ExecutionScope
    ) -> None:
        """
        Initializes the LambdaPredicate instance.

        Args:
            param_name (str): Candidate parameter identifier.
            body (Expression): Predicate body expression.
            scope (ExecutionScope): Defining execution scope.
        """

        self.param_name: str = param_name
        self.body: Expression = body
        self.scope: ExecutionScope = scope

    def __call__(self, candidate: Any) -> Any:
        """
        Evaluates the lambda body with the candidate bound to the parameter.

        Args:
            candidate (Any): Candidate payload (e.g. a StructInstance or a
                tensor element).

        Returns:
            Any: The raw body result (filters treat it as a boolean;
                value-wise callers use it directly).
        """

        child_scope: ExecutionScope = self.scope.create_child_scope()

        # Struct candidates bind opaquely; basic scalars bind with their
        # runtime type so arithmetic inside the body works
        if isinstance(candidate, (bool, int, float, str)):
            bound: Value = LiteralExpr(value=candidate).evaluate(scope=child_scope)

        else:
            bound = ValueExternal(payload=candidate)

        child_scope.globals_dict[self.param_name] = bound

        result: Value = self.body.evaluate(scope=child_scope)

        return result.get()

    def __repr__(self) -> str:
        """
        Returns string representation of the predicate.

        Returns:
            str: Predicate representation.
        """

        return f"LambdaPredicate({self.param_name}: {repr(self.body)})"


class LambdaExpr(Expression):
    """
    AST node representing a lambda expression '(param: ParamType) -> body'.

    Evaluates to an opaque ValueExternal wrapping a LambdaPredicate callable,
    suitable for query predicates such as '.where((h: Human) -> h.age > 18)'.

    Attributes:
        param_name (str): Candidate parameter identifier.
        param_type_name (Optional[str]): Raw parameter type annotation.
        body (Expression): Predicate body expression.
    """

    def __init__(
        self,
        param_name: str,
        body: Expression,
        param_type_name: Optional[str] = None
    ) -> None:
        """
        Initializes the LambdaExpr instance.

        Args:
            param_name (str): Candidate parameter identifier.
            body (Expression): Predicate body expression.
            param_type_name (Optional[str], optional): Raw parameter annotation. Defaults to None.
        """

        self.param_name: str = param_name
        self.body: Expression = body
        self.param_type_name: Optional[str] = param_type_name

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Builds the predicate callable bound to the defining scope.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: ValueExternal wrapping the LambdaPredicate callable.
        """

        predicate: LambdaPredicate = LambdaPredicate(
            param_name=self.param_name,
            body=self.body,
            scope=scope
        )

        return ValueExternal(payload=predicate)

    def __repr__(self) -> str:
        """
        Returns string representation of the lambda.

        Returns:
            str: Lambda representation.
        """

        type_str: str = f"{self.param_type_name}" if self.param_type_name else ""

        return f"({self.param_name}: {type_str}) -> {repr(self.body)}"


class ArrayLiteralExpr(Expression):
    """
    AST node representing an array literal (e.g. ['a', 'b']).

    Attributes:
        elements (list[Expression]): Element expressions in declaration order.
        declared_type (Optional[TypeArray]): Declared element type, when known
            from the surrounding declaration. Required for empty literals.
    """

    def __init__(
        self,
        elements: Optional[list[Expression]] = None,
        declared_type: Optional[TypeArray] = None
    ) -> None:
        """
        Initializes the ArrayLiteralExpr instance.

        Args:
            elements (Optional[list[Expression]], optional): Element expressions.
                Defaults to None (empty array).
            declared_type (Optional[TypeArray], optional): Declared array type.
                Defaults to None (type inferred from elements).
        """

        self.elements: list[Expression] = elements if elements is not None else []
        self.declared_type: Optional[TypeArray] = declared_type

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates all elements and builds the typed array Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Typed ValueArray result.

        Raises:
            ValueError: If the element type cannot be determined.
        """

        evaluated: list[Value] = [
            element.evaluate(scope=scope) for element in self.elements
        ]

        # Resolve the array type: declared type first, else infer from elements
        if self.declared_type is not None:
            array_type: TypeArray = self.declared_type

        elif evaluated:
            array_type = TypeArray(value_type=evaluated[0].value_type)

        else:
            raise ValueError("Cannot infer the type of an empty array literal")

        raw_elements: list[Any] = [element.get() for element in evaluated]

        return ValueArray(value_type=array_type, elements=raw_elements)

    def __repr__(self) -> str:
        """
        Returns string representation of the ArrayLiteralExpr.

        Returns:
            str: Bracketed comma separated element representation.
        """

        element_strs: list[str] = [repr(element) for element in self.elements]

        return f"[{', '.join(element_strs)}]"


class DictLiteralExpr(Expression):
    """
    AST node representing a dictionary literal (e.g. {'key': 'value'}).

    Also serves as the empty-brace form of queue declarations: when the
    surrounding declaration type is a TypeQueue or TypePriorityQueue, an
    empty brace literal evaluates to the matching empty queue.

    Attributes:
        entries (list[tuple[Expression, Expression]]): Key / value expression
            pairs in declaration order.
        declared_type (Optional[Type]): Declared type (TypeDict, or a
            TypeQueue / TypePriorityQueue for empty-brace queue forms).
    """

    def __init__(
        self,
        entries: Optional[list[tuple[Expression, Expression]]] = None,
        declared_type: Optional[Type] = None
    ) -> None:
        """
        Initializes the DictLiteralExpr instance.

        Args:
            entries (Optional[list[tuple[Expression, Expression]]], optional):
                Key / value expression pairs. Defaults to None (empty dict).
            declared_type (Optional[Type], optional): Declared type (TypeDict
                or an empty-brace queue form). Defaults to None (dict types
                inferred from entries).
        """

        self.entries: list[tuple[Expression, Expression]] = (
            entries if entries is not None else []
        )
        self.declared_type: Optional[Type] = declared_type

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates all entries and builds the typed dictionary Value, or the
        empty queue Value when declared as a queue type.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Typed ValueDict / ValueQueue / ValuePriorityQueue result.

        Raises:
            ValueError: If the key / value types cannot be determined.
        """

        # Empty braces under a queue or sparse grid declaration produce the
        # empty container
        if isinstance(
            self.declared_type, (TypeQueue, TypePriorityQueue, TypeSpatialGrid)
        ):
            return make_value(value_type=self.declared_type, raw=[])

        evaluated_entries: list[tuple[Value, Value]] = [
            (key.evaluate(scope=scope), value.evaluate(scope=scope))
            for key, value in self.entries
        ]

        # Resolve the dict type: declared type first, else infer from entries
        if isinstance(self.declared_type, TypeDict):
            dict_type: TypeDict = self.declared_type

        elif evaluated_entries:
            dict_type = TypeDict(
                key_type=evaluated_entries[0][0].value_type,
                value_type=evaluated_entries[0][1].value_type
            )

        else:
            raise ValueError("Cannot infer the type of an empty dict literal")

        raw_entries: dict[Any, Any] = {
            key.get(): value.get() for key, value in evaluated_entries
        }

        return ValueDict(value_type=dict_type, entries=raw_entries)

    def __repr__(self) -> str:
        """
        Returns string representation of the DictLiteralExpr.

        Returns:
            str: Braced comma separated key / value representation.
        """

        entry_strs: list[str] = [
            f"{repr(key)}: {repr(value)}" for key, value in self.entries
        ]

        return f"{{{', '.join(entry_strs)}}}"


class SetLiteralExpr(Expression):
    """
    AST node representing a set literal (e.g. {'a', 'b'}).  

    Attributes:
        elements (list[Expression]): Element expressions in source order.
        declared_type (Optional[TypeSet]): Declared element type, when known
            from the surrounding declaration (empty literals require it).
    """

    def __init__(
        self,
        elements: Optional[list[Expression]] = None,
        declared_type: Optional[TypeSet] = None
    ) -> None:
        """
        Initializes the SetLiteralExpr instance.

        Args:
            elements (Optional[list[Expression]], optional): Element
                expressions. Defaults to None (empty set).
            declared_type (Optional[TypeSet], optional): Declared set type.
                Defaults to None (type inferred from elements).
        """

        self.elements: list[Expression] = elements if elements is not None else []
        self.declared_type: Optional[TypeSet] = declared_type

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates all elements and builds the typed set Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Typed ValueSet result.

        Raises:
            ValueError: If the element type cannot be determined.
        """

        evaluated: list[Value] = [
            element.evaluate(scope=scope) for element in self.elements
        ]

        # Resolve the set type: declared type first, else infer from elements
        if self.declared_type is not None:
            set_type: TypeSet = self.declared_type

        elif evaluated:
            set_type = TypeSet(value_type=evaluated[0].value_type)

        else:
            raise ValueError("Cannot infer the type of an empty set literal")

        raw_elements: set[Any] = {element.get() for element in evaluated}

        return ValueSet(value_type=set_type, elements=raw_elements)

    def __repr__(self) -> str:
        """
        Returns string representation of the SetLiteralExpr.

        Returns:
            str: Braced comma separated element representation.
        """

        return f"{{{', '.join(repr(element) for element in self.elements)}}}"
