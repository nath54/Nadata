"""
Expression AST nodes and evaluation engine for Nadata method bodies.

Provides AST representations for literals, identifiers, unary operations,
binary operations, member accesses, and function/method calls operating on Value objects.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.types.types import (
    Int32,
    TypeStr,
    Float32,
    TypeBool,
    TypeFloat,
)
from src.core.values.values import (
    Value,
    ValueStr,
    ValueInt,
    ValueBool,
    ValueColor,
    ValueFloat,
    ValueDateTime,
)
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime
from src.core.expressions.scope import ExecutionScope



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
            self.value = ValueInt(value_type=Int32, value=value)
        elif isinstance(value, float):
            self.value = ValueFloat(value_type=Float32, value=value)
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

        # Check if left value has custom operator method (e.g. DateTime - DateTime)
        raw_left: Any = left.get()
        raw_right: Any = right.get()

        if hasattr(raw_left, "__sub__") and op == "-":
            custom_diff: Any = raw_left - raw_right

            if isinstance(custom_diff, Value):
                return custom_diff

            if isinstance(custom_diff, DateTime):
                return ValueDateTime(value=custom_diff)

        if hasattr(raw_left, "__add__") and op == "+":
            custom_sum: Any = raw_left + raw_right

            if isinstance(custom_sum, Value):
                return custom_sum

            if isinstance(custom_sum, DateTime):
                return ValueDateTime(value=custom_sum)

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

        cmp_table: dict[str, bool] = {
            "==": bool(raw_left == raw_right),
            "!=": bool(raw_left != raw_right),
            "<=": bool(raw_left <= raw_right),
            "<": bool(raw_left < raw_right),
            ">=": bool(raw_left >= raw_right),
            ">": bool(raw_left > raw_right),
        }

        return ValueBool(value_type=TypeBool(), value=cmp_table[op])

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
            return arith_res

        # 2. Attempt comparison evaluation
        comp_res: Optional[ValueBool] = self._eval_comparison(left_val, self.op, right_val)

        if comp_res is not None:
            return comp_res

        # 3. Logical operations
        if self.op in ("and", "&&"):
            bool_and: bool = bool(left_val.get()) and bool(right_val.get())
            return ValueBool(value_type=TypeBool(), value=bool_and)

        if self.op in ("or", "||"):
            bool_or: bool = bool(left_val.get()) or bool(right_val.get())
            return ValueBool(value_type=TypeBool(), value=bool_or)

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

        # Check dictionary-backed attributes on struct instances
        if hasattr(target_obj, "attributes"):
            attrs: dict[str, Any] = getattr(target_obj, "attributes")

            if self.member in attrs:
                attr_val: Any = attrs[self.member]

                if isinstance(attr_val, Value):
                    return attr_val

        # Check native Python object attributes
        if hasattr(target_obj, self.member):
            native_val: Any = getattr(target_obj, self.member)

            if isinstance(native_val, Value):
                return native_val

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

        # Check Python attributes / methods / staticmethods (e.g. DateTime.now)
        if hasattr(target_obj, self.member):
            return getattr(target_obj, self.member)

        # Check methods on struct definition
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

        raise AttributeError(f"Target '{target_obj}' has no callable member '{self.member}'")

    def __repr__(self) -> str:
        """
        Returns string representation of the MemberAccessExpr.

        Returns:
            str: Dotted member representation.
        """

        return f"{repr(self.target)}.{self.member}"


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

            # Check if symbol is a StructMethod (implicit self method call)
            if hasattr(symbol, "execute") and scope.self_instance is not None:
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

        # Invoke callable
        if callable(callable_obj):
            # Check if callable is a native Python class or function (e.g. DateTime)
            is_type: bool = isinstance(callable_obj, type)
            is_home_type: bool = getattr(callable_obj, "__module__", "").startswith(
                "src.core.types.home_types"
            )

            if is_type or is_home_type:
                raw_args: list[Any] = [a.get() for a in eval_args]
                raw_kwargs: dict[str, Any] = {k: v.get() for k, v in eval_kwargs.items()}
                res: Any = callable_obj(*raw_args, **raw_kwargs)
            else:
                res = callable_obj(*eval_args, **eval_kwargs)


            if isinstance(res, Value):
                return res

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
