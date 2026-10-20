"""
Expression AST nodes and evaluation engine for Nadata method bodies.

Provides AST representations for literals, identifiers, unary operations,
binary operations, member accesses, and function/method calls operating on Value objects.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.types.types import (
    Int64,
    TypeStr,
    Float32,
    TypeBool,
    TypeInt,
    TypeFloat,
    TypeArray,
    TypeDict,
    Float64,
    UInt64,
)
from src.core.values.values import (
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
)
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime
from src.core.expressions.builtins import (
    _is_struct_instance,
    _struct_from_object,
    resolve_virtual_callable,
)
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

        # Unwrap Value containers to expose the payload object
        if isinstance(target_obj, Value):
            target_obj = target_obj.get()

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

        # Virtual members: introspection and schema-constrained access
        virtual: Optional[Any] = resolve_virtual_callable(
            target_obj=target_obj,
            member=self.member,
        )

        if virtual is not None:
            return virtual

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

            # Wrap primitive results (e.g. QuerySet.count()) into typed Values
            if isinstance(res, (bool, int, float, str)):
                return LiteralExpr(value=res).evaluate(scope=scope)

            # Native callables returning None (e.g. print) yield an empty opaque Value
            if res is None:
                return ValueExternal(payload=None)

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

    def __call__(self, candidate: Any) -> bool:
        """
        Evaluates the lambda body with the candidate bound to the parameter.

        Args:
            candidate (Any): Candidate payload (e.g. a StructInstance).

        Returns:
            bool: Boolean result of the body expression.
        """

        child_scope: ExecutionScope = self.scope.create_child_scope()
        child_scope.globals_dict[self.param_name] = ValueExternal(payload=candidate)

        result: Value = self.body.evaluate(scope=child_scope)

        return bool(result.get())

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

    Attributes:
        entries (list[tuple[Expression, Expression]]): Key / value expression
            pairs in declaration order.
        declared_type (Optional[TypeDict]): Declared key / value types, when
            known from the surrounding declaration (empty literals require it).
    """

    def __init__(
        self,
        entries: Optional[list[tuple[Expression, Expression]]] = None,
        declared_type: Optional[TypeDict] = None
    ) -> None:
        """
        Initializes the DictLiteralExpr instance.

        Args:
            entries (Optional[list[tuple[Expression, Expression]]], optional):
                Key / value expression pairs. Defaults to None (empty dict).
            declared_type (Optional[TypeDict], optional): Declared dict type.
                Defaults to None (types inferred from entries).
        """

        self.entries: list[tuple[Expression, Expression]] = (
            entries if entries is not None else []
        )
        self.declared_type: Optional[TypeDict] = declared_type

    def evaluate(self, scope: ExecutionScope) -> Value:
        """
        Evaluates all entries and builds the typed dictionary Value.

        Args:
            scope (ExecutionScope): Current execution scope.

        Returns:
            Value: Typed ValueDict result.

        Raises:
            ValueError: If the key / value types cannot be determined.
        """

        evaluated_entries: list[tuple[Value, Value]] = [
            (key.evaluate(scope=scope), value.evaluate(scope=scope))
            for key, value in self.entries
        ]

        # Resolve the dict type: declared type first, else infer from entries
        if self.declared_type is not None:
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
