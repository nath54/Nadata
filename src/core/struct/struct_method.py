"""
Struct method definition for Nadata schemas.

Represents a computed property or method declared within a struct,
including its name, parameter types, default parameter expressions,
return type, body expression, and parsed AST body block.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.types.types import Type
from src.core.values.values import Value
from src.core.expressions.scope import ExecutionScope
from src.core.expressions.statement import BlockStmt
from src.core.expressions.expression import Expression


class StructMethod:
    """
    Represents a method/computed property within a struct definition.

    Attributes:
        name (str): Method identifier name (e.g. 'age', 'is_broken').
        params (dict[str, Type]): Dictionary mapping parameter names to their Type.
        default_param_exprs (dict[str, Expression]): Optional default expressions for parameters.
        return_type (Type): Expected return type of the method.
        body_expression (str): Raw expression string executed by the method.
        body_block (Optional[BlockStmt]): Parsed AST statement block.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        name: str,
        params: dict[str, Type],
        return_type: Type,
        body_expression: str = "",
        default_param_exprs: Optional[dict[str, Expression]] = None,
        body_block: Optional[BlockStmt] = None
    ) -> None:
        """
        Initializes the StructMethod instance.

        Args:
            name (str): Identifier name of the method.
            params (dict[str, Type]): Parameters and their types.
            return_type (Type): Return type specification.
            body_expression (str, optional): Raw body expression. Defaults to "".
            default_param_exprs (Optional[dict[str, Expression]], optional): Defaults.
            body_block (Optional[BlockStmt], optional): Parsed block. Defaults to None.
        """

        self.name: str = name
        self.params: dict[str, Type] = dict(params)
        self.default_param_exprs: dict[str, Expression] = (
            dict(default_param_exprs) if default_param_exprs is not None else {}
        )
        self.return_type: Type = return_type
        self.body_expression: str = body_expression.strip()
        self.body_block: Optional[BlockStmt] = body_block

    def _bind_parameters(
        self,
        args: list[Value],
        kwargs: dict[str, Value],
        scope: ExecutionScope
    ) -> dict[str, Value]:
        """
        Binds positional and keyword arguments to parameter names, evaluating defaults if needed.

        Args:
            args (list[Value]): Positional argument Value objects.
            kwargs (dict[str, Value]): Keyword argument Value objects.
            scope (ExecutionScope): Scope used to evaluate parameter defaults.

        Returns:
            dict[str, Value]: Fully bound parameter dictionary.

        Raises:
            TypeError: If too many positional arguments or missing required arguments.
        """

        param_names: list[str] = list(self.params.keys())

        # Reject mixing positional and keyword arguments
        if args and kwargs:
            raise TypeError(
                f"Method '@{self.name}' cannot mix positional and keyword arguments"
            )

        # Check argument count
        if len(args) > len(param_names):
            raise TypeError(
                f"Method '@{self.name}' takes {len(param_names)} args but {len(args)} given"
            )

        bound: dict[str, Value] = {}

        # Bind positional arguments
        for idx, arg_val in enumerate(args):
            bound[param_names[idx]] = arg_val

        # Bind keyword arguments
        for kw_name, kw_val in kwargs.items():
            if kw_name not in self.params:
                raise TypeError(
                    f"Method '@{self.name}' got unexpected keyword argument '{kw_name}'"
                )

            bound[kw_name] = kw_val

        # Fill in missing arguments from defaults
        # Parameter declaration order is free: defaults may appear anywhere,
        # since positional binding follows the declared order and named binding
        # follows the parameter names.
        for param_name in param_names:
            if param_name not in bound:
                if param_name in self.default_param_exprs:
                    bound[param_name] = self.default_param_exprs[param_name].evaluate(scope=scope)
                else:
                    raise TypeError(
                        f"Method '@{self.name}' missing required argument '{param_name}'"
                    )

        # Validate types
        for param_name, param_val in bound.items():
            expected_type: Type = self.params[param_name]
            raw_val: Any = param_val.get() if hasattr(param_val, "get") else param_val
            expected_type.check_value(value=raw_val)

        return bound

    def execute(
        self,
        self_instance: Any,
        args: Optional[list[Value]] = None,
        kwargs: Optional[dict[str, Value]] = None,
        globals_dict: Optional[dict[str, Any]] = None
    ) -> Optional[Value]:
        """
        Executes the method body on the given struct instance.

        Args:
            self_instance (Any): Receiver struct instance.
            args (Optional[list[Value]], optional): Positional arguments. Defaults to None.
            kwargs (Optional[dict[str, Value]], optional): Keyword arguments. Defaults to None.
            globals_dict (Optional[dict[str, Any]], optional): Global symbols. Defaults to None.

        Returns:
            Optional[Value]: Typed Value returned by the method.

        Raises:
            RuntimeError: If method has no parsed body block to execute.
        """

        if self.body_block is None:
            raise RuntimeError(f"Method '@{self.name}' has no executable body block.")

        resolved_args: list[Value] = args if args is not None else []
        resolved_kwargs: dict[str, Value] = kwargs if kwargs is not None else {}

        # Create base scope for parameter evaluation
        base_scope: ExecutionScope = ExecutionScope(
            self_instance=self_instance,
            globals_dict=globals_dict
        )

        # Bind arguments and evaluate defaults
        bound_params: dict[str, Value] = self._bind_parameters(
            args=resolved_args,
            kwargs=resolved_kwargs,
            scope=base_scope
        )

        # Create method execution scope
        method_scope: ExecutionScope = ExecutionScope(
            self_instance=self_instance,
            parameters=bound_params,
            globals_dict=globals_dict
        )

        # Execute statement block as function
        result: Optional[Value] = self.body_block.execute_as_function(scope=method_scope)

        # Check return type if return value is produced
        if result is not None:
            raw_res: Any = result.get() if hasattr(result, "get") else result
            self.return_type.check_value(value=raw_res)

        return result

    def get_signature_str(self) -> str:
        """
        Generates formatted method signature string.

        Returns:
            str: Signature representation (e.g. '@age(now: DateTime) -> Time').
        """

        param_strs: list[str] = []

        for param_name, param_type in self.params.items():
            if param_name in self.default_param_exprs:
                def_str: str = repr(self.default_param_exprs[param_name])
                param_strs.append(f"{param_name}: {repr(param_type)} = {def_str}")
            else:
                param_strs.append(f"{param_name}: {repr(param_type)}")

        params_formatted: str = ", ".join(param_strs)

        return f"@{self.name}({params_formatted}) -> {repr(self.return_type)}"

    def __repr__(self) -> str:
        """
        Returns string representation of the StructMethod.

        Returns:
            str: Method signature string.
        """

        return self.get_signature_str()
