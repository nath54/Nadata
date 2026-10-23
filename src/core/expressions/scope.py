"""
Execution scope and symbol resolution for Nadata expressions and statements.

Manages variable scopes storing typed Value objects, function parameters,
self-instance bindings, return flow signals, and the shared loop fuel budget.
"""

# Import Modules
from __future__ import annotations
import re
from typing import Any, Optional

from src.core.values.values import Value

# Identifiers naming variables must be lower_snake_case (types are
# UpperCamelCase); mirrors the resolver-side rule for attributes/params
NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-z_][a-z0-9_]*$")

# Globals dict key holding the shared LoopFuel instance
LOOP_FUEL_KEY: str = "__loop_fuel__"

# Globals dict key holding the autograd Tape of the current execution
TAPE_KEY: str = "__tape__"

# Default maximum number of iterations per loop
DEFAULT_LOOP_LIMIT: int = 10_000


class LoopLimitError(Exception):
    """
    Raised when a loop exceeds its fuel limit of allowed iterations.
    """


class LoopFuel:
    """
    Mutable loop iteration budget shared by all scopes of an execution.

    Attributes:
        limit (int): Maximum number of iterations per loop.
    """

    def __init__(self, limit: int = DEFAULT_LOOP_LIMIT) -> None:
        """
        Initializes the LoopFuel instance.

        Args:
            limit (int, optional): Maximum iterations per loop. Defaults to
                DEFAULT_LOOP_LIMIT.
        """

        self.limit: int = limit


class BreakSignal(Exception):
    """
    Control flow signal raised by a break statement to exit the innermost loop.
    """


class ContinueSignal(Exception):
    """
    Control flow signal raised by a continue statement to skip to the next
    loop iteration.
    """


class RaiseSignal(Exception):
    """
    Control flow signal raised by a raise statement in the language.

    Carries a user provided message; catchable by try / catch blocks.

    Attributes:
        message (str): The raised error message.
    """

    def __init__(self, message: str) -> None:
        """
        Initializes the RaiseSignal instance.

        Args:
            message (str): The raised error message.
        """

        self.message: str = message

        super().__init__(message)


class ScopeAwareFunction:
    """
    Native function receiving the live execution scope when invoked.

    Registered in the globals dict like any native; CallExpr detects it and
    passes the current scope to invoke() alongside marshaled raw arguments.
    """

    def invoke(
        self,
        scope: ExecutionScope,
        args: list[Any],
        kwargs: dict[str, Any]
    ) -> Any:
        """
        Invokes the native function with the current scope.

        Args:
            scope (ExecutionScope): Live execution scope of the call site.
            args (list[Value]): Evaluated argument Values.
            kwargs (dict[str, Value]): Evaluated keyword argument Values.

        Returns:
            Any: Call result (wrapped by CallExpr if not a Value).
        """

        raise NotImplementedError("invoke must be implemented by subclasses.")


class ReturnSignal(Exception):
    """
    Control flow signal raised by a return statement to unwind execution.

    Attributes:
        value (Optional[Value]): The typed Value returned by the statement.
    """

    def __init__(self, value: Optional[Value] = None) -> None:
        """
        Initializes the ReturnSignal instance.

        Args:
            value (Optional[Value], optional): Typed Value to propagate. Defaults to None.
        """

        # Initialize parent exception
        super().__init__()

        self.value: Optional[Value] = value


class ExecutionScope:
    """
    Hierarchical execution environment storing typed Value variables, parameters, and self.

    Attributes:
        parent (Optional[ExecutionScope]): Enclosing parent scope for lexical scoping.
        self_instance (Optional[Any]): Reference to the current struct instance receiver.
        variables (dict[str, Value]): Locally declared Value variables in this scope.
        parameters (dict[str, Value]): Bound method Value parameters.
        globals_dict (dict[str, Any]): Global functions, constructors, and types.
        fuel (LoopFuel): Shared loop iteration budget for this execution.
    """

    def __init__(
        self,
        parent: Optional[ExecutionScope] = None,
        self_instance: Optional[Any] = None,
        parameters: Optional[dict[str, Value]] = None,
        globals_dict: Optional[dict[str, Any]] = None,
        fuel: Optional[LoopFuel] = None
    ) -> None:
        """
        Initializes the ExecutionScope instance.

        Args:
            parent (Optional[ExecutionScope], optional): Parent scope. Defaults to None.
            self_instance (Optional[Any], optional): Receiver instance. Defaults to None.
            parameters (Optional[dict[str, Value]], optional): Method parameters. Defaults to None.
            globals_dict (Optional[dict[str, Any]], optional): Global symbols. Defaults to None.
            fuel (Optional[LoopFuel], optional): Explicit loop fuel. Defaults to None
                (inherited from the parent, the reserved globals entry, or a
                fresh budget with the default limit).
        """

        self.parent: Optional[ExecutionScope] = parent
        self.self_instance: Optional[Any] = (
            self_instance if self_instance is not None
            else (parent.self_instance if parent is not None else None)
        )
        self.variables: dict[str, Value] = {}
        self.parameters: dict[str, Value] = dict(parameters) if parameters is not None else {}
        self.globals_dict: dict[str, Any] = (
            dict(globals_dict) if globals_dict is not None
            else (parent.globals_dict if parent is not None else {})
        )

        # Loop fuel: explicit, inherited, reserved globals entry, or default
        if fuel is not None:
            self.fuel: LoopFuel = fuel
        elif parent is not None:
            self.fuel = parent.fuel
        elif LOOP_FUEL_KEY in self.globals_dict:
            self.fuel = self.globals_dict[LOOP_FUEL_KEY]
        else:
            self.fuel = LoopFuel()

    @property
    def tape(self) -> Optional[Any]:
        """
        Resolves the autograd tape of the current execution pass.

        The tape travels through the reserved globals entry (like loop
        fuel), so copies of the globals dict share the same tape object.

        Returns:
            Optional[Any]: The live Tape, or None outside script
                execution (schema-default evaluation, probes).
        """

        if TAPE_KEY in self.globals_dict:
            return self.globals_dict[TAPE_KEY]

        if self.parent is not None:
            return self.parent.tape

        return None

    def define(self, name: str, value: Value) -> None:
        """
        Declares a new typed Value variable in the local block scope.

        Args:
            name (str): Variable identifier.
            value (Value): Typed runtime Value to store.

        Raises:
            TypeError: If value is not an instance of Value.
        """

        if not isinstance(value, Value):
            raise TypeError(f"Expected Value instance, got {type(value).__name__}")

        # Naming rule: variables are lower_snake_case (types are
        # UpperCamelCase); enforced here for scripts and loop variables
        if not NAME_PATTERN.match(name):
            raise SyntaxError(
                f"Variable '{name}' must not start with an uppercase letter "
                f"(lower_snake_case is reserved for attributes, parameters, "
                f"and variables; types are UpperCamelCase)"
            )

        # Store in current local scope variables
        self.variables[name] = value

    def assign(self, name: str, value: Value) -> None:
        """
        Reassigns an existing Value variable in the scope hierarchy.

        Args:
            name (str): Variable identifier.
            value (Value): New typed Value to assign.

        Raises:
            TypeError: If value is not an instance of Value.
            NameError: If variable is not defined anywhere in the scope hierarchy.
        """

        if not isinstance(value, Value):
            raise TypeError(f"Expected Value instance, got {type(value).__name__}")

        if name == "self":
            raise NameError("Cannot assign to 'self'")

        # Update in local variables if present
        if name in self.variables:
            self.variables[name] = value
            return

        # Check parameters
        if name in self.parameters:
            self.parameters[name] = value
            return

        # Check parent scope recursively
        if self.parent is not None:
            self.parent.assign(name=name, value=value)
            return

        raise NameError(f"Cannot assign to undefined variable '{name}'")

    def get(self, name: str) -> Any:
        """
        Resolves an identifier through self, locals, parameters, parent, and globals.

        Attribute access on the receiver is explicit: method bodies read
        instance state through 'self.attribute_name', never through bare
        identifiers. This keeps parameters, locals, attributes, and global
        symbols in disjoint namespaces.

        Args:
            name (str): Identifier name to look up.

        Returns:
            Any: Value bound to the identifier, or global symbol/callable.

        Raises:
            NameError: If identifier cannot be resolved.
        """

        # 1. The receiver itself (explicit self in method bodies)
        if name == "self" and self.self_instance is not None:
            return self.self_instance

        # 2. Check local variables
        if name in self.variables:
            return self.variables[name]

        # 3. Check method parameters
        if name in self.parameters:
            return self.parameters[name]

        # 4. Check own global symbols (child scope bindings live here)
        if name in self.globals_dict:
            return self.globals_dict[name]

        # 5. Check parent scope recursively
        if self.parent is not None:
            return self.parent.get(name=name)

        raise NameError(f"Identifier '{name}' is not defined")

    def create_child_scope(self) -> ExecutionScope:
        """
        Creates a new child scope inheriting from this scope.

        Returns:
            ExecutionScope: New child scope instance.
        """

        return ExecutionScope(
            parent=self,
            self_instance=self.self_instance,
            parameters={},
            globals_dict=self.globals_dict
        )
