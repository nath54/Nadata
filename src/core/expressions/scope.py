"""
Execution scope and symbol resolution for Nadata expressions and statements.

Manages variable scopes storing typed Value objects, function parameters,
self-instance bindings, and return flow signals.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.values.values import Value


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
    """

    def __init__(
        self,
        parent: Optional[ExecutionScope] = None,
        self_instance: Optional[Any] = None,
        parameters: Optional[dict[str, Value]] = None,
        globals_dict: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initializes the ExecutionScope instance.

        Args:
            parent (Optional[ExecutionScope], optional): Parent scope. Defaults to None.
            self_instance (Optional[Any], optional): Receiver instance. Defaults to None.
            parameters (Optional[dict[str, Value]], optional): Method parameters. Defaults to None.
            globals_dict (Optional[dict[str, Any]], optional): Global symbols. Defaults to None.
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
