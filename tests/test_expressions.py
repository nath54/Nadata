"""
Unit tests for AST expression and statement execution engine.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from nadata.core.struct.struct import Struct
from nadata.core.types.types import (
    Int32,
    Float32,
    TypeBool,
    TypeFloat,
)
from nadata.core.values.values import (
    Value,
    ValueInt,
    ValueBool,
    ValueFloat,
)
from nadata.core.struct.struct_method import StructMethod
from nadata.core.expressions.scope import ExecutionScope
from nadata.core.struct.struct_attribute import StructAttribute
from nadata.core.expressions.statement import (
    IfStmt,
    BlockStmt,
    AssignStmt,
    ReturnStmt,
    VarDeclStmt,
)
from nadata.core.expressions.expression import (
    CallExpr,
    LiteralExpr,
    BinaryOpExpr,
    IdentifierExpr,
    MemberAccessExpr,
)


class MockObjectInstance:
    """
    Mock struct instance used for testing method execution and attribute resolution.

    Attributes:
        struct (Struct): Associated struct schema definition.
        attributes (dict[str, Value]): Bound attributes dictionary.
    """

    def __init__(self, struct: Struct, attributes: dict[str, Value]) -> None:
        """
        Initializes the MockObjectInstance.

        Args:
            struct (Struct): Struct schema.
            attributes (dict[str, Value]): Attributes map.
        """

        self.struct: Struct = struct
        self.attributes: dict[str, Value] = attributes

    def get_attribute(self, name: str) -> Value:
        """
        Retrieves an attribute value.

        Args:
            name (str): Attribute identifier.

        Returns:
            Value: Attribute Value.
        """

        return self.attributes[name]

    def __repr__(self) -> str:
        """
        Returns string representation of the instance.

        Returns:
            str: Representation string.
        """

        return f"Mock({self.struct.name})"


def test_is_broken_method_logic() -> None:
    """
    Tests execution of the @is_broken method AST from structs.nastruct:
    broken: Bool = false;
    if (state <= 0) {
        broken = true;
    }
    return broken;
    """

    # Build AST statements
    # 1. broken: Bool = false;
    stmt_decl: VarDeclStmt = VarDeclStmt(
        name="broken",
        init_expr=LiteralExpr(value=ValueBool(value_type=TypeBool(), value=False)),
        var_type=TypeBool()
    )

    # 2. if (self.state <= 0) { broken = true; }
    cond_expr: BinaryOpExpr = BinaryOpExpr(
        left=MemberAccessExpr(
            target=IdentifierExpr(name="self"),
            member="state"
        ),
        op="<=",
        right=LiteralExpr(value=ValueInt(value_type=Int32, value=0))
    )
    then_block: BlockStmt = BlockStmt(statements=[
        AssignStmt(
            target_name="broken",
            value_expr=LiteralExpr(value=ValueBool(value_type=TypeBool(), value=True))
        )
    ])
    stmt_if: IfStmt = IfStmt(condition=cond_expr, then_block=then_block)

    # 3. return broken;
    stmt_return: ReturnStmt = ReturnStmt(value_expr=IdentifierExpr(name="broken"))

    # Assemble body block
    body: BlockStmt = BlockStmt(statements=[stmt_decl, stmt_if, stmt_return])

    # Create StructMethod
    method_is_broken: StructMethod = StructMethod(
        name="is_broken",
        params={},
        return_type=TypeBool(),
        body_block=body
    )

    struct_obj: Struct = Struct(name="ObjectInstance")
    struct_obj.add_attribute(
        attribute=StructAttribute(name="state", attr_type=TypeFloat(num_bits=32))
    )
    struct_obj.add_method(method=method_is_broken)

    # Test Case 1: state = 1.0 -> not broken
    mock_intact: MockObjectInstance = MockObjectInstance(
        struct=struct_obj,
        attributes={"state": ValueFloat(value_type=Float32, value=1.0)}
    )
    result_intact: Optional[Value] = method_is_broken.execute(self_instance=mock_intact)

    assert result_intact is not None
    assert isinstance(result_intact, ValueBool)
    assert result_intact.get() is False

    # Test Case 2: state = -0.5 -> broken
    mock_broken: MockObjectInstance = MockObjectInstance(
        struct=struct_obj,
        attributes={"state": ValueFloat(value_type=Float32, value=-0.5)}
    )
    result_broken: Optional[Value] = method_is_broken.execute(self_instance=mock_broken)

    assert result_broken is not None
    assert isinstance(result_broken, ValueBool)
    assert result_broken.get() is True


def test_member_call_and_older_logic() -> None:
    """
    Tests member call expressions: other_entity.age() and comparison logic.
    """

    struct_entity: Struct = Struct(name="Entity")

    # Define @age method returning ValueInt
    method_age: StructMethod = StructMethod(
        name="age",
        params={},
        return_type=Int32,
        body_block=BlockStmt(statements=[
            ReturnStmt(value_expr=MemberAccessExpr(
                target=IdentifierExpr(name="self"),
                member="stored_age"
            ))
        ])
    )
    struct_entity.add_method(method=method_age)

    # Instances
    tom: MockObjectInstance = MockObjectInstance(
        struct=struct_entity,
        attributes={"stored_age": ValueInt(value_type=Int32, value=23)}
    )
    jean: MockObjectInstance = MockObjectInstance(
        struct=struct_entity,
        attributes={"stored_age": ValueInt(value_type=Int32, value=54)}
    )

    # Test CallExpr on member access: jean.age()
    call_expr: CallExpr = CallExpr(
        callee=MemberAccessExpr(
            target=IdentifierExpr(name="other_entity"),
            member="age"
        )
    )

    scope: ExecutionScope = ExecutionScope(
        parameters={"other_entity": jean}  # type: ignore[dict-item]
    )
    res_age: Value = call_expr.evaluate(scope=scope)

    assert isinstance(res_age, ValueInt)
    assert res_age.get() == 54

    # Build @older method on Entity:
    # other_age: Int = other.age();
    # self_age: Int = age();
    # if (other_age >= self_age) { return false; }
    # return true;
    body_older: BlockStmt = BlockStmt(statements=[
        VarDeclStmt(
            name="other_age",
            init_expr=CallExpr(
                callee=MemberAccessExpr(
                    target=IdentifierExpr(name="other_entity"),
                    member="age"
                )
            ),
            var_type=Int32
        ),
        VarDeclStmt(
            name="self_age",
            init_expr=CallExpr(
                callee=MemberAccessExpr(
                    target=IdentifierExpr(name="self"),
                    member="age"
                )
            ),
            var_type=Int32
        ),
        IfStmt(
            condition=BinaryOpExpr(
                left=IdentifierExpr(name="other_age"),
                op=">=",
                right=IdentifierExpr(name="self_age")
            ),
            then_block=BlockStmt(statements=[
                ReturnStmt(value_expr=LiteralExpr(value=False))
            ])
        ),
        ReturnStmt(value_expr=LiteralExpr(value=True))
    ])

    method_older: StructMethod = StructMethod(
        name="older",
        params={"other_entity": struct_entity},  # type: ignore[dict-item]
        return_type=TypeBool(),
        body_block=body_older
    )
    struct_entity.add_method(method=method_older)

    # tom (age 23) is NOT older than jean (age 54) -> False
    res_older: Optional[Value] = method_older.execute(
        self_instance=tom,
        args=[jean]  # type: ignore[list-item]
    )

    assert res_older is not None
    assert isinstance(res_older, ValueBool)
    assert res_older.get() is False

    # jean (age 54) IS older than tom (age 23) -> True
    res_jean_older: Optional[Value] = method_older.execute(
        self_instance=jean,
        args=[tom]  # type: ignore[list-item]
    )

    assert res_jean_older is not None
    assert isinstance(res_jean_older, ValueBool)
    assert res_jean_older.get() is True


def test_logical_operators_keywords() -> None:
    """
    Tests the keyword-only logical operators: and, or, xor, nand.
    """

    def eval_op(op: str, left: bool, right: bool) -> bool:
        expr: BinaryOpExpr = BinaryOpExpr(
            left=LiteralExpr(value=ValueBool(value_type=TypeBool(), value=left)),
            op=op,
            right=LiteralExpr(value=ValueBool(value_type=TypeBool(), value=right)),
        )
        result: Value = expr.evaluate(scope=ExecutionScope())
        return bool(result.get())

    # and
    assert eval_op("and", True, True) is True
    assert eval_op("and", True, False) is False

    # or
    assert eval_op("or", False, True) is True
    assert eval_op("or", False, False) is False

    # xor
    assert eval_op("xor", True, False) is True
    assert eval_op("xor", True, True) is False
    assert eval_op("xor", False, False) is False

    # nand
    assert eval_op("nand", True, True) is False
    assert eval_op("nand", True, False) is True
    assert eval_op("nand", False, False) is True
