"""
Tests for the expression parsing package: tokenizer, splitters, expression
parser, and the declaration text parser built on top of them.
"""

# Import Modules
from __future__ import annotations
import pytest

from src.core.expressions.expression import (
    ArrayLiteralExpr,
    BinaryOpExpr,
    CallExpr,
    DictLiteralExpr,
    IdentifierExpr,
    LiteralExpr,
    MemberAccessExpr,
    UnaryOpExpr,
)
from src.parser.expressions.expression_parser import ExpressionParser
from src.parser.expressions.splitter import find_top_level, split_top_level
from src.parser.expressions.tokenizer import Token, TokenType, Tokenizer
from src.parser.nastruct_parser.declaration import parse_declaration_text
from src.parser.nastruct_parser.nastruct_parser import NaStructParser


class TestTokenizer:
    """
    Tests for the Tokenizer.
    """

    def test_identifiers_and_keywords(self) -> None:
        """
        Tests that keywords and identifiers are distinguished.
        """

        tokens: list[Token] = Tokenizer(content="true my_var and").tokenize()

        assert [t.token_type for t in tokens] == [
            TokenType.KEYWORD, TokenType.IDENTIFIER, TokenType.KEYWORD, TokenType.EOF
        ]
        assert tokens[1].value == "my_var"

    def test_numbers(self) -> None:
        """
        Tests integer and float scanning, including member access on ints.
        """

        tokens: list[Token] = Tokenizer(content="21 0.90 1.5.x").tokenize()

        assert [t.token_type for t in tokens] == [
            TokenType.INT, TokenType.FLOAT, TokenType.FLOAT,
            TokenType.PUNCTUATION, TokenType.IDENTIFIER, TokenType.EOF
        ]

    def test_string_escapes(self) -> None:
        """
        Tests escape decoding and position tracking in strings.
        """

        tokens: list[Token] = Tokenizer(content='"a\\"b\\n"').tokenize()

        assert tokens[0].token_type == TokenType.STRING
        assert tokens[0].value == 'a"b\n'
        assert tokens[0].position == 0
        assert tokens[0].end_position == 8

    def test_operators(self) -> None:
        """
        Tests multi-char and single-char operator tokenization.
        """

        tokens: list[Token] = Tokenizer(content="a == b -> c <= d = e !=").tokenize()

        operators: list[str] = [t.value for t in tokens if t.token_type == TokenType.OPERATOR]

        assert operators == ["==", "->", "<=", "=", "!="]

    def test_unterminated_string(self) -> None:
        """
        Tests that an unterminated string raises a SyntaxError.
        """

        with pytest.raises(SyntaxError, match="Unterminated string"):
            Tokenizer(content='"abc').tokenize()

    def test_invalid_escape(self) -> None:
        """
        Tests that an invalid escape sequence raises a SyntaxError.
        """

        with pytest.raises(SyntaxError, match="Invalid escape"):
            Tokenizer(content='"a\\x"').tokenize()

    def test_unexpected_character(self) -> None:
        """
        Tests that an unexpected character raises a SyntaxError.
        """

        with pytest.raises(SyntaxError, match="Unexpected character"):
            Tokenizer(content="a $ b").tokenize()


class TestSplitter:
    """
    Tests for the depth-aware token splitters.
    """

    def test_find_top_level_skips_compound_operators(self) -> None:
        """
        Tests that '==' is not matched when searching for '='.
        """

        tokens: list[Token] = Tokenizer(content="a == b = c").tokenize()

        assert find_top_level(tokens=tokens, value="=") == 3

    def test_find_top_level_skips_nested(self) -> None:
        """
        Tests that separators nested in brackets or strings are skipped.
        """

        tokens: list[Token] = Tokenizer(content='f(a, "x:y")').tokenize()

        assert find_top_level(tokens=tokens, value=":") == -1
        assert find_top_level(tokens=tokens, value=",") == -1

    def test_split_top_level(self) -> None:
        """
        Tests splitting a parameter list on top-level commas.
        """

        tokens: list[Token] = Tokenizer(
            content='a: Int32, b: Array[Str] = ["x", "y"], c: Bool = (p == q)'
        ).tokenize()

        segments: list[list[Token]] = split_top_level(tokens=tokens, separator=",")

        assert len(segments) == 3
        assert segments[0][0].value == "a"
        assert segments[1][0].value == "b"
        assert segments[2][0].value == "c"


class TestExpressionParser:
    """
    Tests for the recursive descent expression parser.
    """

    def test_literals(self) -> None:
        """
        Tests int, float, string, and boolean literal parsing.
        """

        assert isinstance(ExpressionParser.parse_text(text="42"), LiteralExpr)
        assert isinstance(ExpressionParser.parse_text(text="1.5"), LiteralExpr)
        assert isinstance(ExpressionParser.parse_text(text='"hi"'), LiteralExpr)
        assert isinstance(ExpressionParser.parse_text(text="true"), LiteralExpr)

    def test_negative_literal(self) -> None:
        """
        Tests that unary minus produces a UnaryOpExpr.
        """

        expression = ExpressionParser.parse_text(text="-128")

        assert isinstance(expression, UnaryOpExpr)
        assert expression.op == "-"

    def test_precedence(self) -> None:
        """
        Tests that multiplicative binds tighter than additive.
        """

        expression = ExpressionParser.parse_text(text="1 + 2 * 3")

        assert isinstance(expression, BinaryOpExpr)
        assert expression.op == "+"
        assert isinstance(expression.right, BinaryOpExpr)
        assert expression.right.op == "*"

    def test_comparison_and_logic(self) -> None:
        """
        Tests comparison and keyword logical operators.
        """

        expression = ExpressionParser.parse_text(text="a >= 18 and b != c or d")

        assert isinstance(expression, BinaryOpExpr)
        assert expression.op == "or"

    def test_chained_comparison_rejected(self) -> None:
        """
        Tests that chained comparisons raise a SyntaxError.
        """

        with pytest.raises(SyntaxError, match="Chained comparisons"):
            ExpressionParser.parse_text(text="a < b < c")

    def test_member_access_and_calls(self) -> None:
        """
        Tests postfix member access and call parsing.
        """

        expression = ExpressionParser.parse_text(text="DateTime.now()")

        assert isinstance(expression, CallExpr)
        assert isinstance(expression.callee, MemberAccessExpr)
        assert expression.callee.member == "now"

        expression = ExpressionParser.parse_text(text="obj.method(1, name=x)")

        assert isinstance(expression, CallExpr)
        assert len(expression.args) == 1
        assert list(expression.kwargs) == ["name"]
        assert isinstance(expression.kwargs["name"], IdentifierExpr)
        assert isinstance(expression.callee, MemberAccessExpr)
        assert isinstance(expression.callee.target, IdentifierExpr)

    def test_mixed_call_arguments_rejected(self) -> None:
        """
        Tests that positional arguments cannot follow named arguments.
        """

        with pytest.raises(SyntaxError, match="Positional argument after"):
            ExpressionParser.parse_text(text="f(a=1, 2)")

    def test_array_and_dict_literals(self) -> None:
        """
        Tests array and dict literal parsing.
        """

        expression = ExpressionParser.parse_text(text='["nadata-engine", "replication"]')

        assert isinstance(expression, ArrayLiteralExpr)
        assert len(expression.elements) == 2

        expression = ExpressionParser.parse_text(text='{"key": "value"}')

        assert isinstance(expression, DictLiteralExpr)
        assert len(expression.entries) == 1

    def test_trailing_tokens_rejected(self) -> None:
        """
        Tests that trailing tokens after a full expression raise.
        """

        with pytest.raises(SyntaxError, match="Unexpected trailing token"):
            ExpressionParser.parse_text(text="a b")


class TestParseDeclarationText:
    """
    Tests for the token-based declaration text parser.
    """

    def test_simple_declaration(self) -> None:
        """
        Tests a plain 'name: Type' declaration.
        """

        name: str
        attr_type: str
        default: str | None
        name, attr_type, default = parse_declaration_text(declaration_text="age: UInt8")

        assert name == "age"
        assert attr_type == "UInt8"
        assert default is None

    def test_default_value(self) -> None:
        """
        Tests a declaration with a default value.
        """

        name: str
        attr_type: str
        default: str | None
        name, attr_type, default = parse_declaration_text(
            declaration_text='ip_address: Str = "127.0.0.1"'
        )

        assert name == "ip_address"
        assert attr_type == "Str"
        assert default == '"127.0.0.1"'

    def test_comparison_default(self) -> None:
        """
        Tests that a default containing '==' is split correctly.
        """

        name: str
        attr_type: str
        default: str | None
        name, attr_type, default = parse_declaration_text(
            declaration_text="is_valid: Bool = a == b"
        )

        assert name == "is_valid"
        assert attr_type == "Bool"
        assert default == "a == b"

    def test_generic_type_with_default(self) -> None:
        """
        Tests a generic type annotation with a collection default.
        """

        name: str
        attr_type: str
        default: str | None
        name, attr_type, default = parse_declaration_text(
            declaration_text='routing_table: Dict[Str, Str] = {"a": "b"}'
        )

        assert name == "routing_table"
        assert attr_type == "Dict[Str, Str]"
        assert default == '{"a": "b"}'


class TestParseCallables:
    """
    Tests for method and standalone function parsing on real examples.
    """

    def test_methods_parsed_from_example(self) -> None:
        """
        Tests that struct methods are parsed with parameters and raw bodies.
        """

        parser: NaStructParser = NaStructParser()
        parser.parse_folder(folder_path="./examples/ex2_inheritance/structs")

        server_tmp = parser.parser_state.tmp_structs["Server"]

        assert "is_overloaded" in server_tmp.methods

        method = server_tmp.methods["is_overloaded"]

        assert method.fn_output_type == "Bool"
        assert "current_load" in method.fn_argument
        assert method.fn_argument["current_load"].attribute_type == "UFloat32"
        assert "return" in method.fn_body

    def test_inherited_method_parsed(self) -> None:
        """
        Tests that the base Node.describe() method is parsed.
        """

        parser: NaStructParser = NaStructParser()
        parser.parse_folder(folder_path="./examples/ex2_inheritance/structs")

        node_tmp = parser.parser_state.tmp_structs["Node"]

        assert "describe" in node_tmp.methods
        assert node_tmp.methods["describe"].fn_output_type == "Str"
        assert node_tmp.methods["describe"].fn_argument == {}

    def test_default_parameter_value(self) -> None:
        """
        Tests that a parameter with a default value is parsed.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=(
            "class A {\n"
            "    @m(x: Int32 = 3) -> Int32 {\n"
            "        return x;\n"
            "    }\n"
            "}\n"
        ))

        method = parser.parser_state.tmp_structs["A"].methods["m"]

        assert method.fn_argument["x"].attribute_default_value == "3"

    def test_standalone_function_parsed(self) -> None:
        """
        Tests that standalone functions are registered in tmp_functions.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=(
            "class Human {\n"
            "    age: UInt8;\n"
            "}\n"
            "\n"
            "fn is_adult(h: Human) -> Bool {\n"
            "    return h.age >= 18;\n"
            "}\n"
        ))

        assert "is_adult" in parser.parser_state.tmp_functions

        fn = parser.parser_state.tmp_functions["is_adult"]

        assert fn.fn_output_type == "Bool"
        assert fn.fn_argument["h"].attribute_type == "Human"
        assert "return" in fn.fn_body

    def test_duplicate_method_rejected(self) -> None:
        """
        Tests that a duplicated method name raises a SyntaxError.
        """

        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="already defined"):
            parser.parse_file(content=(
                "class A {\n"
                "    @m() -> Int32 {\n"
                "        return 1;\n"
                "    }\n"
                "    @m() -> Int32 {\n"
                "        return 2;\n"
                "    }\n"
                "}\n"
            ))

    def test_duplicate_function_rejected(self) -> None:
        """
        Tests that a duplicated function name raises a SyntaxError.
        """

        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="already defined"):
            parser.parse_file(content=(
                "fn f() -> Int32 {\n"
                "    return 1;\n"
                "}\n"
                "fn f() -> Int32 {\n"
                "    return 2;\n"
                "}\n"
            ))

class TestDependencyCollection:
    """
    Tests for AST based dependency collection on real declarations.
    """

    def test_member_call_default_excludes_member_name(self) -> None:
        """
        Tests that 'DateTime.now()' depends only on 'DateTime', not 'now'.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=(
            "class A {\n"
            "    birthday: DateTime = DateTime.now();\n"
            "}\n"
        ))

        attribute = parser.parser_state.tmp_structs["A"].attributes["birthday"]

        assert attribute.dependencies == {"DateTime"}

    def test_plain_identifier_default_is_collected(self) -> None:
        """
        Tests that a bare identifier default is still a dependency.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=(
            "class A {\n"
            "    is_valid: Bool = some_flag;\n"
            "}\n"
        ))

        attribute = parser.parser_state.tmp_structs["A"].attributes["is_valid"]

        assert attribute.dependencies == {"Bool", "some_flag"}

    def test_nested_member_chain_default(self) -> None:
        """
        Tests that only the root identifier of a member chain is collected.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=(
            "class A {\n"
            "    x: Int32 = obj.inner.count();\n"
            "}\n"
        ))

        attribute = parser.parser_state.tmp_structs["A"].attributes["x"]

        assert attribute.dependencies == {"Int32", "obj"}
