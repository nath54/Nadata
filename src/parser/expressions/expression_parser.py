"""
Recursive descent expression parser for Nadata.

Converts token streams into src.core.expressions AST nodes. The precedence
levels (loosest to tightest) are: or, xor, nand, and, comparisons, additive,
multiplicative, unary minus, postfix (member access, calls), primary.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.core.expressions.expression import (
    ArrayLiteralExpr,
    BinaryOpExpr,
    CallExpr,
    DictLiteralExpr,
    Expression,
    IdentifierExpr,
    LiteralExpr,
    MemberAccessExpr,
    UnaryOpExpr,
)
from src.parser.expressions.tokenizer import Token, TokenType, Tokenizer

# Comparison operators (single, non-chained comparison per expression)
COMPARISON_OPS: tuple[str, ...] = ("==", "!=", "<=", ">=", "<", ">")

# Boolean literal keywords mapped to Python values
BOOLEAN_KEYWORDS: dict[str, bool] = {"true": True, "false": False}


class ExpressionParser:
    """
    Parses a token stream into an Expression AST using recursive descent.
    """

    def __init__(self, tokens: list[Token]) -> None:
        """
        Initializes the ExpressionParser instance.

        Args:
            tokens (list[Token]): Token stream to parse (terminated by EOF).
        """

        self.tokens: list[Token] = tokens
        self.cursor: int = 0

    @classmethod
    def parse_text(cls, text: str) -> Expression:
        """
        Tokenizes and parses a raw expression text in one step.

        Args:
            text (str): Raw expression source text.

        Returns:
            Expression: Parsed expression AST.
        """

        tokens: list[Token] = Tokenizer(content=text).tokenize()

        return cls(tokens=tokens).parse()

    def parse(self) -> Expression:
        """
        Parses the whole token stream into a single expression.

        Returns:
            Expression: Parsed expression AST.

        Raises:
            SyntaxError: On trailing tokens after the expression.
        """

        expression: Expression = self._parse_or()

        # Check nothing remains after the expression
        self._expect_end()

        return expression

    def parse_expression_at(self, cursor: int) -> tuple[Expression, int]:
        """
        Parses a single expression starting at the given token cursor.

        Does not check for end of stream: the returned cursor points just
        after the expression, for the caller to continue scanning.

        Args:
            cursor (int): Token index to start parsing from.

        Returns:
            tuple[Expression, int]: The parsed expression and the new cursor.

        Raises:
            SyntaxError: On malformed expressions.
        """

        self.cursor = cursor

        expression: Expression = self._parse_or()

        return expression, self.cursor

    # --- Precedence levels (loosest first) ---

    def _parse_or(self) -> Expression:
        """
        Parses the 'or' precedence level.

        Returns:
            Expression: Parsed expression.
        """

        left: Expression = self._parse_xor()

        # Left-associate chained 'or' operators
        while self._match_keyword("or"):
            right: Expression = self._parse_xor()
            left = BinaryOpExpr(left=left, op="or", right=right)

        return left

    def _parse_xor(self) -> Expression:
        """
        Parses the 'xor' precedence level.

        Returns:
            Expression: Parsed expression.
        """

        left: Expression = self._parse_nand()

        # Left-associate chained 'xor' operators
        while self._match_keyword("xor"):
            right: Expression = self._parse_nand()
            left = BinaryOpExpr(left=left, op="xor", right=right)

        return left

    def _parse_nand(self) -> Expression:
        """
        Parses the 'nand' precedence level.

        Returns:
            Expression: Parsed expression.
        """

        left: Expression = self._parse_and()

        # Left-associate chained 'nand' operators
        while self._match_keyword("nand"):
            right: Expression = self._parse_and()
            left = BinaryOpExpr(left=left, op="nand", right=right)

        return left

    def _parse_and(self) -> Expression:
        """
        Parses the 'and' precedence level.

        Returns:
            Expression: Parsed expression.
        """

        left: Expression = self._parse_comparison()

        # Left-associate chained 'and' operators
        while self._match_keyword("and"):
            right: Expression = self._parse_comparison()
            left = BinaryOpExpr(left=left, op="and", right=right)

        return left

    def _parse_comparison(self) -> Expression:
        """
        Parses the comparison precedence level (non-chained).

        Returns:
            Expression: Parsed expression.

        Raises:
            SyntaxError: On chained comparisons (e.g. 'a < b < c').
        """

        left: Expression = self._parse_additive()

        # A single optional comparison
        if self._current().token_type == TokenType.OPERATOR:

            operator_token: Token = self._current()

            if operator_token.value in COMPARISON_OPS:
                self.cursor += 1

                right: Expression = self._parse_additive()

                # Check no comparison chaining follows
                if (
                    self._current().token_type == TokenType.OPERATOR
                    and self._current().value in COMPARISON_OPS
                ):
                    raise SyntaxError(
                        f"Chained comparisons are not supported at index "
                        f"{self._current().position}"
                    )

                return BinaryOpExpr(
                    left=left, op=operator_token.value, right=right
                )

        return left

    def _parse_additive(self) -> Expression:
        """
        Parses the additive precedence level ('+' and '-').

        Returns:
            Expression: Parsed expression.
        """

        left: Expression = self._parse_multiplicative()

        # Left-associate chained additive operators
        while (
            self._current().token_type == TokenType.OPERATOR
            and self._current().value in ("+", "-")
        ):
            operator: str = self._current().value
            self.cursor += 1

            right: Expression = self._parse_multiplicative()
            left = BinaryOpExpr(left=left, op=operator, right=right)

        return left

    def _parse_multiplicative(self) -> Expression:
        """
        Parses the multiplicative precedence level ('*', '/', '%').

        Returns:
            Expression: Parsed expression.
        """

        left: Expression = self._parse_unary()

        # Left-associate chained multiplicative operators
        while (
            self._current().token_type == TokenType.OPERATOR
            and self._current().value in ("*", "/", "%")
        ):
            operator: str = self._current().value
            self.cursor += 1

            right: Expression = self._parse_unary()
            left = BinaryOpExpr(left=left, op=operator, right=right)

        return left

    def _parse_unary(self) -> Expression:
        """
        Parses the unary minus precedence level.

        Returns:
            Expression: Parsed expression.
        """

        # Unary minus applied to the following unary expression
        if (
            self._current().token_type == TokenType.OPERATOR
            and self._current().value == "-"
        ):
            self.cursor += 1

            operand: Expression = self._parse_unary()

            return UnaryOpExpr(op="-", operand=operand)

        return self._parse_postfix()

    def _parse_postfix(self) -> Expression:
        """
        Parses postfix operations: member access and calls.

        Returns:
            Expression: Parsed expression.
        """

        expression: Expression = self._parse_primary()

        # Consume postfix operators while present
        while True:

            # Member access '.name'
            if self._match_punctuation("."):
                member_token: Token = self._expect_identifier()
                expression = MemberAccessExpr(
                    target=expression, member=member_token.value
                )

            # Call '(...)' with positional or named arguments
            elif self._match_punctuation("("):
                expression = self._parse_call(callee=expression)

            # No more postfix operators
            else:
                break

        return expression

    def _parse_primary(self) -> Expression:
        """
        Parses a primary expression: literals, identifiers, parentheses, and
        array / dict literals.

        Returns:
            Expression: Parsed expression.

        Raises:
            SyntaxError: If no primary expression starts at the cursor.
        """

        token: Token = self._current()

        # Numeric literals
        if token.token_type == TokenType.INT:
            self.cursor += 1
            return LiteralExpr(value=int(token.value))

        if token.token_type == TokenType.FLOAT:
            self.cursor += 1
            return LiteralExpr(value=float(token.value))

        # String literals
        if token.token_type == TokenType.STRING:
            self.cursor += 1
            return LiteralExpr(value=token.value)

        # Boolean keywords
        if token.token_type == TokenType.KEYWORD and token.value in BOOLEAN_KEYWORDS:
            self.cursor += 1
            return LiteralExpr(value=BOOLEAN_KEYWORDS[token.value])

        # Plain identifiers
        if token.token_type == TokenType.IDENTIFIER:
            self.cursor += 1
            return IdentifierExpr(name=token.value)

        # Parenthesized expressions
        if token.token_type == TokenType.PUNCTUATION and token.value == "(":
            self.cursor += 1

            inner: Expression = self._parse_or()

            self._expect_punctuation(")")

            return inner

        # Array literals
        if token.token_type == TokenType.PUNCTUATION and token.value == "[":
            return self._parse_array_literal()

        # Dict literals
        if token.token_type == TokenType.PUNCTUATION and token.value == "{":
            return self._parse_dict_literal()

        raise SyntaxError(
            f"Unexpected token '{token.value}' at index {token.position}"
        )

    def _parse_array_literal(self) -> ArrayLiteralExpr:
        """
        Parses an array literal '[element, element, ...]'.

        Returns:
            ArrayLiteralExpr: Parsed array literal node.
        """

        # Consume the opening bracket
        self._expect_punctuation("[")

        # Empty array fast path
        if self._match_punctuation("]"):
            return ArrayLiteralExpr(elements=[])

        # Parse comma separated elements
        elements: list[Expression] = [self._parse_or()]

        while self._match_punctuation(","):
            elements.append(self._parse_or())

        self._expect_punctuation("]")

        return ArrayLiteralExpr(elements=elements)

    def _parse_dict_literal(self) -> DictLiteralExpr:
        """
        Parses a dict literal '{key: value, ...}'.

        Returns:
            DictLiteralExpr: Parsed dict literal node.
        """

        # Consume the opening brace
        self._expect_punctuation("{")

        # Empty dict fast path
        if self._match_punctuation("}"):
            return DictLiteralExpr(entries=[])

        # Parse comma separated key / value entries
        entries: list[tuple[Expression, Expression]] = []

        while True:

            key: Expression = self._parse_or()

            self._expect_punctuation(":")

            value: Expression = self._parse_or()

            entries.append((key, value))

            # Continue on comma, else close the dict
            if not self._match_punctuation(","):
                break

        self._expect_punctuation("}")

        return DictLiteralExpr(entries=entries)

    def _parse_call(self, callee: Expression) -> CallExpr:
        """
        Parses a call argument list after the opening parenthesis.

        Args:
            callee (Expression): Parsed callee expression.

        Returns:
            CallExpr: Parsed call node.

        Raises:
            SyntaxError: On mixed positional and named arguments.
        """

        # Empty argument list fast path
        if self._match_punctuation(")"):
            return CallExpr(callee=callee)

        args: list[Expression] = []
        kwargs: dict[str, Expression] = {}

        # Parse comma separated arguments
        while True:

            # Named argument 'name=value'
            if self._is_named_argument():
                name_token: Token = self._expect_identifier()

                # Consume the '=' (guaranteed by _is_named_argument)
                self._expect_operator("=")

                kwargs[name_token.value] = self._parse_or()

            # Positional argument
            else:
                # Positional arguments cannot follow named arguments
                if kwargs:
                    raise SyntaxError(
                        "Positional argument after a named argument is not allowed"
                    )

                args.append(self._parse_or())

            # Continue on comma, else close the call
            if not self._match_punctuation(","):
                break

        self._expect_punctuation(")")

        return CallExpr(callee=callee, args=args, kwargs=kwargs)

    # --- Token stream helpers ---

    def _current(self) -> Token:
        """
        Returns the token at the current cursor.

        Returns:
            Token: The current token.
        """

        return self.tokens[self.cursor]

    def _match_keyword(self, keyword: str) -> bool:
        """
        Consumes the current token if it is the given keyword.

        Args:
            keyword (str): Expected keyword value.

        Returns:
            bool: True if the keyword was matched and consumed.
        """

        token: Token = self._current()

        if token.token_type == TokenType.KEYWORD and token.value == keyword:
            self.cursor += 1
            return True

        return False

    def _match_punctuation(self, punctuation: str) -> bool:
        """
        Consumes the current token if it is the given punctuation.

        Args:
            punctuation (str): Expected punctuation value.

        Returns:
            bool: True if the punctuation was matched and consumed.
        """

        token: Token = self._current()

        if token.token_type == TokenType.PUNCTUATION and token.value == punctuation:
            self.cursor += 1
            return True

        return False

    def _expect_punctuation(self, punctuation: str) -> None:
        """
        Consumes the current token, which must be the given punctuation.

        Args:
            punctuation (str): Expected punctuation value.

        Raises:
            SyntaxError: If the current token does not match.
        """

        token: Token = self._current()

        if not (token.token_type == TokenType.PUNCTUATION and token.value == punctuation):
            raise SyntaxError(
                f"Expected '{punctuation}' but found '{token.value}' "
                f"at index {token.position}"
            )

        self.cursor += 1

    def _expect_operator(self, operator: str) -> None:
        """
        Consumes the current token, which must be the given operator.

        Args:
            operator (str): Expected operator value.

        Raises:
            SyntaxError: If the current token does not match.
        """

        token: Token = self._current()

        if not (token.token_type == TokenType.OPERATOR and token.value == operator):
            raise SyntaxError(
                f"Expected '{operator}' but found '{token.value}' "
                f"at index {token.position}"
            )

        self.cursor += 1

    def _expect_identifier(self) -> Token:
        """
        Consumes the current token, which must be an identifier.

        Returns:
            Token: The consumed identifier token.

        Raises:
            SyntaxError: If the current token is not an identifier.
        """

        token: Token = self._current()

        if token.token_type != TokenType.IDENTIFIER:
            raise SyntaxError(
                f"Expected identifier but found '{token.value}' "
                f"at index {token.position}"
            )

        self.cursor += 1

        return token

    def _expect_end(self) -> None:
        """
        Checks that only the EOF token remains.

        Raises:
            SyntaxError: On trailing tokens after the expression.
        """

        token: Token = self._current()

        if token.token_type != TokenType.EOF:
            raise SyntaxError(
                f"Unexpected trailing token '{token.value}' at index {token.position}"
            )

    def _is_named_argument(self) -> bool:
        """
        Checks whether a named argument starts at the cursor.

        Returns:
            bool: True if the cursor is on 'identifier =' (and not '==').
        """

        current_token: Token = self._current()
        next_token: Token = self.tokens[self.cursor + 1]

        is_identifier: bool = current_token.token_type == TokenType.IDENTIFIER
        is_assignment: bool = (
            next_token.token_type == TokenType.OPERATOR
            and next_token.value == "="
        )

        return is_identifier and is_assignment
