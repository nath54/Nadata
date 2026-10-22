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
    IndexExpr,
    LambdaExpr,
    LiteralExpr,
    MemberAccessExpr,
    SetLiteralExpr,
    SliceOperand,
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
        Parses the comparison precedence level (non-chained), including the
        keyword membership operators 'in' and 'not in'.

        Returns:
            Expression: Parsed expression.

        Raises:
            SyntaxError: On chained comparisons (e.g. 'a < b < c') or a
                trailing keyword operator (e.g. 'a < b in c').
        """

        left: Expression = self._parse_additive()

        # Membership keyword operators bind at comparison precedence
        membership_op: Optional[str] = self._parse_membership_operator()

        if membership_op is not None:
            membership_right: Expression = self._parse_additive()

            self._reject_trailing_comparison()

            return BinaryOpExpr(
                left=left, op=membership_op, right=membership_right
            )

        # A single optional comparison
        if self._current().token_type == TokenType.OPERATOR:

            operator_token: Token = self._current()

            if operator_token.value in COMPARISON_OPS:
                self.cursor += 1

                right: Expression = self._parse_additive()

                self._reject_trailing_comparison()

                return BinaryOpExpr(
                    left=left, op=operator_token.value, right=right
                )

        return left

    def _parse_membership_operator(self) -> Optional[str]:
        """
        Parses a membership keyword operator at the cursor.

        Returns:
            Optional[str]: 'in', 'not in', or None if no membership
            operator is present.

        Raises:
            SyntaxError: On a dangling 'not' that is not followed by 'in'.
        """

        token: Token = self._current()

        if token.token_type != TokenType.KEYWORD:
            return None

        if token.value == "in":
            self.cursor += 1

            return "in"

        if token.value == "not":
            next_index: int = self.cursor + 1

            if next_index < len(self.tokens):
                next_token: Token = self.tokens[next_index]

                if next_token.token_type == TokenType.KEYWORD and next_token.value == "in":
                    self.cursor += 2

                    return "not in"

            raise SyntaxError(
                f"Expected 'in' after 'not' at index {token.position}"
            )

        return None

    def _reject_trailing_comparison(self) -> None:
        """
        Rejects comparison or membership operators following an already
        parsed comparison or membership expression.

        Raises:
            SyntaxError: If a chaining operator is present at the cursor.
        """

        token: Token = self._current()

        if token.token_type == TokenType.OPERATOR and token.value in COMPARISON_OPS:
            raise SyntaxError(
                f"Chained comparisons are not supported at index "
                f"{token.position}"
            )

        if token.token_type == TokenType.KEYWORD and token.value in ("in", "not"):
            raise SyntaxError(
                f"Chained membership expressions are not supported at "
                f"index {token.position}"
            )

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

        # Unary negation applied to the following unary expression
        if (
            self._current().token_type == TokenType.KEYWORD
            and self._current().value == "not"
        ):
            self.cursor += 1

            operand: Expression = self._parse_unary()

            return UnaryOpExpr(op="not", operand=operand)

        # Unary minus applied to the following unary expression
        if (
            self._current().token_type == TokenType.OPERATOR
            and self._current().value == "-"
        ):
            self.cursor += 1

            minus_operand: Expression = self._parse_unary()

            return UnaryOpExpr(op="-", operand=minus_operand)

        return self._parse_postfix()

    def _parse_postfix(self) -> Expression:
        """
        Parses postfix operations: member access and calls.

        Returns:
            Expression: Parsed expression.
        """

        expression: Expression

        # Lambda '(param: ParamType) -> body' has primary-like precedence
        if self._at_lambda_start():
            expression = self._parse_lambda()

        else:
            expression = self._parse_primary()

        # Consume postfix operators while present
        while True:

            # Member access '.name'
            if self._match_punctuation("."):
                member_token: Token = self._expect_identifier()
                expression = MemberAccessExpr(
                    target=expression, member=member_token.value
                )

            # Indexing '[index]' — arrays, dictionaries, and multi-index
            # tensors (comma-separated, 'start:stop' slices allowed),
            # chainable
            elif self._match_punctuation("["):
                index_expr: Expression = self._parse_index_item()
                extra_indices: list[Expression] = []

                # Comma-separated indices form a multi-index tensor access
                while self._match_punctuation(","):
                    extra_indices.append(self._parse_index_item())

                self._expect_punctuation("]")

                expression = IndexExpr(
                    target=expression,
                    index=index_expr,
                    extra_indices=extra_indices,
                )

            # Call '(...)' with positional or named arguments
            elif self._match_punctuation("("):
                expression = self._parse_call(callee=expression)

            # No more postfix operators
            else:
                break

        return expression

    def _at_lambda_start(self) -> bool:
        """
        Checks whether a lambda expression starts at the cursor.

        A lambda starts with '( identifier :' — any other parenthesized
        expression starts differently and is parsed as a primary.

        Returns:
            bool: True if a lambda expression starts at the cursor.
        """

        token: Token = self._current()

        if not (token.token_type == TokenType.PUNCTUATION and token.value == "("):
            return False

        # Guard against running past the stream end
        if self.cursor + 2 >= len(self.tokens):
            return False

        name_token: Token = self.tokens[self.cursor + 1]
        colon_token: Token = self.tokens[self.cursor + 2]

        return (
            name_token.token_type == TokenType.IDENTIFIER
            and colon_token.token_type == TokenType.PUNCTUATION
            and colon_token.value == ":"
        )

    def _parse_lambda(self) -> LambdaExpr:
        """
        Parses a lambda expression '(param: ParamType) -> body'.

        Returns:
            LambdaExpr: The parsed lambda expression.

        Raises:
            SyntaxError: On malformed parameter list or missing arrow.
        """

        # Consume the opening parenthesis
        self._expect_punctuation("(")

        param_token: Token = self._expect_identifier()

        self._expect_punctuation(":")

        # Collect the raw parameter type tokens up to the closing parenthesis
        type_values: list[str] = []

        while not (
            self._current().token_type == TokenType.PUNCTUATION
            and self._current().value == ")"
        ):
            if self._current().token_type == TokenType.EOF:
                raise SyntaxError(
                    f"Missing ')' in lambda parameter list at index {self._current().position}"
                )

            type_values.append(self._current().value)
            self.cursor += 1

        # Rebuild the annotation without internal padding spaces
        param_type_name: str = self._join_type_values(type_values=type_values)

        # Consume the closing parenthesis
        self._expect_punctuation(")")

        # Consume the arrow (tokenized as an operator, not punctuation)
        arrow_token: Token = self._current()

        if not (arrow_token.token_type == TokenType.OPERATOR and arrow_token.value == "->"):
            raise SyntaxError(
                f"Expected '->' in lambda but found '{arrow_token.value}' "
                f"at index {arrow_token.position}"
            )

        self.cursor += 1

        body: Expression = self._parse_or()

        return LambdaExpr(
            param_name=param_token.value,
            body=body,
            param_type_name=param_type_name
        )

    def _join_type_values(self, type_values: list[str]) -> str:
        """
        Joins type annotation tokens into a compact raw string.

        Args:
            type_values (list[str]): Token values of the annotation.

        Returns:
            str: Reconstructed annotation (e.g. 'Array[Human]').
        """

        joined: str = ""

        for value in type_values:
            is_closing: bool = value in ("])")
            is_opening: bool = value == "["

            if joined and not is_closing and not is_opening and joined[-1] not in "[(":
                joined += " "

            joined += value

        return joined

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

        # Dict and set literals (disambiguated after the first element)
        if token.token_type == TokenType.PUNCTUATION and token.value == "{":
            return self._parse_brace_literal()

        raise SyntaxError(
            f"Unexpected token '{token.value}' at index {token.position}"
        )

    def _parse_index_item(self) -> Expression:
        """
        Parses one index operand: an expression or a 'start:stop' slice.

        Slice bounds are optional ('[:2]', '[1:]', '[:]' are valid);
        steps are not supported.

        Returns:
            Expression: The parsed index or SliceOperand.

        Raises:
            SyntaxError: If brackets contain neither an expression nor a
                slice (e.g. '[]' without contents is still handled
                elsewhere; here only ':' forms are slice candidates).
        """

        start: Optional[Expression] = None

        # Leading colon: slice from the beginning
        if not self._check_punctuation(":"):
            start = self._parse_or()

        # No colon: a plain index expression
        if not self._match_punctuation(":"):

            if start is None:
                raise SyntaxError("Expected index expression inside brackets")

            return start

        # Slice: the stop bound is optional before ',' or ']'
        stop: Optional[Expression] = None

        if not (
            self._check_punctuation(",") or self._check_punctuation("]")
        ):
            stop = self._parse_or()

        return SliceOperand(start=start, stop=stop)

    def _parse_array_literal(self) -> ArrayLiteralExpr:
        """
        Parses an array literal '[element, element, ...]'.

        Returns:
            ArrayLiteralExpr: Parsed array literal node.
        """

        # Consume the opening bracket
        self._expect_punctuation("[")

        # Parse comma separated elements, tolerating a trailing comma
        elements: list[Expression] = []

        while True:

            # Closing bracket ends the literal
            if self._match_punctuation("]"):
                break

            elements.append(self._parse_or())

            # Elements are comma separated; a trailing comma may precede ']'
            if not self._match_punctuation(","):
                self._expect_punctuation("]")
                break

            if self._match_punctuation("]"):
                break

        return ArrayLiteralExpr(elements=elements)

    def _parse_brace_literal(self) -> Expression:
        """
        Parses a brace literal: a dict '{key: value, ...}' or a set
        '{element, element, ...}', disambiguated after the first element.

        An empty '{}' is a dict literal (Python convention); empty sets are
        written via a declared type ('Set[Str] tags = {};').

        Returns:
            Expression: Parsed DictLiteralExpr or SetLiteralExpr node.

        Raises:
            SyntaxError: On a malformed first element (neither a dict key
                followed by ':' nor a set element).
        """

        # Consume the opening brace
        self._expect_punctuation("{")

        # Empty dict fast path
        if self._match_punctuation("}"):
            return DictLiteralExpr(entries=[])

        # Parse the first element, then look for the dict / set discriminator
        first: Expression = self._parse_or()

        if self._match_punctuation(":"):
            return self._parse_dict_entries(first_key=first)

        if self._match_punctuation(","):
            return self._parse_set_entries(first_element=first)

        if self._match_punctuation("}"):
            # Singleton set: '{element}'
            return SetLiteralExpr(elements=[first])

        token: Token = self._current()

        raise SyntaxError(
            f"Expected ':', ',', or '}}' in brace literal at index {token.position}"
        )

    def _parse_dict_entries(self, first_key: Expression) -> DictLiteralExpr:
        """
        Parses remaining dict entries after the first key and its ':'.

        Args:
            first_key (Expression): Already parsed first key.

        Returns:
            DictLiteralExpr: Parsed dict literal node.
        """

        first_value: Expression = self._parse_or()

        entries: list[tuple[Expression, Expression]] = [(first_key, first_value)]

        while True:

            # Continue on comma, else close the dict
            if not self._match_punctuation(","):
                self._expect_punctuation("}")
                break

            # Tolerate a trailing comma before the closing brace
            if self._match_punctuation("}"):
                break

            key: Expression = self._parse_or()

            self._expect_punctuation(":")

            value: Expression = self._parse_or()

            entries.append((key, value))

        return DictLiteralExpr(entries=entries)

    def _parse_set_entries(self, first_element: Expression) -> SetLiteralExpr:
        """
        Parses remaining set elements after the first element and its ','.

        Args:
            first_element (Expression): Already parsed first element.

        Returns:
            SetLiteralExpr: Parsed set literal node.
        """

        elements: list[Expression] = [first_element]

        while True:

            # Closing brace ends the literal
            if self._match_punctuation("}"):
                break

            elements.append(self._parse_or())

            # Elements are comma separated; a trailing comma may precede '}'
            if not self._match_punctuation(","):
                self._expect_punctuation("}")
                break

            if self._match_punctuation("}"):
                break

        return SetLiteralExpr(elements=elements)

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
                self._expect_punctuation(")")
                break

            # Tolerate a trailing comma before the closing parenthesis
            if self._match_punctuation(")"):
                break

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

    def _check_punctuation(self, punctuation: str) -> bool:
        """
        Returns whether the current token is the given punctuation
        without consuming it.

        Args:
            punctuation (str): Punctuation value to peek for.

        Returns:
            bool: True if the current token matches.
        """

        token: Token = self._current()

        return (
            token.token_type == TokenType.PUNCTUATION
            and token.value == punctuation
        )

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
