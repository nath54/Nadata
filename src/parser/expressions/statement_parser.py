"""
Recursive descent statement parser for Nadata method and function bodies.

Converts a token stream into a list of src.core.expressions Statement nodes.
Supported statements: typed and untyped variable declarations, assignments,
if / else blocks, returns, and expression statements. Semicolons are optional
before a closing brace.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.core.expressions.statement import (
    AssignStmt,
    BlockStmt,
    ExprStmt,
    IfStmt,
    ReturnStmt,
    Statement,
    VarDeclStmt,
)
from src.core.expressions.expression import Expression
from src.parser.expressions.expression_parser import ExpressionParser
from src.parser.expressions.tokenizer import Token, TokenType, Tokenizer


class StatementParser:
    """
    Parses a token stream into a list of Statement AST nodes.
    """

    def __init__(self, tokens: list[Token], raw_text: str = "") -> None:
        """
        Initializes the StatementParser instance.

        Args:
            tokens (list[Token]): Token stream to parse (terminated by EOF).
            raw_text (str, optional): Raw source text the tokens come from,
                used to slice type annotation spans. Defaults to "".
        """

        self.tokens: list[Token] = tokens
        self.raw_text: str = raw_text
        self.cursor: int = 0

        # Shared expression parser over the same token stream
        self.expression_parser: ExpressionParser = ExpressionParser(tokens=tokens)

    @classmethod
    def parse_body(cls, text: str) -> list[Statement]:
        """
        Tokenizes and parses a raw body text in one step.

        Args:
            text (str): Raw statement sequence source text.

        Returns:
            list[Statement]: Parsed statements in source order.

        Raises:
            SyntaxError: On malformed statements.
        """

        tokens: list[Token] = Tokenizer(content=text).tokenize()

        return cls(tokens=tokens, raw_text=text).parse_statements()

    def parse_statements(self) -> list[Statement]:
        """
        Parses statements until the end of the token stream.

        Returns:
            list[Statement]: Parsed statements in source order.

        Raises:
            SyntaxError: On unexpected trailing tokens.
        """

        statements: list[Statement] = []

        # Parse until the EOF sentinel
        while self._current().token_type != TokenType.EOF:

            # Tolerate stray semicolons between statements
            if self._match_punctuation(";"):
                continue

            statements.append(self._parse_statement())

        return statements

    # --- Statement dispatch ---

    def _parse_statement(self) -> Statement:
        """
        Parses a single statement at the current cursor.

        Returns:
            Statement: The parsed statement.

        Raises:
            SyntaxError: If no statement can start at the cursor.
        """

        token: Token = self._current()

        # Return statement
        if token.token_type == TokenType.KEYWORD and token.value == "return":
            return self._parse_return()

        # If / else block
        if token.token_type == TokenType.KEYWORD and token.value == "if":
            return self._parse_if()

        # Declaration or assignment (both contain a top-level '=')
        declaration: Optional[Statement] = self._try_parse_declaration()

        if declaration is not None:
            return declaration

        # Any other expression, evaluated for its side effects
        return self._parse_expression_statement()

    def _parse_return(self) -> ReturnStmt:
        """
        Parses a return statement with an optional value.

        Returns:
            ReturnStmt: The parsed return statement.
        """

        # Consume the 'return' keyword
        self.cursor += 1

        # Check whether a value expression follows
        if self._at_statement_end():
            return ReturnStmt()

        value: Expression
        value, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._consume_statement_end()

        return ReturnStmt(value_expr=value)

    def _parse_if(self) -> IfStmt:
        """
        Parses an if statement with an optional else block.

        Returns:
            IfStmt: The parsed if statement.

        Raises:
            SyntaxError: On missing condition, body, or malformed else.
        """

        # Consume the 'if' keyword
        self.cursor += 1

        # Parse the parenthesized condition
        self._expect_punctuation("(")

        condition: Expression
        condition, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._expect_punctuation(")")

        # Parse the then block
        then_block: BlockStmt = self._parse_block()

        # Parse the optional else block
        else_block: Optional[BlockStmt] = None

        if (
            self._current().token_type == TokenType.KEYWORD
            and self._current().value == "else"
        ):
            self.cursor += 1
            else_block = self._parse_block()

        return IfStmt(condition=condition, then_block=then_block, else_block=else_block)

    def _try_parse_declaration(self) -> Optional[Statement]:
        """
        Attempts to parse a declaration or assignment at the cursor.

        Detects the top-level '=' separating the left hand side from the
        initializer, then inspects the tokens before it: a single identifier
        is an assignment, a type annotation followed by an identifier is a
        typed declaration.

        Returns:
            Optional[Statement]: The parsed statement, or None if no top-level
                '=' exists (the caller falls back to an expression statement).

        Raises:
            SyntaxError: On unsupported assignment targets.
        """

        equals_offset: int = self._find_top_level_equals()

        # No top-level '=': not a declaration nor an assignment
        if equals_offset == -1:
            return None

        lhs_tokens: list[Token] = self.tokens[self.cursor:self.cursor + equals_offset]

        # Colon form: 'name: Type = ...' (annotation starts after ':')
        if len(lhs_tokens) >= 3 and lhs_tokens[1].value == ":":
            if lhs_tokens[0].token_type != TokenType.IDENTIFIER:
                raise SyntaxError(
                    f"Unsupported declaration name '{lhs_tokens[0].value}' "
                    f"at index {lhs_tokens[0].position}"
                )

            # Move the cursor onto the '=' before consuming it
            self.cursor += equals_offset

            return self._finish_declaration(
                target_name=lhs_tokens[0].value,
                type_name=self._slice_text(
                    start_token=lhs_tokens[2],
                    end_token=lhs_tokens[-1]
                )
            )

        # Single identifier: plain assignment
        if len(lhs_tokens) == 1:
            if lhs_tokens[0].token_type != TokenType.IDENTIFIER:
                raise SyntaxError(
                    f"Unsupported assignment target '{lhs_tokens[0].value}' "
                    f"at index {lhs_tokens[0].position}"
                )

            # Move the cursor onto the '=' before consuming it
            self.cursor += equals_offset

            return self._finish_assignment(target_name=lhs_tokens[0].value)

        # Type annotation followed by an identifier: typed declaration
        last_token: Token = lhs_tokens[-1]
        before_last: Token = lhs_tokens[-2]

        is_declaration: bool = (
            last_token.token_type == TokenType.IDENTIFIER
            and (
                before_last.token_type == TokenType.IDENTIFIER
                or before_last.value == "]"
            )
        )

        # Anything else is not a valid left hand side
        if not is_declaration:
            raise SyntaxError(
                f"Unsupported assignment target before '=' at index {last_token.position}"
            )

        # Move the cursor onto the '=' before consuming it
        self.cursor += equals_offset

        # Rebuild the raw type annotation from the source span
        type_name: str = self._slice_text(
            start_token=lhs_tokens[0],
            end_token=lhs_tokens[-2]
        )

        return self._finish_declaration(
            target_name=last_token.value,
            type_name=type_name
        )

    def _finish_assignment(self, target_name: str) -> AssignStmt:
        """
        Consumes '=' and the initializer of a plain assignment.

        Args:
            target_name (str): Name of the assigned variable.

        Returns:
            AssignStmt: The parsed assignment statement.
        """

        # Consume the '=' (guaranteed by the caller)
        self.cursor += 1

        value: Expression
        value, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._consume_statement_end()

        return AssignStmt(target_name=target_name, value_expr=value)

    def _finish_declaration(
        self,
        target_name: str,
        type_name: Optional[str]
    ) -> VarDeclStmt:
        """
        Consumes '=' and the initializer of a typed declaration.

        Args:
            target_name (str): Name of the declared variable.
            type_name (Optional[str]): Raw type annotation name.

        Returns:
            VarDeclStmt: The parsed declaration statement.
        """

        # Consume the '=' (guaranteed by the caller)
        self.cursor += 1

        value: Expression
        value, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._consume_statement_end()

        return VarDeclStmt(
            name=target_name,
            init_expr=value,
            var_type_name=type_name
        )

    def _parse_expression_statement(self) -> ExprStmt:
        """
        Parses a bare expression evaluated for its side effects.

        Returns:
            ExprStmt: The parsed expression statement.
        """

        expression: Expression
        expression, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._consume_statement_end()

        return ExprStmt(expression=expression)

    def _parse_block(self) -> BlockStmt:
        """
        Parses a '{ ... }' statement block.

        Returns:
            BlockStmt: The parsed block.

        Raises:
            SyntaxError: On missing or unbalanced braces.
        """

        # Consume the opening brace
        self._expect_punctuation("{")

        statements: list[Statement] = []

        # Parse until the closing brace
        while not (
            self._current().token_type == TokenType.PUNCTUATION
            and self._current().value == "}"
        ):

            # Bail out on unexpected end of stream
            if self._current().token_type == TokenType.EOF:
                raise SyntaxError(
                    f"Missing '}}' before end of body at index {self._current().position}"
                )

            # Tolerate stray semicolons between statements
            if self._match_punctuation(";"):
                continue

            statements.append(self._parse_statement())

        # Consume the closing brace
        self.cursor += 1

        return BlockStmt(statements=statements)

    # --- Token stream helpers ---

    def _current(self) -> Token:
        """
        Returns the token at the current cursor.

        Returns:
            Token: The current token.
        """

        return self.tokens[self.cursor]

    def _at_statement_end(self) -> bool:
        """
        Checks whether the cursor sits at a statement terminator.

        Returns:
            bool: True if the current token is ';', '}', or EOF.
        """

        token: Token = self._current()

        if token.token_type == TokenType.EOF:
            return True

        return (
            token.token_type == TokenType.PUNCTUATION
            and token.value in (";", "}")
        )

    def _consume_statement_end(self) -> None:
        """
        Consumes an optional ';' statement terminator.

        The semicolon is optional before a closing brace or the end of body.
        """

        if (
            self._current().token_type == TokenType.PUNCTUATION
            and self._current().value == ";"
        ):
            self.cursor += 1

    def _find_top_level_equals(self) -> int:
        """
        Finds the offset of the next top-level '=' relative to the cursor.

        Returns:
            int: Token offset of the '=' operator, or -1 if none before EOF.
        """

        depth: int = 0

        # Single left-to-right scan from the cursor
        for offset in range(self.cursor, len(self.tokens)):
            token: Token = self.tokens[offset]

            # Stop at the EOF sentinel
            if token.token_type == TokenType.EOF:
                return -1

            # Track bracket nesting
            if token.value in "([{":
                depth += 1

            elif token.value in ")]}":
                depth -= 1

            # Top-level '=' found
            elif depth == 0 and token.value == "=":
                return offset - self.cursor

        return -1

    def _slice_text(
        self,
        start_token: Token,
        end_token: Token
    ) -> str:
        """
        Slices the raw source text spanned by two tokens.

        Args:
            start_token (Token): First token of the span.
            end_token (Token): Last token of the span.

        Returns:
            str: The raw source text between both tokens (inclusive).
        """

        return self.raw_text[start_token.position:end_token.end_position]

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
