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
    BreakStmt,
    ContinueStmt,
    ExprStmt,
    ForInStmt,
    ForStmt,
    IfStmt,
    ReturnStmt,
    Statement,
    TargetAssignStmt,
    VarDeclStmt,
    WhileStmt,
)
from src.core.expressions.expression import (
    BinaryOpExpr,
    Expression,
    IdentifierExpr,
    IndexExpr,
    LiteralExpr,
    MemberAccessExpr,
)
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

        # Nesting depth of loop bodies, for break / continue validation
        self.loop_depth: int = 0

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

        # While loop
        if token.token_type == TokenType.KEYWORD and token.value == "while":
            return self._parse_while()

        # For loops (for-in and C-style)
        if token.token_type == TokenType.KEYWORD and token.value == "for":
            return self._parse_for()

        # Break statement
        if token.token_type == TokenType.KEYWORD and token.value == "break":
            return self._parse_break()

        # Continue statement
        if token.token_type == TokenType.KEYWORD and token.value == "continue":
            return self._parse_continue()

        # Postfix increment / decrement and compound assignment statements
        # ('count++;', 'total += 2;', ...)
        if (
            token.token_type == TokenType.IDENTIFIER
            and self.tokens[self.cursor + 1].token_type == TokenType.OPERATOR
            and self.tokens[self.cursor + 1].value
            in ("++", "--", "+=", "-=", "*=", "/=", "%=")
        ):
            update: Statement = self._parse_for_update_entry()

            self._consume_statement_end()

            return update

        # Compound assignment / increment on a member or index target
        # ('self.count += 1;', 'arr[i]++;', ...)
        if (
            token.token_type == TokenType.IDENTIFIER
            and self.tokens[self.cursor + 1].value in (".", "[")
            and self._find_top_level_compound_op() != -1
        ):
            target: Expression
            target, self.cursor = self._parse_assignment_target()

            update = self._parse_for_update_entry_from_target(target=target)

            self._consume_statement_end()

            return update

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
        Parses an if statement with an optional else / elif chain.

        Returns:
            IfStmt: The parsed if statement.

        Raises:
            SyntaxError: On missing condition, body, or malformed else.
        """

        # Consume the 'if' keyword
        self.cursor += 1

        return self._parse_if_body()

    def _parse_if_body(self) -> IfStmt:
        """
        Parses the condition, body, and optional else / elif of an if whose
        leading keyword is already consumed.

        Returns:
            IfStmt: The parsed if statement.

        Raises:
            SyntaxError: On missing condition, body, or malformed else.
        """

        # Parse the parenthesized condition
        self._expect_punctuation("(")

        condition: Expression
        condition, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._expect_punctuation(")")

        # Parse the then block
        then_block: BlockStmt = self._parse_block()

        # Parse the optional else block ('elif' chains recurse here)
        else_block: Optional[BlockStmt] = None

        next_token: Token = self._current()

        if next_token.token_type == TokenType.KEYWORD and next_token.value == "else":
            self.cursor += 1

            # 'else if (cond) { ... }' is sugar for a nested if in the else
            if (
                self._current().token_type == TokenType.KEYWORD
                and self._current().value == "if"
            ):
                else_block = BlockStmt(statements=[self._parse_if_body()])

            else:
                else_block = self._parse_block()

        elif next_token.token_type == TokenType.KEYWORD and next_token.value == "elif":
            self.cursor += 1

            else_block = BlockStmt(statements=[self._parse_if_body()])

        return IfStmt(condition=condition, then_block=then_block, else_block=else_block)

    def _parse_while(self) -> WhileStmt:
        """
        Parses a while loop with its condition and body block.

        Returns:
            WhileStmt: The parsed while loop.

        Raises:
            SyntaxError: On missing condition or body.
        """

        # Consume the 'while' keyword
        self.cursor += 1

        # Parse the parenthesized condition
        self._expect_punctuation("(")

        condition: Expression
        condition, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._expect_punctuation(")")

        self.loop_depth += 1

        try:
            body: BlockStmt = self._parse_block()
        finally:
            self.loop_depth -= 1

        return WhileStmt(condition=condition, body=body)

    def _parse_for(self) -> Statement:
        """
        Parses a for loop, detecting the for-in and C-style forms.

        Returns:
            Statement: The parsed ForInStmt or ForStmt.

        Raises:
            SyntaxError: On malformed headers.
        """

        # Consume the 'for' keyword
        self.cursor += 1

        self._expect_punctuation("(")

        kind: str = self._detect_for_kind()

        if kind == "in":
            return self._parse_for_in()

        return self._parse_c_style_for()

    def _detect_for_kind(self) -> str:
        """
        Detects whether the for header is a for-in or a C-style loop.

        Scans forward from the cursor for the first top-level 'in' keyword or
        '=' operator: 'in' means for-in, '=' means C-style (its init always
        declares with an initializer).

        Returns:
            str: 'in' or 'cstyle'.

        Raises:
            SyntaxError: If neither marker appears before the closing paren.
        """

        depth: int = 0

        for offset in range(self.cursor, len(self.tokens)):
            token: Token = self.tokens[offset]

            if token.token_type == TokenType.EOF:
                break

            if token.token_type == TokenType.PUNCTUATION and token.value in "([{":
                depth += 1

            elif token.token_type == TokenType.PUNCTUATION and token.value in ")]}":
                if depth == 0 and token.value == ")":
                    break

                depth -= 1

            elif (
                depth == 0
                and token.token_type == TokenType.KEYWORD
                and token.value == "in"
            ):
                return "in"

            elif depth == 0 and token.token_type == TokenType.OPERATOR and token.value == "=":
                return "cstyle"

            elif depth == 0 and token.token_type == TokenType.OPERATOR and token.value == "::":
                # No '=' before the separator: C-style with a missing
                # initializer (the init parser reports the exact error)
                return "cstyle"

        raise SyntaxError(
            f"Malformed for header: expected 'for (x: Type in iterable)' or "
            f"'for (x: Type = init :: condition :: update)' near index "
            f"{self._current().position}"
        )

    def _parse_for_in(self) -> ForInStmt:
        """
        Parses a for-in loop: 'for (x: Type in iterable) { ... }'.

        Returns:
            ForInStmt: The parsed loop.

        Raises:
            SyntaxError: On malformed header or body.
        """

        # Loop variable name
        name_token: Token = self._current()

        if name_token.token_type != TokenType.IDENTIFIER:
            raise SyntaxError(
                f"Expected loop variable name at index {name_token.position}"
            )

        self.cursor += 1

        # Type annotation
        self._expect_punctuation(":")

        # Raw type annotation: slice up to the 'in' keyword
        in_offset: int = self._find_top_level_keyword(keyword="in")

        if in_offset == -1:
            raise SyntaxError(
                f"Missing 'in' in for loop header at index {self._current().position}"
            )

        type_name: str = self._slice_text(
            start_token=self.tokens[self.cursor],
            end_token=self.tokens[self.cursor + in_offset - 1],
        ).strip()

        # Skip the type tokens and the 'in' keyword
        self.cursor += in_offset + 1

        iterable: Expression
        iterable, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._expect_punctuation(")")

        self.loop_depth += 1

        try:
            body: BlockStmt = self._parse_block()
        finally:
            self.loop_depth -= 1

        return ForInStmt(
            var_name=name_token.value,
            var_type=None,
            var_type_name=type_name,
            iterable=iterable,
            body=body,
        )

    def _parse_c_style_for(self) -> ForStmt:
        """
        Parses a C-style loop:
        'for (x: T = init, y: T = init :: condition :: update, update) { ... }'.

        Returns:
            ForStmt: The parsed loop.

        Raises:
            SyntaxError: On malformed header or body.
        """

        # Init declarations (comma separated, all typed with initializers)
        init_statements: list[Statement] = []

        while True:
            init_statements.append(self._parse_for_init_declaration())

            if self._match_operator("::"):
                break

            if self._match_punctuation(","):
                continue

            raise SyntaxError(
                f"Expected ',' or '::' in for init at index {self._current().position}"
            )

        # Condition
        condition: Expression
        condition, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        self._expect_operator("::")

        # Update statements (comma separated)
        update_statements: list[Statement] = []

        while True:
            update_statements.append(self._parse_for_update_entry())

            if self._match_punctuation(","):
                continue

            break

        self._expect_punctuation(")")

        self.loop_depth += 1

        try:
            body: BlockStmt = self._parse_block()
        finally:
            self.loop_depth -= 1

        return ForStmt(
            init_statements=init_statements,
            condition=condition,
            update_statements=update_statements,
            body=body,
        )

    def _parse_for_init_declaration(self) -> VarDeclStmt:
        """
        Parses one typed init declaration of a C-style for header.

        Returns:
            VarDeclStmt: The parsed declaration (type resolved at execution).

        Raises:
            SyntaxError: On malformed declarations or missing initializers.
        """

        name_token: Token = self._current()

        if name_token.token_type != TokenType.IDENTIFIER:
            raise SyntaxError(
                f"Expected loop variable name at index {name_token.position}"
            )

        self.cursor += 1

        self._expect_punctuation(":")

        # Raw type annotation: slice up to the mandatory '='
        equals_offset: int = self._find_top_level_operator(operator="=")

        if equals_offset == -1:
            raise SyntaxError(
                f"For init declaration of '{name_token.value}' requires an "
                f"initializer at index {self._current().position}"
            )

        type_name: str = self._slice_text(
            start_token=self.tokens[self.cursor],
            end_token=self.tokens[self.cursor + equals_offset - 1],
        ).strip()

        # Skip the type tokens and consume the '='
        self.cursor += equals_offset + 1

        init_expr: Expression
        init_expr, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        return VarDeclStmt(
            name=name_token.value,
            init_expr=init_expr,
            var_type_name=type_name,
        )

    def _parse_for_update_entry(self) -> Statement:
        """
        Parses one update entry of a C-style for header.

        Supported forms: 'x = expr', 'x += expr' (and -= *= /= %=), the
        increments 'x++' / 'x--', and the same forms over member or index
        targets ('self.count += 1', 'arr[i]++', 'self.a[0] = y'). Compound
        forms desugar into plain assignments over the existing binary
        operators.

        Returns:
            Statement: The desugared assignment statement.

        Raises:
            SyntaxError: On malformed update entries.
        """

        target: Expression
        target, self.cursor = self._parse_assignment_target()

        return self._parse_for_update_entry_from_target(target=target)

    def _parse_for_update_entry_from_target(self, target: Expression) -> Statement:
        """
        Parses the operator and right hand side of an update entry whose
        target expression has already been consumed.

        Args:
            target (Expression): The already-parsed target expression.

        Returns:
            Statement: The desugared assignment statement.

        Raises:
            SyntaxError: On malformed update entries.
        """

        token: Token = self._current()

        # Plain assignment
        if token.token_type == TokenType.OPERATOR and token.value == "=":
            self.cursor += 1

            value: Expression
            value, self.cursor = self.expression_parser.parse_expression_at(
                cursor=self.cursor
            )

            return self._target_or_plain_assign(target=target, value_expr=value)

        # Compound assignments and increments desugar into binary operations
        binary_op: Optional[str] = None

        if token.token_type == TokenType.OPERATOR and token.value in ("+=", "-=", "*=", "/=", "%="):
            binary_op = token.value[0]

        elif token.token_type == TokenType.OPERATOR and token.value == "++":
            binary_op = "+"

        elif token.token_type == TokenType.OPERATOR and token.value == "--":
            binary_op = "-"

        if binary_op is None:
            raise SyntaxError(
                f"Unsupported update entry '{token.value}' at index {token.position}"
            )

        self.cursor += 1

        # 'x++' / 'x--' imply a literal step of 1
        if token.value in ("++", "--"):
            rhs: Expression = LiteralExpr(value=1)

        else:
            rhs_parsed: Expression
            rhs_parsed, self.cursor = self.expression_parser.parse_expression_at(
                cursor=self.cursor
            )

            rhs = rhs_parsed

        return self._desugar_compound(
            target=target,
            binary_op=binary_op,
            rhs=rhs,
        )

    def _parse_assignment_target(self) -> tuple[Expression, int]:
        """
        Parses an assignment target expression at the cursor.

        Args:
            (none)

        Returns:
            tuple[Expression, int]: The target expression and the new cursor
                position (after the target).

        Raises:
            SyntaxError: If the expression at the cursor is not a valid
                assignment target.
        """

        target: Expression
        target, new_cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        if not isinstance(target, (IdentifierExpr, MemberAccessExpr, IndexExpr)):
            raise SyntaxError(
                f"Unsupported assignment target '{repr(target)}' "
                f"at index {self.cursor}"
            )

        return target, new_cursor

    def _target_or_plain_assign(
        self,
        target: Expression,
        value_expr: Expression,
    ) -> Statement:
        """
        Builds an assignment statement for a parsed target.

        Args:
            target (Expression): The parsed target expression.
            value_expr (Expression): Expression producing the new Value.

        Returns:
            Statement: AssignStmt for identifier targets, TargetAssignStmt
                for member or index targets.
        """

        if isinstance(target, IdentifierExpr):
            return AssignStmt(target_name=target.name, value_expr=value_expr)

        return TargetAssignStmt(target=target, value_expr=value_expr)

    def _desugar_compound(
        self,
        target: Expression,
        binary_op: str,
        rhs: Expression,
    ) -> Statement:
        """
        Desugars a compound assignment into an assignment over a binary op.

        Args:
            target (Expression): The assignment target expression.
            binary_op (str): The binary operator ('+', '-', '*', '/', '%').
            rhs (Expression): The right hand side expression.

        Returns:
            Statement: The desugared assignment statement.
        """

        return self._target_or_plain_assign(
            target=target,
            value_expr=BinaryOpExpr(op=binary_op, left=target, right=rhs),
        )

    def _find_top_level_compound_op(self) -> int:
        """
        Finds the offset of a top-level compound operator relative to the
        cursor, for standalone statements like 'self.count += 1;'.

        Returns:
            int: Token offset of the compound operator, or -1 if none
                before statement end.
        """

        depth: int = 0

        # Single left-to-right scan from the cursor
        for offset in range(self.cursor, len(self.tokens)):
            token: Token = self.tokens[offset]

            # Stop at the EOF sentinel
            if token.token_type == TokenType.EOF:
                return -1

            # A call argument list is never part of a target
            if depth == 0 and token.value in ("(", ")"):
                return -1

            # Track bracket nesting
            if token.value in "([{":
                depth += 1

            elif token.value in ")]}":
                depth -= 1
            elif depth == 0 and token.token_type == TokenType.OPERATOR:
                # A plain '=' means the statement is handled elsewhere
                if token.value == "=":
                    return -1

                if token.value in ("+=", "-=", "*=", "/=", "%=", "++", "--"):
                    return offset - self.cursor

        return -1

    def _parse_break(self) -> BreakStmt:
        """
        Parses a break statement.

        Returns:
            BreakStmt: The parsed statement.

        Raises:
            SyntaxError: If break appears outside of a loop body.
        """

        self.cursor += 1

        if self.loop_depth == 0:
            raise SyntaxError("'break' outside of a loop body")

        self._consume_statement_end()

        return BreakStmt()

    def _parse_continue(self) -> ContinueStmt:
        """
        Parses a continue statement.

        Returns:
            ContinueStmt: The parsed statement.

        Raises:
            SyntaxError: If continue appears outside of a loop body.
        """

        self.cursor += 1

        if self.loop_depth == 0:
            raise SyntaxError("'continue' outside of a loop body")

        self._consume_statement_end()

        return ContinueStmt()

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

    def _match_operator(self, operator: str) -> bool:
        """
        Consumes the current token if it is the given operator.

        Args:
            operator (str): Operator value to match.

        Returns:
            bool: True if the operator was matched and consumed.
        """

        token: Token = self._current()

        if token.token_type == TokenType.OPERATOR and token.value == operator:
            self.cursor += 1
            return True

        return False

    def _find_top_level_keyword(self, keyword: str) -> int:
        """
        Finds the offset of a top-level keyword relative to the cursor.

        Args:
            keyword (str): Keyword value to locate.

        Returns:
            int: Token offset of the keyword, or -1 if none before the
                closing parenthesis or EOF.
        """

        depth: int = 0

        for offset in range(self.cursor, len(self.tokens)):
            token: Token = self.tokens[offset]

            if token.token_type == TokenType.EOF:
                return -1

            if token.token_type == TokenType.PUNCTUATION and token.value in "([{":
                depth += 1

            elif token.token_type == TokenType.PUNCTUATION and token.value in ")]}":
                if depth == 0 and token.value == ")":
                    return -1

                depth -= 1

            elif (
                depth == 0
                and token.token_type == TokenType.KEYWORD
                and token.value == keyword
            ):
                return offset - self.cursor

        return -1

    def _find_top_level_operator(self, operator: str) -> int:
        """
        Finds the offset of a top-level operator relative to the cursor.

        Args:
            operator (str): Operator value to locate.

        Returns:
            int: Token offset of the operator, or -1 if none before the
                closing parenthesis, a '::' separator, or EOF.
        """

        depth: int = 0

        for offset in range(self.cursor, len(self.tokens)):
            token: Token = self.tokens[offset]

            if token.token_type == TokenType.EOF:
                return -1

            if token.token_type == TokenType.PUNCTUATION and token.value in "([{":
                depth += 1

            elif token.token_type == TokenType.PUNCTUATION and token.value in ")]}":
                if depth == 0 and token.value == ")":
                    return -1

                depth -= 1

            elif token.token_type == TokenType.OPERATOR and token.value == "::":
                if depth == 0:
                    return -1

            elif (
                depth == 0
                and token.token_type == TokenType.OPERATOR
                and token.value == operator
            ):
                return offset - self.cursor

        return -1

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

        # A leading '=' (e.g. after an already-consumed expression statement)
        # has no left hand side at all
        if not lhs_tokens:
            raise SyntaxError(
                f"Unsupported assignment target '=' at index {self.cursor}"
            )

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
        # (checked first, so 'Array[Int32] items = ...' is not mistaken
        # for an index assignment target)
        last_token: Token = lhs_tokens[-1]
        before_last: Token = lhs_tokens[-2]

        is_declaration: bool = (
            last_token.token_type == TokenType.IDENTIFIER
            and (
                before_last.token_type == TokenType.IDENTIFIER
                or before_last.value == "]"
            )
        )

        if is_declaration:
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

        # Member or index target ('self.count = x', 'arr[0] = y'): the LHS
        # starts with an identifier and contains a top-level '.' or '['
        is_target: bool = (
            lhs_tokens[0].token_type == TokenType.IDENTIFIER
            and any(token.value in (".", "[") for token in lhs_tokens)
        )

        if is_target:
            target: Expression
            target, self.cursor = self._parse_assignment_target()

            # The helper consumes the '=' before parsing the right hand side
            return self._target_or_plain_assign(
                target=target,
                value_expr=self._parse_rhs_after_equals(),
            )

        # Anything else is not a valid left hand side
        raise SyntaxError(
            f"Unsupported assignment target before '=' at index {last_token.position}"
        )

    def _parse_rhs_after_equals(self) -> Expression:
        """
        Consumes a top-level '=' and parses the right hand side expression.

        Args:
            (none)

        Returns:
            Expression: The parsed right hand side.
        """

        # Consume the '=' (guaranteed by the caller)
        self.cursor += 1

        value: Expression
        value, self.cursor = self.expression_parser.parse_expression_at(
            cursor=self.cursor
        )

        return value

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

            # A top-level parenthesis is a call argument list — never part of
            # an LHS, so the '=' belongs to a later statement
            if depth == 0 and token.value in ("(", ")"):
                return -1

            # Track bracket nesting
            if token.value in "([{":
                depth += 1

            elif token.value in ")]}":
                depth -= 1

            # Top-level '=' found
            elif depth == 0 and token.value == "=":
                return offset - self.cursor

            # A '=' only marks a declaration while the left hand side so far
            # looks like 'name', 'name: Type', or 'Type name' — anything else
            # (e.g. a call argument) means the '=' belongs to a later statement
            elif depth == 0 and not self._is_lhs_token(token=token):
                return -1

        return -1

    def _is_lhs_token(self, token: Token) -> bool:
        """
        Checks whether a token can appear in a declaration left hand side.

        Args:
            token (Token): Token to inspect.

        Returns:
            bool: True if the token fits a declaration LHS shape.
        """

        if token.token_type == TokenType.IDENTIFIER:
            return True

        return (
            token.token_type == TokenType.PUNCTUATION
            and token.value in ("[", "]", ":", ".")
        )

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
