"""
Callable definition parsing for .nastruct schema files.

Provides the parsers for struct method declarations and standalone function
declarations, which share the same 'name(params) -> ReturnType' header grammar.
Bodies are stored as raw text spans; statement parsing happens in the
resolution phase.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.parser.expressions.splitter import split_top_level
from src.parser.expressions.tokenizer import Token, TokenType, Tokenizer
from src.parser.utils.brackets import BracketTree
from src.parser.utils.identifiers import IdentifierExtractor
from src.parser.nastruct_parser.declaration import parse_declaration_text
from src.parser.nastruct_parser.tmp_structs import TmpAttribute, TmpFunction, TmpStruct
from src.parser.nastruct_parser.parser_state import NaStructParserState


def parse_struct_method(
    parser_state: NaStructParserState,
    struct_name: str,
    struct_method_name: str,
    header: str,
    method_start: int,
    method_end: int,
    content: str,
    brackets_inside: list[BracketTree]
) -> None:
    """
    Parses a struct method definition block.

    Args:
        parser_state (NaStructParserState): Shared parsing session state.
        struct_name (str): Name of the struct owning the method.
        struct_method_name (str): Name of the method (without '@').
        header (str): Full header text between '@' and the opening brace
            (e.g. 'age(now: DateTime) -> DateTime').
        method_start (int): Index of the first content character inside the braces.
        method_end (int): Index of the closing brace of the method body.
        content (str): Text of the file.
        brackets_inside (list[Token]): Nested bracket trees in the body.
    """

    # Parse the shared header grammar (parameters and return type)
    fn_arguments: dict[str, TmpAttribute]
    fn_output_type: str
    fn_arguments, fn_output_type = _parse_callable_header(header_text=header)

    # Extract the raw body text between the braces
    fn_body: str = content[method_start:method_end].strip()

    # Check the owning struct exists in the intermediate representation
    tmp_struct: Optional[TmpStruct] = parser_state.tmp_structs.get(struct_name)

    if tmp_struct is None:
        raise SyntaxError(f"Method '{struct_method_name}' refers to unknown struct '{struct_name}'")

    # Check the method is not already declared on the struct
    if struct_method_name in tmp_struct.methods:
        raise SyntaxError(
            f"Method '{struct_method_name}' is already defined on struct '{struct_name}'"
        )

    # Register the method in the tmp struct
    tmp_struct.methods[struct_method_name] = TmpFunction(
        fn_name=struct_method_name,
        fn_output_type=fn_output_type,
        fn_argument=fn_arguments,
        fn_body=fn_body,
        dependencies=_collect_callable_dependencies(
            fn_arguments=fn_arguments,
            fn_output_type=fn_output_type
        )
    )


def parse_fn(
    parser_state: NaStructParserState,
    name: str,
    fn_start: int,
    fn_end: int,
    brackets_inside: list[BracketTree],
    content: str
) -> None:
    """
    Parses a standalone function definition block.

    Args:
        parser_state (NaStructParserState): Shared parsing session state.
        name (str): Full header text of the function
            (e.g. 'fn is_adult(h: Human) -> Bool').
        fn_start (int): Index of the first content character inside the braces.
        fn_end (int): Index of the closing brace of the block.
        brackets_inside (list[Token]): Nested bracket trees in the body.
        content (str): Text of the file.
    """

    # Strip the 'fn' keyword from the header
    header_body: str = name[len("fn"):].strip()

    # Parse the shared header grammar (function name, parameters, return type)
    fn_name: str
    fn_arguments: dict[str, TmpAttribute]
    fn_output_type: str
    fn_name, fn_arguments, fn_output_type = _parse_fn_header(header_text=header_body)

    # Extract the raw body text between the braces
    fn_body: str = content[fn_start:fn_end].strip()

    # Check the function is not already defined
    if fn_name in parser_state.tmp_functions:
        raise SyntaxError(f"Function '{fn_name}' is already defined")

    # Register the function in the parser state
    parser_state.tmp_functions[fn_name] = TmpFunction(
        fn_name=fn_name,
        fn_output_type=fn_output_type,
        fn_argument=fn_arguments,
        fn_body=fn_body,
        dependencies=_collect_callable_dependencies(
            fn_arguments=fn_arguments,
            fn_output_type=fn_output_type
        )
    )


def _parse_fn_header(
    header_text: str
) -> tuple[str, dict[str, TmpAttribute], str]:
    """
    Parses a standalone function header into its name, parameters, and return type.

    Args:
        header_text (str): Header text without the 'fn' keyword
            (e.g. 'is_adult(h: Human) -> Bool').

    Returns:
        tuple[str, dict[str, TmpAttribute], str]: The function name, the
            parameters by name, and the raw return type annotation.

    Raises:
        SyntaxError: If the header is malformed.
    """

    # Check the header is present
    if not header_text:
        raise SyntaxError("Empty function header")

    # Locate the parameter list opening parenthesis
    paren_index: int = header_text.find("(")

    # Check the parameter list is present
    if paren_index == -1:
        raise SyntaxError(f"Missing parameter list in function header: '{header_text}'")

    # The function name is the text before the parameter list
    fn_name: str = header_text[:paren_index].strip()

    # Check the function name is a single token
    if len(fn_name.split()) != 1:
        raise SyntaxError(f"Malformed function name in header: '{header_text}'")

    # Parse the shared parameter list and return type
    fn_arguments: dict[str, TmpAttribute]
    fn_output_type: str
    fn_arguments, fn_output_type = _parse_callable_header(header_text=header_text)

    return fn_name, fn_arguments, fn_output_type


def _parse_callable_header(
    header_text: str
) -> tuple[dict[str, TmpAttribute], str]:
    """
    Parses the '(...)' parameter list and '-> ReturnType' of a callable header.

    Args:
        header_text (str): Header text containing the parameter list
            (e.g. 'age(now: DateTime) -> DateTime').

    Returns:
        tuple[dict[str, TmpAttribute], str]: The parameters by name and the
            raw return type annotation.

    Raises:
        SyntaxError: If the header is malformed.
    """

    tokens: list[Token] = Tokenizer(content=header_text).tokenize()

    # Locate the parameter list boundaries at the token level
    open_index: int
    close_index: int
    open_index, close_index = _find_param_list_boundaries(
        tokens=tokens,
        header_text=header_text
    )

    # Parse the comma separated parameter declarations
    param_tokens: list[Token] = tokens[open_index + 1:close_index]
    fn_arguments: dict[str, TmpAttribute] = _parse_parameter_tokens(
        param_tokens=param_tokens,
        header_text=header_text
    )

    # Parse the return type annotation after the closing parenthesis
    fn_output_type: str = _parse_return_type(
        tokens=tokens,
        close_index=close_index,
        header_text=header_text
    )

    return fn_arguments, fn_output_type


def _find_param_list_boundaries(
    tokens: list[Token],
    header_text: str
) -> tuple[int, int]:
    """
    Finds the token indexes of the parameter list opening and closing parens.

    Args:
        tokens (list[Token]): Tokenized header text (terminated by EOF).
        header_text (str): Original header text (for error messages).

    Returns:
        tuple[int, int]: Indexes of the '(' and matching ')' tokens.

    Raises:
        SyntaxError: If the parameter list is missing or unbalanced.
    """

    # Locate the opening parenthesis token
    open_index: int = -1

    for index, token in enumerate(tokens):

        if token.token_type == TokenType.PUNCTUATION and token.value == "(":
            open_index = index
            break

    # Check the opening parenthesis is present
    if open_index == -1:
        raise SyntaxError(f"Missing parameter list in header: '{header_text}'")

    # Scan for the matching closing parenthesis with depth tracking
    depth: int = 0

    for index in range(open_index, len(tokens)):
        loop_token: Token = tokens[index]

        if loop_token.token_type == TokenType.EOF:
            break

        if loop_token.token_type == TokenType.PUNCTUATION:
            if loop_token.value == "(":
                depth += 1

            elif loop_token.value == ")":
                depth -= 1

                # Matching closing parenthesis found
                if depth == 0:
                    return open_index, index

    raise SyntaxError(f"Unbalanced parameter list in header: '{header_text}'")


def _parse_parameter_tokens(
    param_tokens: list[Token],
    header_text: str
) -> dict[str, TmpAttribute]:
    """
    Parses the parameter declaration tokens into TmpAttribute entries.

    Args:
        param_tokens (list[Token]): Tokens between the parameter parentheses.
        header_text (str): Original header text (for error messages).

    Returns:
        dict[str, TmpAttribute]: Parameters by name.

    Raises:
        SyntaxError: On empty or duplicated parameter declarations.
    """

    fn_arguments: dict[str, TmpAttribute] = {}

    # Split on top-level commas (nested calls / defaults are kept intact)
    segments: list[list[Token]] = split_top_level(tokens=param_tokens, separator=",")

    # An empty parameter list is a single empty segment
    if len(segments) == 1 and not segments[0]:
        return fn_arguments

    for segment in segments:

        # Check the parameter declaration is not empty
        if not segment:
            raise SyntaxError(f"Empty parameter declaration in header: '{header_text}'")

        # Rebuild the raw declaration text from the source span
        param_text: str = header_text[segment[0].position:segment[-1].end_position]

        param_name: str
        param_type: str
        param_default: Optional[str]
        param_name, param_type, param_default = parse_declaration_text(
            declaration_text=param_text
        )

        # Check the parameter is not duplicated
        if param_name in fn_arguments:
            raise SyntaxError(
                f"Parameter '{param_name}' is declared twice in header: '{header_text}'"
            )

        # Collect dependencies from the type annotation and the default value
        dependencies: set[str] = IdentifierExtractor.extract_identifiers(text=param_type)

        if param_default is not None:
            dependencies |= IdentifierExtractor.extract_identifiers(text=param_default)

        fn_arguments[param_name] = TmpAttribute(
            attribute_name=param_name,
            attribute_type=param_type,
            attribute_default_value=param_default,
            dependencies=dependencies
        )

    return fn_arguments


def _parse_return_type(
    tokens: list[Token],
    close_index: int,
    header_text: str
) -> str:
    """
    Parses the '-> ReturnType' annotation following the parameter list.

    Args:
        tokens (list[Token]): Tokenized header text (terminated by EOF).
        close_index (int): Token index of the parameter list closing paren.
        header_text (str): Original header text (for slicing and errors).

    Returns:
        str: The raw return type annotation ('' if absent).

    Raises:
        SyntaxError: If the annotation arrow is present but the type is missing.
    """

    rest_tokens: list[Token] = tokens[close_index + 1:]

    # No return type annotation
    if not rest_tokens or rest_tokens[0].token_type == TokenType.EOF:
        return ""

    # The annotation must start with the '->' arrow operator
    arrow_token: Token = rest_tokens[0]

    if not (arrow_token.token_type == TokenType.OPERATOR and arrow_token.value == "->"):
        raise SyntaxError(f"Malformed return type in header: '{header_text}'")

    # Check the return type is present
    if len(rest_tokens) < 2 or rest_tokens[1].token_type == TokenType.EOF:
        raise SyntaxError(f"Missing return type after '->' in header: '{header_text}'")

    return header_text[arrow_token.end_position:].strip()


def _collect_callable_dependencies(
    fn_arguments: dict[str, TmpAttribute],
    fn_output_type: str
) -> set[str]:
    """
    Collects the dependency identifiers of a callable declaration.

    Args:
        fn_arguments (dict[str, TmpAttribute]): Parameters by name.
        fn_output_type (str): Raw return type annotation.

    Returns:
        set[str]: Union of the parameter and return type dependencies.
    """

    dependencies: set[str] = set()

    # Union the dependencies of every parameter
    for fn_argument in fn_arguments.values():
        dependencies |= fn_argument.dependencies

    # Add the return type dependencies
    if fn_output_type:
        dependencies |= IdentifierExtractor.extract_identifiers(text=fn_output_type)

    return dependencies
