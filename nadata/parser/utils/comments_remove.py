"""
Comment removal utility for Nadata source files.

Provides the CommentsRemover class stripping line comments ('#') and block
comments ('\"\"\"') from .nastruct and .nadata source content, while preserving
string literals and the original character indexing.
"""

# Import Modules
from __future__ import annotations
from enum import Enum


class _ScanState(Enum):
    """
    Internal scanner states for the comment removal state machine.

    Attributes:
        NORMAL (str): Regular code context.
        STRING (str): Inside a double quoted string literal.
        LINE_COMMENT (str): Inside a '#' line comment.
        BLOCK_COMMENT (str): Inside a '\"\"\"' block comment.
    """

    NORMAL = "normal"
    STRING = "string"
    LINE_COMMENT = "line_comment"
    BLOCK_COMMENT = "block_comment"


class CommentsRemover:
    """
    Removes comments from Nadata source content.

    Comment conventions (Python-like):
        - '#' starts a line comment ending at the next newline.
        - '\"\"\"' starts a block comment ending at the next '\"\"\"'
          (may span multiple lines).

    String literals delimited by '\"' are preserved, so comment markers
    appearing inside strings are kept intact.

    The output has the exact same character count as the input: comment
    characters are replaced by spaces (newlines kept), so all indices computed
    on the cleaned content remain valid line/position references for later
    parsing passes and error reporting.
    """

    # Comment and string markers
    TRIPLE_QUOTE: str = '\"\"\"'
    QUOTE: str = '\"'
    HASH: str = "#"
    BACKSLASH: str = "\\"
    NEWLINE: str = "\n"
    SPACE: str = " "

    @staticmethod
    def remove_all_comments(content: str) -> str:
        """
        Strips all '#' and '\"\"\"' comments from the given source content.

        Args:
            content (str): Full raw text content of the file.

        Returns:
            str: Content of identical length with comments replaced by spaces.

        Raises:
            SyntaxError: If a string literal or block comment is unterminated.
        """

        result: list[str] = list(content)
        state: _ScanState = _ScanState.NORMAL

        cursor: int = 0
        length: int = len(content)
        string_start: int = -1

        # Single left-to-right scan dispatching on the current state
        while cursor < length:
            char: str = content[cursor]

            # Regular code context: detect comment and string openers
            if state == _ScanState.NORMAL:
                if content.startswith(CommentsRemover.TRIPLE_QUOTE, cursor):
                    result[cursor:cursor + 3] = [CommentsRemover.SPACE] * 3
                    state = _ScanState.BLOCK_COMMENT
                    cursor += 3
                    continue

                if char == CommentsRemover.QUOTE:
                    string_start = cursor
                    state = _ScanState.STRING
                    cursor += 1
                    continue

                if char == CommentsRemover.HASH:
                    result[cursor] = CommentsRemover.SPACE
                    state = _ScanState.LINE_COMMENT
                    cursor += 1
                    continue

                cursor += 1
                continue

            # String context: preserve content, honor backslash escapes
            if state == _ScanState.STRING:
                if char == CommentsRemover.BACKSLASH:
                    cursor += 2
                    continue

                if char == CommentsRemover.QUOTE:
                    state = _ScanState.NORMAL

                cursor += 1
                continue

            # Line comment context: blank out until the newline
            if state == _ScanState.LINE_COMMENT:
                if char == CommentsRemover.NEWLINE:
                    state = _ScanState.NORMAL
                    cursor += 1
                    continue

                result[cursor] = CommentsRemover.SPACE
                cursor += 1
                continue

            # Block comment context: blank out until the closing triple quote
            if content.startswith(CommentsRemover.TRIPLE_QUOTE, cursor):
                result[cursor:cursor + 3] = [CommentsRemover.SPACE] * 3
                state = _ScanState.NORMAL
                cursor += 3
                continue

            # Preserve newlines inside block comments for line bookkeeping
            if char != CommentsRemover.NEWLINE:
                result[cursor] = CommentsRemover.SPACE

            cursor += 1

        # Detect unterminated constructs
        if state == _ScanState.STRING:
            raise SyntaxError(f"Unterminated string literal starting at index {string_start}")

        if state == _ScanState.BLOCK_COMMENT:
            raise SyntaxError("Unterminated block comment (missing closing '\"\"\"')")

        return "".join(result)
