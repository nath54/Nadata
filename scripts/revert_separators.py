"""
One-shot migration script: reverts ';' argument separators back to ',' inside
parentheses across all example .nastruct and .nadata files.
"""

# Import Modules
from __future__ import annotations
from pathlib import Path
import glob


def convert(content: str) -> str:
    """
    Replaces ';' with ',' when the innermost enclosing bracket is '('.

    Args:
        content (str): Full file content.

    Returns:
        str: Converted content.
    """

    out: list[str] = list(content)
    stack: list[str] = []
    i: int = 0
    n: int = len(content)

    while i < n:
        c: str = content[i]

        # Skip strings (with escapes)
        if c == '"':
            i += 1
            while i < n:
                if content[i] == "\\":
                    i += 2
                    continue
                if content[i] == '"':
                    break
                i += 1
            i += 1
            continue

        # Skip line comments
        if c == "#":
            while i < n and content[i] != "\n":
                i += 1
            continue

        # Skip block comments
        if content.startswith('"""', i):
            i += 3
            while i < n and not content.startswith('"""', i):
                i += 1
            i += 3
            continue

        # Track bracket nesting
        if c in "([{":
            stack.append(c)
        elif c in ")]}" and stack:
            stack.pop()
        elif c == ";" and stack and stack[-1] == "(":
            out[i] = ","

        i += 1

    return "".join(out)


def main() -> None:
    """
    Converts all example files.
    """

    paths: list[str] = (
        glob.glob("examples/**/*.nastruct", recursive=True)
        + glob.glob("examples/**/*.nadata", recursive=True)
    )

    for path in paths:
        src: str = Path(path).read_text(encoding="utf-8")
        Path(path).write_text(convert(src), encoding="utf-8", newline="")
        print(f"converted {path}")


if __name__ == "__main__":
    main()
