"""
TODO
"""

# Import Methods
from typing import Optional


class BracketTree:
    """
    TODO
    """

    def __init__(
        self,
        start: int,
        end: int = -1,
        inside: Optional[list[BracketTree]] = None
    ) -> None:

        self.start: int = start
        self.end: int = end
        self.inside: list[BracketTree] = [] if inside is None else inside


class BracketsAndStrings:
    """
    TODO
    """

    @staticmethod
    def process(content: str) -> tuple[list[BracketTree], dict[int, int]]:
        """
        TODO
        """

        # Strings
        string_couples: dict[int, int] = {}

        # Waiting queue
        waiting_queue: list[BracketTree] = []

        # Larger Trees
        larger_trees: list[BracketTree] = []

        # List of chars to find
        chars: list[str] = ["{", "}", "\""]

        # Pre-compile the regex pattern for maximum speed
        pattern = re.compile(f"[{''.join(chars)}]")

        # Extract starting indices efficiently using finditer
        indices: list[int] = [match.start() for match in pattern.finditer(text)]

        # Indicate if we are inside a string or not
        in_string: bool = False
        last_string: int = -1

        # For all the characters found
        for i in indices:

            # If it is an opening bracket
            if content[i] == "{":

                # Ignore if we are inside a string
                if in_string:
                    continue

                # Else, append to the queue
                waiting_queue.append( BracketTree(start=i) )

            # If it is a closing bracket
            elif content[i] == "}":

                # Ignore if we are inside a string
                if in_string:
                    continue

                # There are too much closing brackets
                if not waiting_queue:
                    raise SyntaxError("TODO")

                # Get the corresponding opening bracket
                bt: BracketTree = waiting_queue.pop(-1)

                bt.end = i

                # This is a larger bracket, no one is parenting it
                if not waiting_queue:
                    larger_trees.append(bt)

                # Else, adding this tree to the parent tree
                else:
                    waiting_queue[-1].inside.append(bt)

                # Add the connexion to the registry
                brackets_couples[j] = i

            # If it is a string
            elif content[i] == "\n":

                # Here, we start the string
                if not in_string:
                    last_string = i

                # Here, we close the string
                else:
                    string_couples[last_string] = i

                # Toggle in string status
                in_string = not in_string

        # Check if unclosed brackets
        if len(waiting) > 1:
            raise SyntaxError("TODO")

        # Return the final results
        return larger_trees, string_couples
