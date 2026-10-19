"""
Nadata package entry point.

Re-exports the high level Nadata facade and the CLI main function so the
library can be imported (import nadata) and the console script installed
(nadata = nadata:main) from a single module.
"""

# Import Modules
import sys

from src.api import Nadata
from src.cli import main

__all__ = ["Nadata", "main", "__version__"]

__version__ = "0.1.0"

if __name__ == "__main__":
    sys.exit(main())
