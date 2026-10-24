"""
Nadata package entry point.

Re-exports the high level Nadata facade and the CLI main function so the
library can be imported (import nadata) and the console script installed
(nadata = nadata.cli:main) from a single package.
"""

# Import Modules
from nadata.api import Nadata
from nadata.cli import main

__all__ = ["Nadata", "main", "__version__"]

__version__ = "0.1.0"
