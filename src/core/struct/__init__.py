"""
Struct package initialization for Nadata.
"""

# Import Modules
from src.core.struct.struct import Struct
from src.core.struct.struct_method import StructMethod
from src.core.struct.struct_attribute import StructAttribute

__all__: list[str] = [
    "Struct",
    "StructMethod",
    "StructAttribute",
]
