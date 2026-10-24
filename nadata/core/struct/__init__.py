"""
Struct package initialization for Nadata.
"""

# Import Modules
from nadata.core.struct.struct import Struct
from nadata.core.struct.struct_method import StructMethod
from nadata.core.struct.struct_attribute import StructAttribute

__all__: list[str] = [
    "Struct",
    "StructMethod",
    "StructAttribute",
]
