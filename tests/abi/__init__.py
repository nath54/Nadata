"""
ABI conformance support package.

Contains the machine-checkable contract shared by every native binding
(`contract.py`) and reusable engine scenario builders (`fixtures.py`).

The Python suite in `tests/test_abi_conformance.py` is the canonical
definition of the ABI behaviors; bindings in other languages replay the
same case ids against their own wrappers.
"""
