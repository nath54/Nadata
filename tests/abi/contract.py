"""
The Nadata native ABI contract, expressed in Python.

Mirrors `native/nadata.h` (the single source of truth) so the test
suite can assert engine behaviors against exact status codes. A sync
test (`tests/test_abi_header_sync.py`) fails whenever this module and
the header disagree.

Three exports matter to binding authors:

- `NdStatus` — the status code enum, value-for-value identical to the
  header's `NdStatusCode`.
- `status_for(exception)` — the canonical Python-exception → status
  mapping the shim implements. Bindings re-raise a native error of the
  same class.
- `abi_case` — decorator stamping each conformance test with a stable
  case id (`"nd_call"/"scenario"`). Other languages replay these ids
  so coverage can be diffed across bindings.
"""

# Import Modules
from enum import IntEnum
from typing import Callable, TypeVar


class NdStatus(IntEnum):
    """
    Status codes returned by every fallible nd_* function.

    Keep value-for-value in sync with NdStatusCode in native/nadata.h
    (enforced by tests/test_abi_header_sync.py).
    """

    ND_OK = 0

    # Error classes — each maps 1:1 to a Python exception family.
    ND_ERR_NOTFOUND = -1   # KeyError        — unknown struct name
    ND_ERR_NAME = -2       # NameError       — unknown symbol/identifier
    ND_ERR_VALUE = -3      # ValueError      — bad .ndb, missing file, ...
    ND_ERR_TYPE = -4       # TypeError       — argument/attribute mismatch
    ND_ERR_SYNTAX = -5     # SyntaxError     — schema resolve-time errors
    ND_ERR_IO = -6         # OSError family  — filesystem failures
    ND_ERR_LIMIT = -7      # LoopLimitError  — per-loop fuel cap exceeded
    ND_ERR_PERMISSION = -8  # PermissionError — sandboxed I/O traversal
    ND_ERR_RUNTIME = -9    # RuntimeError    — engine failure at run time

    # Shim-level errors (raised before Python is involved).
    ND_ERR_INVALID_ARG = -10  # NULL handle, NULL required string, bad tag
    ND_ERR_STATE = -11        # handle used after free / wrong op
    ND_ERR_MEMORY = -12       # allocation failure inside the shim


# Canonical exception -> status mapping, most specific first. The
# lookup walks the exception's MRO, so subclasses (PermissionError ->
# OSError) resolve to the most precise entry automatically.
EXCEPTION_STATUS_MAP: dict[type[BaseException], NdStatus] = {
    KeyError: NdStatus.ND_ERR_NOTFOUND,
    AttributeError: NdStatus.ND_ERR_NOTFOUND,  # unknown attribute (engine convention)
    NameError: NdStatus.ND_ERR_NAME,
    ValueError: NdStatus.ND_ERR_VALUE,
    TypeError: NdStatus.ND_ERR_TYPE,
    SyntaxError: NdStatus.ND_ERR_SYNTAX,
    PermissionError: NdStatus.ND_ERR_PERMISSION,
    OSError: NdStatus.ND_ERR_IO,
    IndexError: NdStatus.ND_ERR_NOTFOUND,
    ZeroDivisionError: NdStatus.ND_ERR_RUNTIME,
    OverflowError: NdStatus.ND_ERR_RUNTIME,
    RuntimeError: NdStatus.ND_ERR_RUNTIME,
    MemoryError: NdStatus.ND_ERR_MEMORY,
}

# Late import to avoid a circular dependency: the engine module does
# not import tests, and this map documents the contract without it.
from nadata.core.expressions.scope import LoopLimitError  # noqa: E402

EXCEPTION_STATUS_MAP[LoopLimitError] = NdStatus.ND_ERR_LIMIT


def status_for(exception: BaseException) -> NdStatus:
    """
    Resolves the ABI status code for an engine exception.

    Walks the exception's MRO and returns the status of the most
    specific mapped class. Unmapped exceptions fall back to
    ND_ERR_RUNTIME — the shim must never crash on an unknown type.

    Args:
        exception (BaseException): The raised engine exception.

    Returns:
        NdStatus: The ABI status code to return to the caller.
    """

    for cls in type(exception).__mro__:
        if cls in EXCEPTION_STATUS_MAP:
            return EXCEPTION_STATUS_MAP[cls]

    return NdStatus.ND_ERR_RUNTIME


# Case id decorator machinery.
F = TypeVar("F", bound=Callable[..., object])

_REGISTRY: set[str] = set()


class AbiCaseDuplicateError(Exception):
    """Two conformance tests claimed the same ABI case id."""


def abi_case(case_id: str) -> Callable[[F], F]:
    """
    Stamps a conformance test with its stable ABI case id.

    Case ids are `"nd_function/scenario"` strings shared across every
    language binding: the Kotlin/Rust/JS suites replay exactly these
    ids so per-binding coverage is diffable. Ids are registered at
    decoration time; duplicates fail fast at import, not at CI review.

    Args:
        case_id (str): Stable id, e.g. "nd_query_first_id/empty_set".

    Returns:
        Callable: The decorated function, carrying `__abi_case__`.
    """

    if case_id in _REGISTRY:
        raise AbiCaseDuplicateError(f"Duplicate ABI case id: {case_id}")

    _REGISTRY.add(case_id)

    def decorator(func: F) -> F:
        func.__abi_case__ = case_id  # type: ignore[attr-defined]
        return func

    return decorator


def registered_case_ids() -> set[str]:
    """
    Returns every case id registered so far in this process.

    Returns:
        set[str]: All decorated case ids.
    """

    return set(_REGISTRY)


# Engine results that are NOT representable in ABI v1. When a call
# produces one of these, the shim returns ND_ERR_TYPE with a message
# naming the type. Bindings must document this in their error docs.
# (Length-paired strings and richer value variants are reserved for
# ABI 0.2 — see native/nadata.h "Reserved for Choice B".)
HOME_TYPES_NOT_REPRESENTABLE: tuple[str, ...] = (
    "DateTime",
    "Color",
)
