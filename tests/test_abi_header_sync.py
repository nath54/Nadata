"""
Sync test between the native ABI header and the Python contract.

Mirrors the grammar-sync approach of tests/test_grammar.py: whenever
`native/nadata.h` and `tests/abi/contract.py` disagree on the ABI
version or a status code value, this suite fails at CI time — long
before a binding is generated from a stale header.
"""

# Import Modules
import re
from pathlib import Path

import pytest

from abi.contract import ABI_VERSION, NdStatus

HEADER_PATH: Path = Path(__file__).parent.parent / "native" / "nadata.h"


def _read_header() -> str:
    """
    Reads the ABI header, skipping the test cleanly if it is absent
    (e.g. a source distribution without native/).

    Returns:
        str: Full header text.

    Raises:
        pytest.skip: When native/nadata.h does not exist.
    """

    if not HEADER_PATH.exists():
        pytest.skip("native/nadata.h not present")

    return HEADER_PATH.read_text(encoding="utf-8")


def _parse_version(text: str) -> int:
    """
    Extracts ND_ABI_VERSION from the header text.

    Args:
        text (str): Header text.

    Returns:
        int: The declared ABI version.

    Raises:
        AssertionError: When the define is missing.
    """

    match: re.Match[str] | None = re.search(
        r"#define\s+ND_ABI_VERSION\s+(\d+)", text
    )

    assert match is not None, "ND_ABI_VERSION not found in nadata.h"

    return int(match.group(1))


def _parse_status_codes(text: str) -> dict[str, int]:
    """
    Extracts the NdStatusCode enum entries from the header text.

    Handles both plain values (`ND_OK = 0`) and negative entries, with
    or without trailing comments.

    Args:
        text (str): Header text.

    Returns:
        dict[str, int]: Symbol name -> numeric value.
    """

    body: re.Match[str] | None = re.search(
        r"typedef\s+enum\s+NdStatusCode\s*\{(.*?)\}\s*NdStatusCode;",
        text,
        re.DOTALL,
    )

    assert body is not None, "NdStatusCode enum not found in nadata.h"

    codes: dict[str, int] = {}

    for name, value in re.findall(r"(ND_\w+)\s*=\s*(-?\d+)", body.group(1)):
        codes[name] = int(value)

    return codes


class TestAbiHeaderSync:
    """Header <-> contract consistency guards."""

    def test_version_matches(self) -> None:
        """ND_ABI_VERSION in the header equals the contract version."""

        header: str = _read_header()

        assert _parse_version(header) == ABI_VERSION

    def test_status_codes_match(self) -> None:
        """
        Every NdStatus entry matches the header's NdStatusCode exactly,
        both by name and numeric value.
        """

        header: str = _read_header()
        header_codes: dict[str, int] = _parse_status_codes(header)

        contract_codes: dict[str, int] = {
            member.name: int(member.value) for member in NdStatus
        }

        assert header_codes == contract_codes, (
            "native/nadata.h and tests/abi/contract.py disagree:\n"
            f"  header only:   {sorted(set(header_codes) - set(contract_codes))}\n"
            f"  contract only: {sorted(set(contract_codes) - set(header_codes))}\n"
            f"  value drift:   "
            f"{sorted(k for k in header_codes if k in contract_codes and header_codes[k] != contract_codes[k])}"
        )

    def test_status_codes_are_negative_or_zero(self) -> None:
        """ND_OK is zero; every error is negative (call-success check)."""

        for member in NdStatus:
            if member is NdStatus.ND_OK:
                assert member.value == 0
            else:
                assert member.value < 0, member.name
