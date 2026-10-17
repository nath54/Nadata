"""
Bit-level representations and conversion utilities for Nadata.

Supports arbitrary-width signed/unsigned integers and custom floating-point specifications.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Iterator

import math


class BitValueType:
    """
    Abstract base type defining bit-level storage layout and endianness.

    Attributes:
        endian (str): Byte and bit order, either 'big' (MSB first) or 'little' (LSB first).
    """

    def __init__(
        self,
        endian: str = "big"
    ) -> None:
        """
        Initializes the BitValueType instance.

        Args:
            endian (str, optional): Byte/bit order, either 'big' or 'little'. Defaults to 'big'.

        Raises:
            ValueError: If endian is not 'big' or 'little'.
        """

        # Check endian value
        if endian not in ("big", "little"):
            raise ValueError(f"Endian must be 'big' or 'little', got '{endian}'")

        # Store endian value
        self.endian: str = endian


class BitValueTypeInt(BitValueType):
    """
    Bit specification for integer types (supports arbitrary bit widths, signed/unsigned).

    Attributes:
        endian (str): Byte and bit order ('big' or 'little').
        num_bits (int): Total bit width (e.g. 2, 4, 8, 16, 32, 64).
        signed (bool): Whether the integer is signed (two's complement) or unsigned.
        num_bits_val (int): Number of value bits (excluding sign bit if signed).
        val_min (int): Minimum allowable integer value.
        val_max (int): Maximum allowable integer value.
    """

    def __init__(
        self,
        num_bits: int = 32,
        signed: bool = True,
        endian: str = "big"
    ) -> None:
        """
        Initializes the BitValueTypeInt specification.

        Args:
            num_bits (int, optional): Total bit width. Defaults to 32.
            signed (bool, optional): Whether integer is signed. Defaults to True.
            endian (str, optional): Byte/bit order ('big' or 'little'). Defaults to 'big'.

        Raises:
            ValueError: If num_bits <= 0 or endian is invalid.
        """

        # Parent init
        super().__init__(endian=endian)

        # Check number of bits
        if num_bits <= 0:
            raise ValueError(f"num_bits must be positive, got {num_bits}")

        self.num_bits: int = num_bits
        self.signed: bool = signed

        # Number of value bits (excluding sign bit if signed)
        self.num_bits_val: int = num_bits - int(signed)

        # Compute interval range
        self.val_min: int = 0
        self.val_max: int = (1 << num_bits) - 1

        if signed:
            self.val_min: int = -(1 << (num_bits - 1))
            self.val_max: int = (1 << (num_bits - 1)) - 1

    def check_value(self, value: int) -> None:
        """
        Validates that an integer fits within the bounds of this type.

        Args:
            value (int): Integer value to validate.

        Raises:
            TypeError: If value is not an int or is a bool.
            ValueError: If value is strictly outside [val_min, val_max].
        """

        # Check value type
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")

        # Check value interval
        if value < self.val_min or value > self.val_max:
            sign_str = "Int" if self.signed else "UInt"
            raise ValueError(
                f"Value {value} is out of bounds for {sign_str}{self.num_bits} "
                f"[{self.val_min}, {self.val_max}]"
            )


class BitValueTypeFloat(BitValueType):
    """
    Bit specification for floating-point types with arbitrary exponent and mantissa sizes.

    Attributes:
        endian (str): Byte and bit order ('big' or 'little').
        num_bits (int): Total bit width (e.g. 8, 16, 32, 64).
        num_bits_exponent (int): Number of bits reserved for exponent.
        signed (bool): Whether there is a sign bit.
        num_bits_mantissa (int): Number of bits reserved for fraction/mantissa.
        bias (int): Exponent bias calculated as 2^(num_bits_exponent - 1) - 1.
        max_exp (int): Maximum raw exponent value (used for Inf and NaN).
        val_max (float): Maximum finite normal positive float.
        val_min (float): Minimum finite normal float (-val_max if signed, else 0.0).
    """

    def __init__(
        self,
        num_bits: int = 32,
        num_bits_exponent: int = 8,
        signed: bool = True,
        endian: str = "big"
    ) -> None:
        """
        Initializes the BitValueTypeFloat specification.

        Args:
            num_bits (int, optional): Total bit width. Defaults to 32.
            num_bits_exponent (int, optional): Number of bits reserved for exponent. Defaults to 8.
            signed (bool, optional): Whether there is a sign bit. Defaults to True.
            endian (str, optional): Byte/bit order ('big' or 'little'). Defaults to 'big'.

        Raises:
            ValueError: If num_bits_exponent < 1, mantissa bits < 0, or endian is invalid.
        """

        # Parent init
        super().__init__(endian=endian)

        # Check number of exponent bits
        if num_bits_exponent < 1:
            raise ValueError(f"num_bits_exponent must be at least 1, got {num_bits_exponent}")

        self.num_bits: int = num_bits
        self.num_bits_exponent: int = num_bits_exponent
        self.signed: bool = signed
        self.num_bits_mantissa: int = num_bits - int(signed) - num_bits_exponent

        # Check number of mantissa bits
        if self.num_bits_mantissa < 0:
            raise ValueError(
                f"Total bits ({num_bits}) is too small for signed={signed} "
                f"and exponent_bits={num_bits_exponent}"
            )

        # Exponent bias: 2^(exp_bits - 1) - 1
        self.bias: int = (1 << (num_bits_exponent - 1)) - 1
        self.max_exp: int = (1 << num_bits_exponent) - 1

        # Max finite normal float
        max_normal_exp = self.max_exp - 1
        self.val_max: float = (2.0 - (2.0 ** -self.num_bits_mantissa)) * (2.0 ** (max_normal_exp - self.bias))
        self.val_min: float = -self.val_max if signed else 0.0

    def check_value(self, value: float) -> None:
        """
        Validates that a floating-point number is acceptable for this specification.

        Args:
            value (float): Floating-point number to validate.

        Raises:
            TypeError: If value is not a float or int, or is a bool.
            ValueError: If an unsigned float receives a negative value.
        """

        # Check value type
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Expected float or int, got {type(value).__name__}")

        # Check value integrity
        if math.isnan(value) or math.isinf(value):
            return

        # Check value range
        if not self.signed and value < 0:
            raise ValueError(f"Value {value} cannot be negative for unsigned Float{self.num_bits}")


class BitArray:
    """
    Representation of a fixed-size sequence of bits with an associated BitValueType.

    Attributes:
        bits (list[bool]): List of boolean values representing individual bits.
        value_type (BitValueType): Associated bit specification defining layout and endianness.
    """

    def __init__(
        self,
        bits: list[bool],
        value_type: BitValueType
    ) -> None:
        """
        Initializes the BitArray instance.

        Args:
            bits (list[bool]): List of booleans representing the bits.
            value_type (BitValueType): Associated bit specification.
        """

        self.bits: list[bool] = [bool(b) for b in bits]
        self.value_type: BitValueType = value_type

    def __len__(self) -> int:
        """
        Returns the number of bits in the array.

        Returns:
            int: Total bit count.
        """

        return len(self.bits)

    def __getitem__(self, index: int | slice) -> bool | list[bool]:
        """
        Gets a bit or a slice of bits from the array.

        Args:
            index (int | slice): Integer index or slice.

        Returns:
            bool | list[bool]: Single boolean bit or slice of boolean bits.
        """

        return self.bits[index]

    def __iter__(self) -> Iterator[bool]:
        """
        Returns an iterator over the boolean bits.

        Returns:
            Iterator[bool]: Iterator yielding each bit as a bool.
        """

        return iter(self.bits)

    def __eq__(self, other: Any) -> bool:
        """
        Checks equality against another BitArray or list of booleans.

        Args:
            other (Any): Object to compare against.

        Returns:
            bool: True if equal, False otherwise.
        """

        if isinstance(other, BitArray):
            return self.bits == other.bits and self.value_type.endian == other.value_type.endian

        if isinstance(other, list):
            return self.bits == other

        return False

    def to_bin_str(self, delimiter: str = "") -> str:
        """
        Converts bits to a binary string representation.

        Args:
            delimiter (str, optional): Delimiter inserted between bit characters. Defaults to "".

        Returns:
            str: Binary string (e.g. '01010101').
        """

        chars: list[str] = ["1" if b else "0" for b in self.bits]

        return delimiter.join(chars)


    def to_bytes(self) -> bytes:
        """
        Packs bits into bytes, padding trailing 0s if length is not a multiple of 8.

        Returns:
            bytes: Byte sequence containing the packed bits.
        """

        byte_list = bytearray()

        for i in range(0, len(self.bits), 8):

            chunk = self.bits[i:i + 8]
            byte_val = 0

            for b in chunk:
                byte_val = (byte_val << 1) | int(b)

            if len(chunk) < 8:
                byte_val <<= (8 - len(chunk))

            byte_list.append(byte_val)

        return bytes(byte_list)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        value_type: BitValueType,
        num_bits: int | None = None
    ) -> BitArray:
        """
        Creates a BitArray from raw bytes.

        Args:
            data (bytes): Input bytes to unpack.
            value_type (BitValueType): Associated bit specification.
            num_bits (int, optional): Target number of bits to extract. Defaults to None (all bits).

        Returns:
            BitArray: Constructed BitArray instance.
        """

        bits: list[bool] = []

        for byte in data:
            for shift in range(7, -1, -1):
                bits.append(bool((byte >> shift) & 1))

        target_bits = num_bits if num_bits is not None else len(bits)

        return cls(bits=bits[:target_bits], value_type=value_type)

    def __repr__(self) -> str:
        """
        Returns the developer representation of the BitArray.

        Returns:
            str: String representation.
        """

        return f"BitArray('{self.to_bin_str()}', type={self.value_type.__class__.__name__})"

    def __str__(self) -> str:
        """
        Returns the binary string representation of the BitArray.

        Returns:
            str: Binary string.
        """

        return self.to_bin_str()


class Bits:
    """
    Conversion utilities between Python native numbers (int, float) and BitArray.
    """

    @staticmethod
    def int_val_to_bits(
        value: int,
        value_type: BitValueTypeInt
    ) -> BitArray:
        """
        Converts an integer into a BitArray according to its BitValueTypeInt specification.

        Uses two's complement for negative integers when signed.

        Args:
            value (int): Integer value to convert.
            value_type (BitValueTypeInt): Target integer bit specification.

        Returns:
            BitArray: BitArray containing the encoded bits.

        Raises:
            TypeError: If value is not an int or is a bool.
            ValueError: If value is out of bounds for the given bit specification.
        """

        # Validate value
        value_type.check_value(value=value)

        # Mask to bit width (two's complement for negatives)
        raw_uint = value & ((1 << value_type.num_bits) - 1)

        # Generate bits from MSB to LSB
        bits: list[bool] = [
            bool((raw_uint >> (value_type.num_bits - 1 - i)) & 1)
            for i in range(value_type.num_bits)
        ]

        # Little endian reverses the bit order
        if value_type.endian == "little":
            bits.reverse()

        return BitArray(bits=bits, value_type=value_type)

    @staticmethod
    def bits_to_int_val(
        bit_arr: BitArray,
        value_type: BitValueTypeInt
    ) -> int:
        """
        Converts a BitArray back into a Python integer according to BitValueTypeInt.

        Args:
            bit_arr (BitArray): Input bit array.
            value_type (BitValueTypeInt): Bit specification for decoding.

        Returns:
            int: Decoded Python integer (signed or unsigned).

        Raises:
            ValueError: If bit array length does not match expected num_bits.
        """

        bits = list(bit_arr.bits)

        if len(bits) != value_type.num_bits:
            raise ValueError(
                f"BitArray length ({len(bits)}) does not match expected num_bits ({value_type.num_bits})"
            )

        # Restore MSB-first order if little-endian
        if value_type.endian == "little":
            bits.reverse()

        # Reconstruct unsigned integer
        raw_uint = 0
        for b in bits:
            raw_uint = (raw_uint << 1) | int(b)

        # Handle two's complement sign
        if value_type.signed and bits[0]:
            return raw_uint - (1 << value_type.num_bits)

        return raw_uint

    @staticmethod
    def float_val_to_bits(
        value: float,
        value_type: BitValueTypeFloat
    ) -> BitArray:
        """
        Converts a float into a BitArray according to BitValueTypeFloat specification.

        Handles IEEE-like zero, subnormal, normal, infinity, and NaN values.

        Args:
            value (float): Float value to convert.
            value_type (BitValueTypeFloat): Target floating-point bit specification.

        Returns:
            BitArray: BitArray containing encoded sign, exponent, and mantissa bits.

        Raises:
            TypeError: If value is not a float or int, or is a bool.
            ValueError: If an unsigned float specification receives a negative value.
        """

        value_type.check_value(value=value)

        exp_bits_len = value_type.num_bits_exponent
        mant_bits_len = value_type.num_bits_mantissa
        bias = value_type.bias
        max_exp = value_type.max_exp

        sign_bit = (math.copysign(1.0, value) < 0) if value_type.signed else False

        # Handle NaN
        if math.isnan(value):
            biased_exp = max_exp
            mantissa = 1 << (mant_bits_len - 1) if mant_bits_len > 0 else 0

        # Handle Infinity
        elif math.isinf(value):
            biased_exp = max_exp
            mantissa = 0

        # Handle Zero
        elif value == 0.0:
            biased_exp = 0
            mantissa = 0

        else:

            abs_val = abs(value)

            # Use frexp: abs_val = m * 2^e_raw, where 0.5 <= m < 1.0
            m, e_raw = math.frexp(abs_val)

            # Normalized exponent for 1.f * 2^e is e_raw - 1
            e = e_raw - 1
            biased_exp = e + bias

            if biased_exp >= max_exp:

                # Overflow to infinity
                biased_exp = max_exp
                mantissa = 0

            elif biased_exp <= 0:

                # Subnormal number: exponent is 0, fraction relative to 2^(1 - bias)
                biased_exp = 0
                f = abs_val / (2.0 ** (1 - bias))
                mantissa = round(f * (2 ** mant_bits_len))

                if mantissa >= (1 << mant_bits_len):
                    biased_exp = 1
                    mantissa = 0

            else:

                # Normal number: fraction f in [0, 1)
                f = (abs_val / (2.0 ** e)) - 1.0
                mantissa = round(f * (2 ** mant_bits_len))

                if mantissa >= (1 << mant_bits_len):

                    biased_exp += 1
                    mantissa = 0

                    if biased_exp >= max_exp:
                        biased_exp = max_exp
                        mantissa = 0

        # Construct exponent bits (MSB first)
        exponent_bits = [
            bool((biased_exp >> (exp_bits_len - 1 - i)) & 1)
            for i in range(exp_bits_len)
        ]

        # Construct mantissa bits (MSB first)
        mantissa_bits = [
            bool((mantissa >> (mant_bits_len - 1 - i)) & 1)
            for i in range(mant_bits_len)
        ]

        # Assemble layout
        res: list[bool] = []
        if value_type.signed:
            res.append(sign_bit)
        res.extend(exponent_bits)
        res.extend(mantissa_bits)

        # Little endian reverses the bit order
        if value_type.endian == "little":
            res.reverse()

        return BitArray(bits=res, value_type=value_type)

    @staticmethod
    def bits_to_float_val(
        bit_arr: BitArray,
        value_type: BitValueTypeFloat
    ) -> float:
        """
        Converts a BitArray back into a Python float according to BitValueTypeFloat.

        Args:
            bit_arr (BitArray): Input bit array.
            value_type (BitValueTypeFloat): Target floating-point bit specification.

        Returns:
            float: Decoded Python floating-point number.

        Raises:
            ValueError: If bit array length does not match expected num_bits.
        """

        bits: list[bool] = list(bit_arr.bits)

        if len(bits) != value_type.num_bits:
            raise ValueError(
                f"BitArray length ({len(bits)}) does not match expected num_bits ({value_type.num_bits})"
            )

        # Restore MSB-first order if little-endian
        if value_type.endian == "little":
            bits.reverse()

        idx = 0
        sign_bit = False

        if value_type.signed:
            sign_bit = bits[0]
            idx += 1

        exp_bits_len = value_type.num_bits_exponent
        mant_bits_len = value_type.num_bits_mantissa
        bias = value_type.bias
        max_exp = value_type.max_exp

        # Extract exponent
        biased_exp = 0

        for b in bits[idx : idx + exp_bits_len]:
            biased_exp = (biased_exp << 1) | int(b)

        idx += exp_bits_len

        # Extract mantissa
        mantissa = 0

        for b in bits[idx : idx + mant_bits_len]:
            mantissa = (mantissa << 1) | int(b)

        sign = -1.0 if (value_type.signed and sign_bit) else 1.0

        # Special cases: Inf and NaN
        if biased_exp == max_exp:
            if mantissa != 0:
                return float("nan")
            return sign * float("inf")

        # Subnormal or zero
        if biased_exp == 0:
            if mantissa == 0:
                return sign * 0.0
            return sign * (mantissa / (2.0 ** mant_bits_len)) * (2.0 ** (1 - bias))

        # Normal float
        return sign * (1.0 + (mantissa / (2.0 ** mant_bits_len))) * (2.0 ** (biased_exp - bias))
