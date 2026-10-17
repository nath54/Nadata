"""
TODO
"""


class BitValueType:
    """
    TODO: abstract class
    """

    def __init__(
        self,
        endian: str = "big"
    ) -> None:
        """
        Init function.
        """

        self.endian: str = endian


class BitValueTypeInt(BitValueType):
    """
    TODO
    """

    def __init__(
        self,
        num_bits: int = 32,
        signed: bool = True,
        endian: str = "big"
    ) -> None:
        """
        Init function.
        """

        # Parent init
        super().__init__(endian=endian)

        self.num_bits: int = num_bits
        self.num_bits_val: int = num_bits - int(signed)
        self.signed: bool = signed

        # Compute interval range
        self.val_min: int = 0
        self.val_max: int = 2**self.num_bits_val

        if signed:
            self.val_min = -2**(self.num_bits_val-1)
            self.val_max = 2**(self.num_bits_val-1) - 1

    def check_value(self, value: int) -> None:
        """
        TODO: raise error if value is incorrect.
        """

        # Check if value is below min value
        if value < self.val_min:
            raise ValueError("TODO")

        # Check if value is above max value
        if value > self.val_max:
            raise ValueError("TODO")


class BitValueTypeFloat(BitValueType):
    """
    TODO
    """

    def __init__(
        self,
        num_bits: int = 32,
        num_bits_exponent: int = 8
        signed: bool = True,
        endian: str = "big"
    ) -> None:
        """
        Init function.
        """

        # Parent init
        super().__init__(endian=endian)

        self.num_bits: int = num_bits
        self.num_bits_exponent: int = num_bits_exponent
        self.num_bits_mantissa: int = num_bits - int(signed) - num_bits_exponent
        self.signed: bool = signed

    def check_value(self, value: float) -> None:
        """
        TODO: raise error if value is incorrect.
        """

        pass


class BitArray:
    """
    TODO
    """

    def __init__(
        self,
        bits: list[bool],
        value_type: BitValueType
    ) -> None:

        self.bits: list[bool] = bits
        self.value_type: BitValueType = BitValueType


class Bits:
    """
    TODO
    """

    @staticmethod
    def float_val_to_bits(
        self,
        value: float,
        value_type: BitValueTypeFloat
    ) -> BitArray:
        """
        TODO
        """

        # Check value
        value_type.check_value(value=value)

        # Prepare bits results
        res: list[bool] = []

        # Bit sign
        if signed:
            res.append( value >= 0 )

        # Exponent and Mantissa bits values
        exp_value: int = 0
        mantissa_value: int = 0

        exponent_bits: list[bool] = []
        mantissa_bits: list[bool] = []

        # Assemble
        res += exponent_bits + mantissa_bits

        # TODO: manage endian

        # Final bits
        bit_arr: BitArray = BitArray(
            bits=res,
            value_type=value_type
        )


    @staticmethod
    def int_val_to_bits(
        self,
        value: int,
        value_type: BitValueTypeInt
    ) -> BitArray:
        """
        TODO
        """

        value_type.check_value(value=value)


    @staticmethod
    def bits_to_int_val(
        self,
        bit_arr: BitArray,
        value_type: BitValueTypeInt
    ) -> int:
        """
        TODO
        """

        return 0

    @staticmethod
    def bits_to_float_val(
        self,
        bit_arr: BitArray,
        value_type: BitValueTypeFloat
    ) -> float:
        """
        TODO
        """

        return 0
