"""
DateTime and Time representations for Nadata.

Provides the DateTime class supporting date (year, month, day) and
time (hour, minute, second, millisecond) components, arithmetic,
comparisons, and factory constructors.
"""

# Import Modules
from __future__ import annotations
from typing import Any
import datetime as py_datetime



class DateTime:
    """
    Represents a specific point in time or a date/time duration in Nadata.

    Attributes:
        year (int): Calendar year.
        month (int): Month index (1-12).
        day (int): Day of the month (1-31).
        hour (int): Hour of the day (0-23).
        minute (int): Minute of the hour (0-59).
        second (int): Second of the minute (0-59).
        millisecond (int): Millisecond of the second (0-999).
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,invalid-name
    def __init__(
        self,
        y: int = 0,
        m: int = 0,
        d: int = 0,
        H: int = 0,
        M: int = 0,
        S: int = 0,
        MS: int = 0
    ) -> None:

        """
        Initializes the DateTime instance.

        Args:
            y (int): Year component. Defaults to 0.
            m (int): Month component. Defaults to 0.
            d (int): Day component. Defaults to 0.
            H (int): Hour component. Defaults to 0.
            M (int): Minute component. Defaults to 0.
            S (int): Second component. Defaults to 0.
            MS (int): Millisecond component. Defaults to 0.
        """

        # Store individual components with safe zero defaults
        self.year: int = y
        self.month: int = m
        self.day: int = d
        self.hour: int = H
        self.minute: int = M
        self.second: int = S
        self.millisecond: int = MS

    @classmethod
    def now(cls) -> DateTime:
        """
        Creates a DateTime instance representing the current local time.

        Returns:
            DateTime: Current date and time.
        """

        # Get current python local datetime
        current: py_datetime.datetime = py_datetime.datetime.now()

        # Extract milliseconds from microseconds
        ms: int = current.microsecond // 1000

        return cls(
            y=current.year,
            m=current.month,
            d=current.day,
            H=current.hour,
            M=current.minute,
            S=current.second,
            MS=ms
        )

    def to_total_milliseconds(self) -> int:
        """
        Converts calendar and time components into approximate total milliseconds.

        Returns:
            int: Total elapsed milliseconds since epoch year 0.
        """

        # Day calculations approximating 365.2425 days per year and 30.4375 days per month
        total_days: float = (
            self.year * 365.2425
            + self.month * 30.4375
            + self.day
        )

        total_hours: float = total_days * 24.0 + self.hour
        total_minutes: float = total_hours * 60.0 + self.minute
        total_seconds: float = total_minutes * 60.0 + self.second

        return int(total_seconds * 1000.0 + self.millisecond)

    def __sub__(self, other: Any) -> DateTime:
        """
        Subtracts another DateTime from this one, returning the difference as a DateTime.

        Args:
            other (Any): Another DateTime instance to subtract.

        Returns:
            DateTime: Difference DateTime instance.

        Raises:
            TypeError: If other is not a DateTime instance.
        """

        if not isinstance(other, DateTime):
            raise TypeError(f"Cannot subtract {type(other).__name__} from DateTime")

        # Compute difference across all individual fields
        diff_ms: int = self.millisecond - other.millisecond
        diff_s: int = self.second - other.second
        diff_m: int = self.minute - other.minute
        diff_h: int = self.hour - other.hour
        diff_d: int = self.day - other.day
        diff_mon: int = self.month - other.month
        diff_y: int = self.year - other.year

        # Handle borrow for millisecond
        if diff_ms < 0:
            diff_ms += 1000
            diff_s -= 1

        # Handle borrow for second
        if diff_s < 0:
            diff_s += 60
            diff_m -= 1

        # Handle borrow for minute
        if diff_m < 0:
            diff_m += 60
            diff_h -= 1

        # Handle borrow for hour
        if diff_h < 0:
            diff_h += 24
            diff_d -= 1

        # Handle borrow for day
        if diff_d < 0:
            diff_d += 30
            diff_mon -= 1

        # Handle borrow for month
        if diff_mon < 0:
            diff_mon += 12
            diff_y -= 1

        return DateTime(
            y=diff_y,
            m=diff_mon,
            d=diff_d,
            H=diff_h,
            M=diff_m,
            S=diff_s,
            MS=diff_ms
        )

    @staticmethod
    def _add_time_components(
        t1: tuple[int, int, int, int],
        t2: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int, int]:
        """
        Adds millisecond, second, minute, and hour components with carries.

        Args:
            t1 (tuple[int, int, int, int]): (ms, s, m, h) from first datetime.
            t2 (tuple[int, int, int, int]): (ms, s, m, h) from second datetime.

        Returns:
            tuple[int, int, int, int, int]: (rem_ms, rem_s, rem_m, rem_h, carry_days).
        """

        tot_ms: int = t1[0] + t2[0]
        tot_s: int = t1[1] + t2[1] + (tot_ms // 1000)
        tot_m: int = t1[2] + t2[2] + (tot_s // 60)
        tot_h: int = t1[3] + t2[3] + (tot_m // 60)

        return (
            tot_ms % 1000,
            tot_s % 60,
            tot_m % 60,
            tot_h % 24,
            tot_h // 24
        )

    def __add__(self, other: Any) -> DateTime:
        """
        Adds another DateTime duration to this one.

        Args:
            other (Any): Another DateTime instance to add.

        Returns:
            DateTime: Sum DateTime instance.

        Raises:
            TypeError: If other is not a DateTime instance.
        """

        if not isinstance(other, DateTime):
            raise TypeError(f"Cannot add {type(other).__name__} to DateTime")

        ms, s, m, h, carry_d = self._add_time_components(
            t1=(self.millisecond, self.second, self.minute, self.hour),
            t2=(other.millisecond, other.second, other.minute, other.hour)
        )

        tot_d: int = self.day + other.day + carry_d
        tot_mon: int = self.month + other.month + (tot_d // 30)

        return DateTime(
            y=self.year + other.year + (tot_mon // 12),
            m=tot_mon % 12,
            d=tot_d % 30,
            H=h,
            M=m,
            S=s,
            MS=ms
        )



    def _to_tuple(self) -> tuple[int, int, int, int, int, int, int]:
        """
        Converts the components to an ordered tuple for comparison.

        Returns:
            tuple[int, int, int, int, int, int, int]: Ordered tuple representation.
        """

        return (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.millisecond
        )

    def __eq__(self, other: Any) -> bool:
        """
        Checks equality with another DateTime instance.

        Args:
            other (Any): Other object.

        Returns:
            bool: True if all fields match, False otherwise.
        """

        if not isinstance(other, DateTime):
            return False

        return self._to_tuple() == other._to_tuple()

    def __lt__(self, other: Any) -> bool:
        """
        Checks if this DateTime is strictly before another.

        Args:
            other (Any): Other object.

        Returns:
            bool: True if less than other.
        """

        if not isinstance(other, DateTime):
            return NotImplemented

        return self._to_tuple() < other._to_tuple()

    def __le__(self, other: Any) -> bool:
        """
        Checks if this DateTime is before or equal to another.

        Args:
            other (Any): Other object.

        Returns:
            bool: True if less than or equal to other.
        """

        if not isinstance(other, DateTime):
            return NotImplemented

        return self._to_tuple() <= other._to_tuple()

    def __gt__(self, other: Any) -> bool:
        """
        Checks if this DateTime is strictly after another.

        Args:
            other (Any): Other object.

        Returns:
            bool: True if greater than other.
        """

        if not isinstance(other, DateTime):
            return NotImplemented

        return self._to_tuple() > other._to_tuple()

    def __ge__(self, other: Any) -> bool:
        """
        Checks if this DateTime is after or equal to another.

        Args:
            other (Any): Other object.

        Returns:
            bool: True if greater than or equal to other.
        """

        if not isinstance(other, DateTime):
            return NotImplemented

        return self._to_tuple() >= other._to_tuple()

    def __repr__(self) -> str:
        """
        Returns string representation of the DateTime.

        Returns:
            str: DateTime constructor style representation.
        """

        parts: list[str] = [
            f"y={self.year}",
            f"m={self.month}",
            f"d={self.day}",
        ]

        if any((self.hour, self.minute, self.second, self.millisecond)):
            parts.extend([
                f"H={self.hour}",
                f"M={self.minute}",
                f"S={self.second}",
                f"MS={self.millisecond}"
            ])

        return f"DateTime({', '.join(parts)})"
