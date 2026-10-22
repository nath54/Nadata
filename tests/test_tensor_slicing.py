"""
Tests for tensor slicing on getters and assignments.

Covers 'start:stop' slices per dimension (with optional bounds),
mixed integer/slice access, rank reduction, block assignment with
row-major payloads, scalar broadcast assignment, and error cases.
"""

# Import Modules
from __future__ import annotations
import pytest
from typing import Any

from src.api import Nadata
from src.core.database.database import NaDataDatabase

SCHEMA: str = (
    "class P {\n"
    "  v: Float32 = 0.0;\n"
    "  m: Tensor[Float32, 3, 4] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, "
    "9.0, 10.0, 11.0, 12.0];\n"
    "}"
)


class TestSlicingGetter:
    """
    Tests sliced reads on tensors.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the probe schema.

        Returns:
            Nadata: Configured application.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        return app

    def test_slice_size_and_shape(self) -> None:
        """
        Tests that a 2D slice reports the sliced shape.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.v = p.m[0:2, 1:3].size();\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["v"].get() == 4

    def test_row_slice_via_open_bound(self) -> None:
        """
        Tests that '[:]' selects a whole dimension.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "row: Tensor[Float32, 4] = p.m[1, :];\n"
            "p.v = row[3];\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["v"].get() == 8.0

    def test_column_slice_keeps_rank(self) -> None:
        """
        Tests that slicing one axis of a 2D tensor yields a 1D tensor.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "col: Tensor[Float32, 2] = p.m[:2, 2];\n"
            "p.v = col[1];\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["v"].get() == 7.0

    def test_element_from_slice_result(self) -> None:
        """
        Tests that a slice result can be indexed immediately.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.v = p.m[1:3, 0:2][0, 1];\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        # Sub-block of rows 1-2, cols 0-1: row-major [5.0, 6.0, 9.0, 10.0]
        assert probe.attributes["v"].get() == 6.0

    def test_slice_bounds_from_expressions(self) -> None:
        """
        Tests that slice bounds accept arbitrary integer expressions.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "offset: Int32 = 1;\n"
            "p.v = p.m[offset:offset + 2, 0].size();\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["v"].get() == 2

    def test_integer_index_unchanged(self) -> None:
        """
        Tests that plain multi-index access still returns elements.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.v = p.m[2, 3];\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["v"].get() == 12.0


class TestSlicingAssignment:
    """
    Tests sliced writes on tensors.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the probe schema.

        Returns:
            Nadata: Configured application.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        return app

    def test_block_assignment_row_major(self) -> None:
        """
        Tests that a flat payload fills the slice row-major.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.m[0:2, 1:3] = [40.0, 41.0, 42.0, 43.0];\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["m"].get() == [
            1.0, 40.0, 41.0, 4.0,
            5.0, 42.0, 43.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]

    def test_nested_payload_flattened(self) -> None:
        """
        Tests that a nested array payload is flattened row-major.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.m[1:3, 0:2] = [[50.0, 51.0], [52.0, 53.0]];\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["m"].get() == [
            1.0, 2.0, 3.0, 4.0,
            50.0, 51.0, 7.0, 8.0,
            52.0, 53.0, 11.0, 12.0,
        ]

    def test_scalar_broadcast(self) -> None:
        """
        Tests that a scalar payload fills every selected cell.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.m[:, 0] = 0.0;\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["m"].get() == [
            0.0, 2.0, 3.0, 4.0,
            0.0, 6.0, 7.0, 8.0,
            0.0, 10.0, 11.0, 12.0,
        ]

    def test_partial_row_assignment(self) -> None:
        """
        Tests assignment to a mixed integer/slice selection.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.m[1, 1:3] = [60.0, 61.0];\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["m"].get() == [
            1.0, 2.0, 3.0, 4.0,
            5.0, 60.0, 61.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]

    def test_wrong_payload_length_rejected(self) -> None:
        """
        Tests that a payload length mismatch raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="expects 4 element"):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.m[0:2, 1:3] = [1.0, 2.0];\n"
            ))

    def test_element_type_violation_rejected(self) -> None:
        """
        Tests that assigning a wrong-typed payload raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.m[0:2, 0] = [1.0, \"no\"];\n"
            ))


class TestSlicingErrors:
    """
    Tests invalid slice usage.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the probe schema.

        Returns:
            Nadata: Configured application.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        return app

    def test_empty_slice_rejected(self) -> None:
        """
        Tests that an empty slice range raises ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="empty"):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.v = p.m[1:1, 0].size();\n"
            ))

    def test_reversed_slice_rejected(self) -> None:
        """
        Tests that a reversed slice range raises ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="Invalid slice"):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.v = p.m[2:1, 0].size();\n"
            ))

    def test_out_of_range_bound_rejected(self) -> None:
        """
        Tests that a bound past the dimension extent raises ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="Invalid slice"):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.v = p.m[0:9, 0].size();\n"
            ))

    def test_out_of_range_integer_rejected(self) -> None:
        """
        Tests that mixed access still bounds-checks integer indices.
        """

        app: Nadata = self._app()

        with pytest.raises(IndexError):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.v = p.m[9, 0:2].size();\n"
            ))

    def test_slice_on_array_rejected(self) -> None:
        """
        Tests that slicing a plain array is not supported.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text="class Q { v: Float32 = 0.0; a: Array[Float32] = [1.0, 2.0]; }")

        with pytest.raises(TypeError):
            app.execute_data(data_text=(
                "q = Q();\n"
                "q.v = q.a[0:1].size();\n"
            ))
