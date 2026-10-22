"""
Tests for tensor initializers, elementwise operations, and reductions.

Covers zeros()/fill()/eye() initialization (schema defaults, constructor
arguments, typed locals, symbolic dims), elementwise tensor-tensor and
broadcast tensor-scalar arithmetic, and the sum/min/max/transpose/fill
tensor members.
"""

# Import Modules
from __future__ import annotations
import pytest
from typing import Any

from src.api import Nadata
from src.core.database.database import NaDataDatabase

SCHEMA: str = (
    "class P {\n"
    "  const dim: Int32 = 3;\n"
    "  v: Float32 = 0.0;\n"
    "  i: Int32 = 0;\n"
    "  w: Tensor[Float32, 2, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\n"
    "  u: Tensor[Float32, 2, 3] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];\n"
    "  r: Tensor[Float32, 3, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
    "  z: Tensor[Float32, 2, 3] = zeros();\n"
    "  f: Tensor[Float32, 2, 2] = fill(2.5);\n"
    "  e: Tensor[Float32, dim, dim] = eye();\n"
    "  n: Tensor[Int32, 4] = [1, 2, 3, 4];\n"
    "}"
)


class TestInitializers:
    """
    Tests zeros()/fill()/eye() initialization.
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

    def _value(self, app: Nadata, data_text: str, attribute: str) -> Any:
        """
        Runs a script and returns one attribute of the probe instance.

        Args:
            app (Nadata): Configured application.
            data_text (str): Script touching 'p'.
            attribute (str): Attribute to read.

        Returns:
            Any: The raw attribute payload.
        """

        app.execute_data(data_text=data_text)

        probe: Any = app.query(struct_name="P").last()

        return probe.attributes[attribute].get()

    def test_schema_default_zeros(self) -> None:
        """
        Tests that a zeros() schema default fills with 0.0.
        """

        app: Nadata = self._app()

        assert self._value(app=app, data_text="p = P();\n p.v = p.z[0, 0];\n", attribute="z") == [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

    def test_schema_default_fill(self) -> None:
        """
        Tests that a fill() schema default fills with the given value.
        """

        app: Nadata = self._app()

        assert self._value(app=app, data_text="p = P();\n p.v = p.f[0, 0];\n", attribute="f") == [
            2.5, 2.5, 2.5, 2.5,
        ]

    def test_schema_default_eye_symbolic(self) -> None:
        """
        Tests that eye() works on a symbolic square shape.
        """

        app: Nadata = self._app()

        assert self._value(app=app, data_text="p = P();\n p.v = p.e[0, 0];\n", attribute="e") == [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

    def test_constructor_argument_fill(self) -> None:
        """
        Tests that an initializer passed to the constructor materializes.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P(z = fill(7.0));\n p.v = p.z[0, 0];\n",
            attribute="z",
        ) == [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]

    def test_typed_local_zeros(self) -> None:
        """
        Tests that an initializer materializes into a typed local.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text=(
                "p = P();\n"
                "m: Tensor[Float32, 2, 2] = zeros();\n"
                "p.v = m[1, 1];\n"
            ),
            attribute="v",
        ) == 0.0

    def test_eye_requires_square(self) -> None:
        """
        Tests that eye() on a non-square shape raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="square"):
            app.execute_data(data_text=(
                "p = P();\n"
                "bad: Tensor[Float32, 2, 3] = eye();\n"
            ))

    def test_zeros_require_numeric(self) -> None:
        """
        Tests that zeros() on a non-numeric element type raises TypeError.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text="class Q { s: Tensor[Str, 2] = zeros(); }")

        with pytest.raises(TypeError, match="numeric"):
            app.execute_data(data_text="q = Q();\n q.s[0] = \"x\";\n")

    def test_initializer_outside_tensor_rejected(self) -> None:
        """
        Tests that an initializer under a non-tensor target fails loudly.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.v = zeros();\n"
            ))


class TestElementwiseOps:
    """
    Tests elementwise and broadcast tensor arithmetic.
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

    def _value(self, app: Nadata, data_text: str, attribute: str) -> Any:
        """
        Runs a script and returns one attribute of the probe instance.

        Args:
            app (Nadata): Configured application.
            data_text (str): Script touching 'p'.
            attribute (str): Attribute to read.

        Returns:
            Any: The raw attribute payload.
        """

        app.execute_data(data_text=data_text)

        probe: Any = app.query(struct_name="P").last()

        return probe.attributes[attribute].get()

    def test_tensor_tensor_add(self) -> None:
        """
        Tests elementwise addition of same-shaped tensors.
        """

        app: Nadata = self._app()

        assert self._value(app=app, data_text="p = P();\n p.z = p.w + p.u;\n", attribute="z") == [
            11.0, 22.0, 33.0, 44.0, 55.0, 66.0,
        ]

    def test_tensor_tensor_subtract(self) -> None:
        """
        Tests elementwise subtraction of same-shaped tensors.
        """

        app: Nadata = self._app()

        assert self._value(app=app, data_text="p = P();\n p.z = p.u - p.w;\n", attribute="z") == [
            9.0, 18.0, 27.0, 36.0, 45.0, 54.0,
        ]

    def test_scalar_broadcast_multiply(self) -> None:
        """
        Tests scalar broadcast multiplication.
        """

        app: Nadata = self._app()

        assert self._value(app=app, data_text="p = P();\n p.z = p.w * 2.0;\n", attribute="z") == [
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
        ]

    def test_scalar_first_divide(self) -> None:
        """
        Tests scalar-first broadcasting (scalar / tensor).
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.z = 12.0 / p.w;\n",
            attribute="z",
        ) == [12.0, 6.0, 4.0, 3.0, 2.4, 2.0]

    def test_shape_mismatch_rejected(self) -> None:
        """
        Tests that tensor-tensor ops on mismatched shapes raise
        ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="shape mismatch"):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.z = p.w + p.e;\n"
            ))

    def test_int_tensor_division_rejected(self) -> None:
        """
        Tests that division on integer element types raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="integer element type"):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.n = p.n / 2;\n"
            ))


class TestReductions:
    """
    Tests the sum/min/max/transpose/fill tensor members.
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

    def _value(self, app: Nadata, data_text: str, attribute: str) -> Any:
        """
        Runs a script and returns one attribute of the probe instance.

        Args:
            app (Nadata): Configured application.
            data_text (str): Script touching 'p'.
            attribute (str): Attribute to read.

        Returns:
            Any: The raw attribute payload.
        """

        app.execute_data(data_text=data_text)

        probe: Any = app.query(struct_name="P").last()

        return probe.attributes[attribute].get()

    def test_sum(self) -> None:
        """
        Tests summing an integer tensor.
        """

        app: Nadata = self._app()

        assert self._value(app=app, data_text="p = P();\n p.i = p.n.sum();\n", attribute="i") == 10

    def test_min_and_max(self) -> None:
        """
        Tests min() and max() reductions.
        """

        app: Nadata = self._app()

        assert self._value(app=app, data_text="p = P();\n p.i = p.n.min();\n", attribute="i") == 1

        assert self._value(app=app, data_text="p = P();\n p.i = p.n.max();\n", attribute="i") == 4

    def test_sum_of_expression_result(self) -> None:
        """
        Tests that reductions chain onto arithmetic results.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.i = (p.n * 3).sum();\n",
            attribute="i",
        ) == 30

    def test_transpose(self) -> None:
        """
        Tests transposing a square matrix.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text=(
                "p = P();\n"
                "p.r = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];\n"
                "p.r = p.r.transpose();\n"
            ),
            attribute="r",
        ) == [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]

    def test_transpose_requires_square(self) -> None:
        """
        Tests that transpose() on a non-square tensor raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="square"):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.z = p.w.transpose();\n"
            ))

    def test_fill_member(self) -> None:
        """
        Tests the in-place fill() member.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.w.fill(9.0);\n",
            attribute="w",
        ) == [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]


class TestConvolutions:
    """
    Tests conv1d/conv2d valid cross-correlation with strides.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with a probe schema.

        Returns:
            Nadata: Configured application.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P {\n"
            "  sig: Tensor[Float32, 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\n"
            "  img: Tensor[Float32, 4, 4] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, "
            "9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];\n"
            "  k1: Tensor[Float32, 3] = [0.25, 0.5, 0.25];\n"
            "  k2: Tensor[Float32, 2, 2] = [1.0, 0.0, 0.0, 1.0];\n"
            "  o6: Tensor[Float32, 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  o3: Tensor[Float32, 3] = [0.0, 0.0, 0.0];\n"
            "  m3: Tensor[Float32, 3, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "}"
        ))

        return app

    def _value(self, app: Nadata, data_text: str, attribute: str) -> Any:
        """
        Runs a script and reads one attribute of the newest instance.

        Args:
            app (Nadata): Configured application.
            data_text (str): Script touching 'p'.
            attribute (str): Attribute to read.

        Returns:
            Any: The raw attribute payload.
        """

        app.execute_data(data_text=data_text)

        probe: Any = app.query(struct_name="P").last()

        return probe.attributes[attribute].get()

    def test_conv1d(self) -> None:
        """
        Tests 1-D convolution output values.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.o6 = p.sig.conv1d(p.k1);\n",
            attribute="o6",
        ) == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def test_conv1d_stride(self) -> None:
        """
        Tests strided 1-D convolution.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.o3 = p.sig.conv1d(p.k1, 2);\n",
            attribute="o3",
        ) == [2.0, 4.0, 6.0]

    def test_conv2d(self) -> None:
        """
        Tests 2-D convolution output values.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.m3 = p.img.conv2d(p.k2);\n",
            attribute="m3",
        ) == [7.0, 9.0, 11.0, 15.0, 17.0, 19.0, 23.0, 25.0, 27.0]

    def test_conv1d_rank_rejected(self) -> None:
        """
        Tests that conv1d on a 2-D tensor raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="1-D"):
            app.execute_data(data_text="p = P();\n p.o3 = p.img.conv1d(p.k1);\n")

    def test_conv_kernel_too_large_rejected(self) -> None:
        """
        Tests that a kernel larger than the input raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="larger"):
            app.execute_data(data_text=(
                "p = P();\n"
                "big: Tensor[Float32, 9] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\n"
                "p.o3 = p.sig.conv1d(big);\n"
            ))


class TestPooling:
    """
    Tests max/min/avg pooling on 1-D and 2-D tensors.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with a probe schema.

        Returns:
            Nadata: Configured application.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P {\n"
            "  sig: Tensor[Float32, 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\n"
            "  img: Tensor[Float32, 4, 4] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, "
            "9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];\n"
            "  o4: Tensor[Float32, 4] = [0.0, 0.0, 0.0, 0.0];\n"
            "  o7: Tensor[Float32, 7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  m2: Tensor[Float32, 2, 2] = [0.0, 0.0, 0.0, 0.0];\n"
            "}"
        ))

        return app

    def _value(self, app: Nadata, data_text: str, attribute: str) -> Any:
        """
        Runs a script and reads one attribute of the newest instance.

        Args:
            app (Nadata): Configured application.
            data_text (str): Script touching 'p'.
            attribute (str): Attribute to read.

        Returns:
            Any: The raw attribute payload.
        """

        app.execute_data(data_text=data_text)

        probe: Any = app.query(struct_name="P").last()

        return probe.attributes[attribute].get()

    def test_max_pool_1d(self) -> None:
        """
        Tests non-overlapping 1-D max pooling.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.o4 = p.sig.max_pool(2);\n",
            attribute="o4",
        ) == [2.0, 4.0, 6.0, 8.0]

    def test_min_pool_strided(self) -> None:
        """
        Tests strided 1-D min pooling.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.o7 = p.sig.min_pool(2, 1);\n",
            attribute="o7",
        ) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def test_avg_pool_1d(self) -> None:
        """
        Tests 1-D average pooling.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.o4 = p.sig.avg_pool(2);\n",
            attribute="o4",
        ) == [1.5, 3.5, 5.5, 7.5]

    def test_max_pool_2d(self) -> None:
        """
        Tests 2-D max pooling.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.m2 = p.img.max_pool(2);\n",
            attribute="m2",
        ) == [6.0, 8.0, 14.0, 16.0]

    def test_avg_pool_on_int_rejected(self) -> None:
        """
        Tests that average pooling an integer tensor raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="float element"):
            app.execute_data(data_text=(
                "p = P();\n"
                "n: Tensor[Int32, 4] = [1, 2, 3, 4];\n"
                "p.o4 = n.avg_pool(2);\n"
            ))

    def test_pool_window_too_large_rejected(self) -> None:
        """
        Tests that a window larger than the input raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="larger"):
            app.execute_data(data_text="p = P();\n p.o4 = p.sig.max_pool(9);\n")
