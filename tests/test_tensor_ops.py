"""
Tests for tensor initializers, elementwise operations, and reductions.

Covers zeros()/fill()/eye() initialization (schema defaults, constructor
arguments, typed locals, symbolic dims), elementwise tensor-tensor and
broadcast tensor-scalar arithmetic, and the sum/min/max/transpose/fill
tensor members.
"""

# Import Modules
from __future__ import annotations
import math
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

    def test_transpose_non_square(self) -> None:
        """
        Tests transposing a rectangular matrix into a new shape.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text=(
                "p = P();\n"
                "p.z = p.w.transpose();\n"
            ),
            attribute="z",
        ) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

    def test_transpose_is_not_in_place(self) -> None:
        """
        Tests that the receiver keeps its shape, cells, and identity.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text=(
                "p = P();\n"
                "p.z = p.w.transpose();\n"
                "p.i = p.w.size();\n"
            ),
            attribute="i",
        ) == 6

    def test_transpose_result_is_independent(self) -> None:
        """
        Tests that mutating the copy leaves the original untouched.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text=(
                "p = P();\n"
                "p.z = p.w.transpose();\n"
                "p.z[0, 0] = 99.0;\n"
                "p.v = p.w[0, 0];\n"
            ),
            attribute="v",
        ) == 1.0

    def test_transpose_rejects_vectors(self) -> None:
        """
        Tests that transpose() on a 1-D tensor raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="2-D"):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.i = p.n.transpose().sum();\n"
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


class TestHigherOrder:
    """
    Tests value-wise function application (map) and math builtins.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with a probe schema and a named function.

        Returns:
            Nadata: Configured application.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "fn scale(v: Float32) -> Float32 { return v * 3.0; }\n"
            "class P {\n"
            "  v: Float32 = 0.0;\n"
            "  sig: Tensor[Float32, 4] = [1.0, -2.0, 3.0, -4.0];\n"
            "  out: Tensor[Float32, 4] = [0.0, 0.0, 0.0, 0.0];\n"
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

    def test_map_lambda(self) -> None:
        """
        Tests mapping a lambda over a tensor.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.out = p.sig.map((v: Float32) -> v * 2.0);\n",
            attribute="out",
        ) == [2.0, -4.0, 6.0, -8.0]

    def test_map_named_function(self) -> None:
        """
        Tests mapping a named standalone function over a tensor.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.out = p.sig.map(scale);\n",
            attribute="out",
        ) == [3.0, -6.0, 9.0, -12.0]

    def test_map_composition(self) -> None:
        """
        Tests composing map with math builtins and reductions.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text=(
                "p = P();\n"
                "p.out = p.sig.map((v: Float32) -> abs(v));\n"
                "p.v = p.out.sum();\n"
            ),
            attribute="v",
        ) == 10.0

    def test_map_result_type_enforced(self) -> None:
        """
        Tests that results violating the element type raise TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError):
            app.execute_data(data_text=(
                "p = P();\n"
                "p.out = p.sig.map((v: Float32) -> exp(v));\n"
                "p.out = p.sig.map((w: Float32) -> w > 0.0);\n"
            ))

    def test_map_requires_function(self) -> None:
        """
        Tests that a non-callable argument raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="lambda or function"):
            app.execute_data(data_text="p = P();\n p.out = p.sig.map(2.0);\n")

    def test_math_builtins(self) -> None:
        """
        Tests the exp/sqrt/log/pow math builtins.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.v = exp(0.0);\n",
            attribute="v",
        ) == 1.0

        assert self._value(
            app=app,
            data_text="p = P();\n p.v = sqrt(16.0);\n",
            attribute="v",
        ) == 4.0

        assert self._value(
            app=app,
            data_text="p = P();\n p.v = pow(2.0, 10.0);\n",
            attribute="v",
        ) == 1024.0

    def test_math_builtin_rejects_non_numeric(self) -> None:
        """
        Tests that math builtins reject non-numeric arguments.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="numeric"):
            app.execute_data(data_text="p = P();\n p.v = sqrt(\"x\");\n")


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


class TestMatmul:
    """
    Tests matrix multiplication across operand ranks.
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
            "  v: Float32 = 0.0;\n"
            "  a: Tensor[Float32, 2, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\n"
            "  b: Tensor[Float32, 3, 2] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];\n"
            "  u: Tensor[Float32, 3] = [1.0, 2.0, 3.0];\n"
            "  d: Float32 = 0.0;\n"
            "  r2: Tensor[Float32, 2, 2] = [0.0, 0.0, 0.0, 0.0];\n"
            "  r1: Tensor[Float32, 2] = [0.0, 0.0];\n"
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

    def test_matmul_2d_2d(self) -> None:
        """
        Tests full matrix-matrix products.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.r2 = p.a.matmul(p.b);\n",
            attribute="r2",
        ) == [58.0, 64.0, 139.0, 154.0]

    def test_matmul_2d_1d(self) -> None:
        """
        Tests matrix-vector products.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.r1 = p.a.matmul(p.u);\n",
            attribute="r1",
        ) == [14.0, 32.0]

    def test_matmul_1d_2d(self) -> None:
        """
        Tests vector-matrix products.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.r1 = p.u.matmul(p.b);\n",
            attribute="r1",
        ) == [58.0, 64.0]

    def test_matmul_1d_1d(self) -> None:
        """
        Tests dot products returning scalars.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.d = p.u.matmul(p.u);\n",
            attribute="d",
        ) == 14.0

    def test_matmul_mismatch_rejected(self) -> None:
        """
        Tests that inner dimension mismatches raise ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="mismatch"):
            app.execute_data(data_text="p = P();\n p.r2 = p.a.matmul(p.a);\n")


class TestArgreduce:
    """
    Tests argmax/argmin and mean reductions.
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
            "  v: Float32 = 0.0;\n"
            "  i: Int64 = 0;\n"
            "  ri: Tensor[Int64, 2] = [0, 0];\n"
            "  sig: Tensor[Float32, 4] = [3.0, -1.0, 4.0, 1.0];\n"
            "  img: Tensor[Float32, 2, 3] = [1.0, 9.0, 3.0, 7.0, 2.0, 5.0];\n"
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

    def test_argmax_1d(self) -> None:
        """
        Tests the argmax index of a 1-D tensor.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.i = p.sig.argmax();\n",
            attribute="i",
        ) == 2

    def test_argmin_1d(self) -> None:
        """
        Tests the argmin index of a 1-D tensor.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.i = p.sig.argmin();\n",
            attribute="i",
        ) == 1

    def test_argmax_2d_per_row(self) -> None:
        """
        Tests per-row argmax on a 2-D tensor.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.ri = p.img.argmax();\n",
            attribute="ri",
        ) == [1, 0]

    def test_mean(self) -> None:
        """
        Tests the mean reduction.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.v = p.sig.mean();\n",
            attribute="v",
        ) == 1.75

    def test_mean_on_int_rejected(self) -> None:
        """
        Tests that mean on an integer element type raises TypeError.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="float element"):
            app.execute_data(data_text=(
                "p = P();\n"
                "n: Tensor[Int32, 4] = [1, 2, 3, 4];\n"
                "p.v = n.mean();\n"
            ))


class TestShapeOps:
    """
    Tests concat, reshape, and flatten.
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
            "  a: Tensor[Float32, 2, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\n"
            "  b: Tensor[Float32, 1, 3] = [10.0, 20.0, 30.0];\n"
            "  r33: Tensor[Float32, 3, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,"
            " 0.0, 0.0, 0.0];\n"
            "  r26: Tensor[Float32, 2, 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,"
            " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  r6: Tensor[Float32, 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  r23: Tensor[Float32, 2, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
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

    def test_concat_axis0(self) -> None:
        """
        Tests row stacking for 2-D tensors.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.r33 = p.a.concat(p.b);\n",
            attribute="r33",
        ) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0]

    def test_concat_axis1(self) -> None:
        """
        Tests column joining for 2-D tensors.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.r26 = p.a.concat(p.a, 1);\n",
            attribute="r26",
        ) == [
            1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
        ]

    def test_flatten(self) -> None:
        """
        Tests flattening to 1-D.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.r6 = p.a.flatten();\n",
            attribute="r6",
        ) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_reshape(self) -> None:
        """
        Tests reshaping a 1-D tensor to 2-D.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.r6 = p.a.flatten();\n"
                      " p.r23 = p.r6.reshape(2, 3);\n",
            attribute="r23",
        ) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_reshape_infer_minus_one(self) -> None:
        """
        Tests the inferred -1 dimension.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.r23 = p.a.reshape(2, -1);\n",
            attribute="r23",
        ) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_reshape_mismatch_rejected(self) -> None:
        """
        Tests that a shape mismatch raises ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="does not match"):
            app.execute_data(data_text="p = P();\n p.r23 = p.a.reshape(5, 5);\n")

    def test_concat_row_mismatch_rejected(self) -> None:
        """
        Tests that mismatched rows on axis 1 raise ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="mismatch"):
            app.execute_data(data_text="p = P();\n p.r26 = p.a.concat(p.b, 1);\n")


class TestGather:
    """
    Tests embedding-style row gathering.
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
            "  emb: Tensor[Float32, 4, 2] = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0,"
            " 4.0, 4.0];\n"
            "  gat: Tensor[Float32, 2, 2] = [0.0, 0.0, 0.0, 0.0];\n"
            "  idx: Tensor[Int64, 2] = [0, 0];\n"
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

    def test_gather_array(self) -> None:
        """
        Tests gathering rows via an integer array literal.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.gat = p.emb.gather([1, 3]);\n",
            attribute="gat",
        ) == [2.0, 2.0, 4.0, 4.0]

    def test_gather_indices_tensor(self) -> None:
        """
        Tests gathering rows via an Int64 indices tensor.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.idx = [3, 0];\n"
                      " p.gat = p.emb.gather(p.idx);\n",
            attribute="gat",
        ) == [4.0, 4.0, 1.0, 1.0]

    def test_gather_out_of_range_rejected(self) -> None:
        """
        Tests that out-of-range indices raise IndexError.
        """

        app: Nadata = self._app()

        with pytest.raises(IndexError):
            app.execute_data(data_text="p = P();\n p.gat = p.emb.gather([4]);\n")


class TestSampling:
    """
    Tests topk index extraction.
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
            "  pos: Tensor[Int64, 5] = [0, 0, 0, 0, 0];\n"
            "  t3: Tensor[Int64, 3] = [0, 0, 0];\n"
            "  idx1: Tensor[Int64, 1] = [0];\n"
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

    def test_topk_indices(self) -> None:
        """
        Tests topk returning descending-order indices.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.pos = arange(1, 11, 2);\n"
                      " p.t3 = p.pos.topk(3);\n",
            attribute="t3",
        ) == [4, 3, 2]

    def test_topk_then_gather(self) -> None:
        """
        Tests the topk -> gather sampling composition.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text=(
                "p = P();\n"
                "p.pos = arange(1, 11, 2);\n"
                "p.idx1 = p.pos.gather(p.pos.topk(1));\n"
            ),
            attribute="idx1",
        ) == [9]

    def test_topk_invalid_count_rejected(self) -> None:
        """
        Tests that out-of-range counts raise ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="topk"):
            app.execute_data(data_text="p = P();\n p.t3 = p.pos.topk(9);\n")


class TestArangeAndTrig:
    """
    Tests arange and the sin/cos builtins.
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
            "  pos: Tensor[Int64, 5] = [0, 0, 0, 0, 0];\n"
            "  t3: Tensor[Int64, 3] = [0, 0, 0];\n"
            "  v: Float32 = 0.0;\n"
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

    def test_arange_stop(self) -> None:
        """
        Tests the single-argument form.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.pos = arange(5);\n",
            attribute="pos",
        ) == [0, 1, 2, 3, 4]

    def test_arange_start_stop_step(self) -> None:
        """
        Tests the three-argument form.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.pos = arange(1, 11, 2);\n",
            attribute="pos",
        ) == [1, 3, 5, 7, 9]

    def test_arange_slice(self) -> None:
        """
        Tests slicing an arange result.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.t3 = arange(5)[2:5];\n",
            attribute="t3",
        ) == [2, 3, 4]

    def test_arange_empty_rejected(self) -> None:
        """
        Tests that empty ranges raise ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="empty"):
            app.execute_data(data_text="p = P();\n p.pos = arange(0);\n")

    def test_sin_cos(self) -> None:
        """
        Tests the trigonometric builtins.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.v = sin(0.0) + cos(0.0);\n",
            attribute="v",
        ) == 1.0


class TestRandn:
    """
    Tests the randn tensor initializer.
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
            "  w: Tensor[Float32, 2, 2] = randn(42);\n"
            "  w2: Tensor[Float32, 2, 2] = randn(42);\n"
            "  z: Tensor[Float32, 2, 2] = zeros();\n"
            "}"
        ))

        return app

    def test_randn_seeded_deterministic(self) -> None:
        """
        Tests that equal seeds produce equal tensors.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="p = P();\n")

        probe: Any = app.query(struct_name="P").last()

        assert probe.w.get() == probe.w2.get()

    def test_randn_finite_floats(self) -> None:
        """
        Tests that randn produces finite float cells.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="p = P();\n")

        probe: Any = app.query(struct_name="P").last()

        cells: list[Any] = probe.w.get()

        assert len(cells) == 4

        for cell in cells:
            assert isinstance(cell, float)
            assert math.isfinite(cell)

    def test_randn_rejects_int_element(self) -> None:
        """
        Tests that randn under an integer element type fails.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        # Materialization is lazy: the schema resolves cleanly, the
        # error surfaces when the default is first read
        app.execute_schema(schema_text=(
            "class Q {\n"
            "  n: Tensor[Int32, 4] = randn(42);\n"
            "}"
        ))

        app.execute_data(data_text="q = Q();\n")

        probe: Any = app.query(struct_name="Q").last()

        with pytest.raises(TypeError, match="float element"):
            probe.get_attribute(name="n")

    def test_zeros_default_readable(self) -> None:
        """
        Tests reading an unset zeros() default through getattr.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="p = P();\n")

        probe: Any = app.query(struct_name="P").last()

        assert probe.z.get() == [0.0, 0.0, 0.0, 0.0]


class TestCasts:
    """
    Tests to_float/to_int element conversions.
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
            "  pos: Tensor[Float32, 4] = zeros();\n"
            "  idx: Tensor[Int64, 4] = [0, 0, 0, 0];\n"
            "  pe: Tensor[Float32, 4] = zeros();\n"
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

    def test_to_float(self) -> None:
        """
        Tests converting an Int tensor to floats.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.pos = arange(4).to_float();\n",
            attribute="pos",
        ) == [0.0, 1.0, 2.0, 3.0]

    def test_to_int(self) -> None:
        """
        Tests converting a Float tensor to ints.
        """

        app: Nadata = self._app()

        assert self._value(
            app=app,
            data_text="p = P();\n p.pos = [1.5, 2.5, 3.5, 4.5];\n"
                      " p.idx = p.pos.to_int();\n",
            attribute="idx",
        ) == [1, 2, 3, 4]

    def test_cast_then_map_pipeline(self) -> None:
        """
        Tests the arange -> to_float -> map(sin) pipeline.
        """

        app: Nadata = self._app()

        result: Any = self._value(
            app=app,
            data_text=(
                "p = P();\n"
                "p.pos = arange(4).to_float();\n"
                "p.pe = p.pos.map((v: Float32) -> sin(v / 4.0));\n"
            ),
            attribute="pe",
        )

        assert result[0] == 0.0
        assert round(result[3], 3) == 0.682
