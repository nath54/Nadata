"""
Autograd tests: tape recording, backward rules vs finite differences,
flags and setters, conversions, the partition data pipeline, and SGD
training through the full engine.
"""

# Import Modules
from __future__ import annotations
import random
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase


class TestFlagsAndRecording:
    """
    Tests learnable/trackable flags, setters, and tape recording.
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
            "  w: Tensor[Float32, 2, 2] = [1.0, 2.0, 3.0, 4.0];\n"
            "  g: Tensor[Float32, 2, 2] = [0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        ))

        return app

    def _probe(self, app: Nadata) -> Any:
        """
        Returns the newest probe instance.

        Args:
            app (Nadata): Configured application.

        Returns:
            Any: The probe instance.
        """

        return app.query(struct_name="P").last()

    def test_backward_deposits_grad(self) -> None:
        """
        Tests that backward deposits a gradient on a learnable leaf.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
            "p.g = p.w.grad();\n"
        ))

        assert self._probe(app).attributes["g"].get() == [2.0, 4.0, 6.0, 8.0]

    def test_grad_is_zero_before_backward(self) -> None:
        """
        Tests that grad() returns zeros when nothing was computed.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="p = P();\n p.g = p.w.grad();\n")

        assert self._probe(app).attributes["g"].get() == [0.0, 0.0, 0.0, 0.0]

    def test_set_trackable_prunes_recording(self) -> None:
        """
        Tests that an untracked subgraph records nothing.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "set_trackable(p, false);\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
            "p.g = p.w.grad();\n"
        ))

        assert self._probe(app).attributes["g"].get() == [0.0, 0.0, 0.0, 0.0]

    def test_set_learnable_stops_deposit(self) -> None:
        """
        Tests that a non-learnable leaf flows gradient through but never
        deposits.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "set_learnable(p.w, false);\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
            "p.g = p.w.grad();\n"
        ))

        assert self._probe(app).attributes["g"].get() == [0.0, 0.0, 0.0, 0.0]

    def test_init_kwargs(self) -> None:
        """
        Tests learnable/trackable flags on tensor initializers.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class Q {\n"
            "  frozen: Tensor[Float32, 2] = randn(3, learnable = false);\n"
            "  loose: Tensor[Float32, 2] = zeros(trackable = false);\n"
            "}"
        ))

        app.execute_data(data_text=(
            "q = Q();\n"
            "loss = (q.loose * q.loose).sum();\n"
            "backward(loss);\n"
        ))

        probe: Any = app.query(struct_name="Q").last()

        # Untracked receivers record nothing, learnable=false leaves no
        # deposit even through a tracked path
        assert probe.attributes["loose"].grad is None

        app.execute_data(data_text=(
            "q = Q();\n"
            "loss = (q.frozen * q.frozen).sum();\n"
            "backward(loss);\n"
        ))

        probe = app.query(struct_name="Q").last()

        assert probe.attributes["frozen"].grad is None

    def test_mutation_between_forward_and_backward_raises(self) -> None:
        """
        Tests the version counter detecting unsafe in-place mutation.
        """

        app: Nadata = self._app()

        with pytest.raises(RuntimeError, match="mutated"):
            app.execute_data(data_text=(
                "p = P();\n"
                "loss = (p.w * p.w).sum();\n"
                "p.w.set_at(0, 0, 9.0);\n"
                "backward(loss);\n"
            ))


class TestGradientRules:
    """
    Validates every backward rule against central finite differences.
    """

    def _numeric_grad(
        self,
        app: Nadata,
        base_cells: list[float],
        loss_expr: str,
        extra: str = "",
        eps: float = 1e-3,
    ) -> list[float]:
        """
        Computes a numeric gradient cell by cell.

        Args:
            app (Nadata): Configured application.
            base_cells (list[float]): Base parameter cells.
            loss_expr (str): Loss expression over 'p.w' and 'p.u'.
            extra (str, optional): Extra attributes for the schema.
            eps (float, optional): Step size. Defaults to 1e-3.

        Returns:
            list[float]: The numeric gradient.
        """

        grads: list[float] = []

        for index in range(len(base_cells)):

            values: list[float] = []

            for direction in (-1.0, 1.0):

                cells: list[float] = list(base_cells)
                cells[index] += direction * eps

                app.execute_data(data_text=(
                    "p = P();\n"
                    "p.w = " + repr(cells) + ";\n"
                    + extra
                    + "p.out = " + loss_expr + ";\n"
                ))

                probe: Any = app.query(struct_name="P").last()
                values.append(probe.attributes["out"].get())

            grads.append((values[1] - values[0]) / (2.0 * eps))

        return grads

    def _analytic_grad(
        self,
        app: Nadata,
        schema: str,
        loss_expr: str,
        extra: str = "",
    ) -> list[float]:
        """
        Computes the analytic gradient through the tape.

        Args:
            app (Nadata): Configured application.
            schema (str): Probe schema body.
            loss_expr (str): Loss expression over 'p.w' and 'p.u'.
            extra (str, optional): Extra attributes for the schema.

        Returns:
            list[float]: The deposited gradient.
        """

        app.execute_schema(schema_text=schema)

        app.execute_data(data_text=(
            "p = P();\n"
            + extra
            + "loss = " + loss_expr + ";\n"
            "backward(loss);\n"
            "p.g = p.w.grad();\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        grad: Any = probe.attributes["g"].get()

        return grad  # type: ignore[no-any-return]

    def check_case(
        self,
        schema: str,
        loss_expr: str,
        base_cells: list[float],
        extra: str = "",
    ) -> None:
        """
        Asserts analytic and numeric gradients agree.

        Args:
            schema (str): Probe schema body.
            loss_expr (str): Loss expression.
            base_cells (list[float]): Base parameter cells.
            extra (str, optional): Extra attributes.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        analytic: list[float] = self._analytic_grad(
            app=app, schema=schema, loss_expr=loss_expr, extra=extra
        )
        numeric: list[float] = self._numeric_grad(
            app=app,
            base_cells=base_cells,
            loss_expr=loss_expr,
            extra=extra,
        )

        assert len(analytic) == len(numeric)

        for a, n in zip(analytic, numeric):
            assert a == pytest.approx(n, abs=1e-3)

    def test_arithmetic_rules(self) -> None:
        """
        Checks add/sub/mul/div against finite differences.
        """

        schema: str = (
            "class P {\n"
            "  w: Tensor[Float32, 2] = [1.5, -2.0];\n"
            "  u: Tensor[Float32, 2] = [3.0, 0.5];\n"
            "  g: Tensor[Float32, 2] = [0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema, loss_expr="(p.w + p.u).sum()",
            base_cells=[1.5, -2.0],
        )
        self.check_case(
            schema=schema, loss_expr="(p.w - p.u).sum()",
            base_cells=[1.5, -2.0],
        )
        self.check_case(
            schema=schema, loss_expr="(p.w * p.u).sum()",
            base_cells=[1.5, -2.0],
        )
        self.check_case(
            schema=schema, loss_expr="(p.w / p.u).sum()",
            base_cells=[1.5, -2.0],
        )
        self.check_case(
            schema=schema, loss_expr="(p.w * 2.5).sum()",
            base_cells=[1.5, -2.0],
        )
        self.check_case(
            schema=schema, loss_expr="(p.w - 0.5).sum()",
            base_cells=[1.5, -2.0],
        )
        self.check_case(
            schema=schema, loss_expr="(2.0 - p.w).sum()",
            base_cells=[1.5, -2.0],
        )

    def test_matmul_rules(self) -> None:
        """
        Checks all four matmul rank combinations.
        """

        schema_2d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 2, 3] = [0.5, 1.5, -1.0, 2.0, 0.5, 1.0];\n"
            "  u: Tensor[Float32, 3, 2] = [1.0, 0.5, -0.5, 2.0, 1.5, 0.0];\n"
            "  g: Tensor[Float32, 2, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_2d, loss_expr="p.w.matmul(p.u).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0],
        )

        schema_mv: str = (
            "class P {\n"
            "  w: Tensor[Float32, 2, 3] = [0.5, 1.5, -1.0, 2.0, 0.5, 1.0];\n"
            "  u: Tensor[Float32, 3] = [1.0, -0.5, 2.0];\n"
            "  g: Tensor[Float32, 2, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_mv, loss_expr="p.w.matmul(p.u).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0],
        )

        schema_vm: str = (
            "class P {\n"
            "  w: Tensor[Float32, 3] = [0.5, 1.5, -1.0];\n"
            "  u: Tensor[Float32, 3, 2] = [1.0, 0.5, -0.5, 2.0, 1.5, 0.0];\n"
            "  g: Tensor[Float32, 3] = [0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_vm, loss_expr="p.w.matmul(p.u).sum()",
            base_cells=[0.5, 1.5, -1.0],
        )

        schema_dot: str = (
            "class P {\n"
            "  w: Tensor[Float32, 3] = [0.5, 1.5, -1.0];\n"
            "  u: Tensor[Float32, 3] = [1.0, -0.5, 2.0];\n"
            "  g: Tensor[Float32, 3] = [0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_dot, loss_expr="p.w.matmul(p.u) * 1.0",
            base_cells=[0.5, 1.5, -1.0],
        )

    def test_reduction_rules(self) -> None:
        """
        Checks sum, mean, max, and min.
        """

        schema: str = (
            "class P {\n"
            "  w: Tensor[Float32, 3] = [0.5, 1.5, -1.0];\n"
            "  g: Tensor[Float32, 3] = [0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema, loss_expr="p.w.sum() * 1.0",
            base_cells=[0.5, 1.5, -1.0],
        )
        self.check_case(
            schema=schema, loss_expr="p.w.mean() * 1.0",
            base_cells=[0.5, 1.5, -1.0],
        )
        self.check_case(
            schema=schema, loss_expr="p.w.max() * 1.0",
            base_cells=[0.5, 1.5, -1.0],
        )
        self.check_case(
            schema=schema, loss_expr="p.w.min() * 1.0",
            base_cells=[0.5, 1.5, -1.0],
        )

    def test_shape_rules(self) -> None:
        """
        Checks transpose, permute, reshape, and flatten.
        """

        schema_2d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 2, 3] = [0.5, 1.5, -1.0, 2.0, 0.5, 1.0];\n"
            "  g: Tensor[Float32, 2, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_2d, loss_expr="p.w.transpose().sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0],
        )
        self.check_case(
            schema=schema_2d, loss_expr="p.w.flatten().sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0],
        )
        self.check_case(
            schema=schema_2d, loss_expr="p.w.reshape(3, 2).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0],
        )

        schema_3d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 2, 2, 2] ="
            " [0.5, 1.5, -1.0, 2.0, 0.5, 1.0, -0.5, 1.25];\n"
            "  g: Tensor[Float32, 2, 2, 2] ="
            " [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_3d, loss_expr="p.w.permute(2, 0, 1).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0, -0.5, 1.25],
        )

    def test_concat_gather_slice_rules(self) -> None:
        """
        Checks concat, gather, and sliced reads.
        """

        schema_concat: str = (
            "class P {\n"
            "  w: Tensor[Float32, 2] = [0.5, 1.5];\n"
            "  u: Tensor[Float32, 2] = [2.0, -1.0];\n"
            "  g: Tensor[Float32, 2] = [0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_concat, loss_expr="p.w.concat(p.u).sum()",
            base_cells=[0.5, 1.5],
        )

        schema_concat2d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 1, 2] = [0.5, 1.5];\n"
            "  u: Tensor[Float32, 1, 2] = [2.0, -1.0];\n"
            "  g: Tensor[Float32, 1, 2] = [0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_concat2d,
            loss_expr="p.w.concat(p.u, 1).sum()",
            base_cells=[0.5, 1.5],
        )

        schema_gather: str = (
            "class P {\n"
            "  w: Tensor[Float32, 3, 2] ="
            " [0.5, 1.5, -1.0, 2.0, 0.5, 1.0];\n"
            "  idx: Tensor[Int64, 2] = [2, 0];\n"
            "  g: Tensor[Float32, 3, 2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_gather,
            loss_expr="(p.w.gather(p.idx) * p.w.gather(p.idx)).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0],
        )

        schema_slice: str = (
            "class P {\n"
            "  w: Tensor[Float32, 3, 3] ="
            " [0.5, 1.5, -1.0, 2.0, 0.5, 1.0, -0.5, 1.25, 2.5];\n"
            "  u: Tensor[Float32, 2, 2] = [1.0, 0.5, -0.5, 2.0];\n"
            "  g: Tensor[Float32, 3, 3] ="
            " [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_slice,
            loss_expr="(p.w[0:2, 1:3] * p.u).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0, -0.5, 1.25, 2.5],
        )

    def test_relu_rule(self) -> None:
        """
        Checks the relu subgradient away from the kink.
        """

        schema: str = (
            "class P {\n"
            "  w: Tensor[Float32, 3] = [0.5, 1.5, -1.0];\n"
            "  g: Tensor[Float32, 3] = [0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema, loss_expr="p.w.relu().sum()",
            base_cells=[0.5, 1.5, -1.0],
        )

    def test_conv_rules(self) -> None:
        """
        Checks conv1d and conv2d against finite differences.
        """

        schema_1d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 5] = [0.5, 1.5, -1.0, 2.0, 0.5];\n"
            "  u: Tensor[Float32, 3] = [1.0, -0.5, 2.0];\n"
            "  g: Tensor[Float32, 5] = [0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_1d, loss_expr="p.w.conv1d(p.u).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5],
        )
        self.check_case(
            schema=schema_1d, loss_expr="p.w.conv1d(p.u, 2).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5],
        )

        schema_2d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 4, 4] ="
            " [0.5, 1.5, -1.0, 2.0, 0.5, 1.0, -0.5, 1.25,"
            "  2.5, 0.25, 1.5, -0.75, 1.0, 2.0, 0.5, 1.25];\n"
            "  u: Tensor[Float32, 2, 2] = [1.0, 0.5, -0.5, 2.0];\n"
            "  g: Tensor[Float32, 4, 4] ="
            " [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,"
            "  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_2d, loss_expr="p.w.conv2d(p.u).sum()",
            base_cells=[
                0.5, 1.5, -1.0, 2.0, 0.5, 1.0, -0.5, 1.25,
                2.5, 0.25, 1.5, -0.75, 1.0, 2.0, 0.5, 1.25,
            ],
        )

    def test_pool_rules(self) -> None:
        """
        Checks max/min/avg pooling against finite differences.
        """

        schema_1d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 4] = [0.1, 0.9, 0.3, 0.7];\n"
            "  g: Tensor[Float32, 4] = [0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_1d, loss_expr="p.w.max_pool(2).sum()",
            base_cells=[0.1, 0.9, 0.3, 0.7],
        )
        self.check_case(
            schema=schema_1d, loss_expr="p.w.min_pool(2).sum()",
            base_cells=[0.1, 0.9, 0.3, 0.7],
        )
        self.check_case(
            schema=schema_1d, loss_expr="p.w.avg_pool(2).sum()",
            base_cells=[0.1, 0.9, 0.3, 0.7],
        )
        self.check_case(
            schema=schema_1d, loss_expr="p.w.max_pool(2, 1).sum()",
            base_cells=[0.1, 0.9, 0.3, 0.7],
        )

        schema_2d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 2, 4] ="
            " [0.1, 0.9, 0.3, 0.7, 1.2, 0.4, 1.8, 0.6];\n"
            "  g: Tensor[Float32, 2, 4] ="
            " [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_2d, loss_expr="p.w.max_pool(2, 1).sum()",
            base_cells=[0.1, 0.9, 0.3, 0.7, 1.2, 0.4, 1.8, 0.6],
        )
        self.check_case(
            schema=schema_2d, loss_expr="p.w.avg_pool(2).sum()",
            base_cells=[0.1, 0.9, 0.3, 0.7, 1.2, 0.4, 1.8, 0.6],
        )

    def test_activation_rules(self) -> None:
        """
        Checks sigmoid, exp, gelu, and softmax against finite differences.
        """

        schema_1d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 3] = [0.5, 1.5, -1.0];\n"
            "  g: Tensor[Float32, 3] = [0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_1d, loss_expr="p.w.sigmoid().sum()",
            base_cells=[0.5, 1.5, -1.0],
        )
        self.check_case(
            schema=schema_1d, loss_expr="p.w.exp().sum()",
            base_cells=[0.5, 1.5, -1.0],
        )
        self.check_case(
            schema=schema_1d, loss_expr="p.w.gelu().sum()",
            base_cells=[0.5, 1.5, -1.0],
        )

        schema_2d: str = (
            "class P {\n"
            "  w: Tensor[Float32, 2, 3] = [0.5, 1.5, -1.0, 2.0, 0.5, 1.0];\n"
            "  g: Tensor[Float32, 2, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        )

        self.check_case(
            schema=schema_2d, loss_expr="p.w.softmax().sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0],
        )
        self.check_case(
            schema=schema_2d,
            loss_expr="(p.w.softmax() * p.w.softmax()).sum()",
            base_cells=[0.5, 1.5, -1.0, 2.0, 0.5, 1.0],
        )


class TestConversions:
    """
    Tests array_to_tensor and tensor_to_array.
    """

    def _app(self) -> Nadata:
        """
        Builds a configured application.

        Returns:
            Nadata: The application.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P {\n"
            "  w: Tensor[Float32, 2, 3] = zeros();\n"
            "  out: Float32 = 0.0;\n"
            "}"
        ))

        return app

    def test_roundtrip(self) -> None:
        """
        Tests tensor -> array -> tensor equality.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\n"
            "arr = tensor_to_array(p.w);\n"
            "t2: Tensor[Float32, 2, 3] = array_to_tensor(arr);\n"
            "p.out = (t2 - p.w).sum();\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        assert probe.attributes["out"].get() == 0.0

    def test_explicit_conversions(self) -> None:
        """
        Tests the conversion functions and shape validation.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "t: Tensor[Float32, 2, 3] = array_to_tensor("
            "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);\n"
            "p.w = t;\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        assert probe.attributes["w"].get() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        app.execute_data(data_text=(
            "p = P();\n"
            "p.w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\n"
            "arr = tensor_to_array(p.w);\n"
            "t2: Tensor[Float32, 2, 3] = array_to_tensor(arr);\n"
            "p.out = t2.sum();\n"
        ))

        probe = app.query(struct_name="P").last()

        assert probe.attributes["out"].get() == 21.0

    def test_cell_count_mismatch_rejected(self) -> None:
        """
        Tests that a shape-incompatible tensor fails assignment.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="Expected 6 elements"):
            app.execute_data(data_text=(
                "p = P();\n"
                "t: Tensor[Float32, 1, 1] = array_to_tensor([[7.0]]);\n"
                "p.w = t;\n"
            ))

    def test_ragged_nesting_rejected(self) -> None:
        """
        Tests that ragged nesting raises ValueError.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="ragged"):
            app.execute_data(data_text=(
                "p = P();\n"
                "t: Tensor[Float32, 2, 3] = array_to_tensor("
                "[[1.0, 2.0, 3.0], [4.0, 5.0]]);\n"
            ))

    def test_conversion_flags(self) -> None:
        """
        Tests learnable/trackable flags on array_to_tensor.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "t: Tensor[Float32, 2, 3] = array_to_tensor("
            "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], learnable = false);\n"
            "p.w = t;\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        assert probe.attributes["w"].grad is None


class TestPartitions:
    """
    Tests the partition data pipeline.
    """

    def _app(self) -> Nadata:
        """
        Builds a configured application.

        Returns:
            Nadata: The application.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P { v: Float64 = 0.0; }"
        ))

        return app

    def test_count_and_batches(self) -> None:
        """
        Tests iteration counts, short tails, and hard stops.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "rows = [[1.0], [2.0], [3.0], [4.0], [5.0]];\n"
            "p.v = get_partition_iteration_count(rows, 7, [1.0], 2, 0);\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        assert probe.attributes["v"].get() == 3

        app.execute_data(data_text=(
            "p = P();\n"
            "rows = [[1.0], [2.0], [3.0], [4.0], [5.0]];\n"
            "b0 = get_batch_of_partition(rows, 7, [1.0], 2, 0, 0);\n"
            "b1 = get_batch_of_partition(rows, 7, [1.0], 2, 0, 1);\n"
            "b2 = get_batch_of_partition(rows, 7, [1.0], 2, 0, 2);\n"
            "b3 = get_batch_of_partition(rows, 7, [1.0], 2, 0, 3);\n"
            "p.v = b2.length() * 10 + b3.length();\n"
        ))

        probe = app.query(struct_name="P").last()

        # Tail batch has 1 element; iteration 3 is exhausted
        assert probe.attributes["v"].get() == 10

    def test_partitions_cover_dataset(self) -> None:
        """
        Tests that weighted partitions cover the dataset exactly.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "rows = " + repr([[float(i)] for i in range(10)]) + ";\n"
            "w = [0.5, 0.3, 0.2];\n"
            "c0 = get_partition_iteration_count(rows, 7, w, 2, 0);\n"
            "c1 = get_partition_iteration_count(rows, 7, w, 2, 1);\n"
            "c2 = get_partition_iteration_count(rows, 7, w, 2, 2);\n"
            "p.v = c0 * 100 + c1 * 10 + c2;\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        # 5, 3, 2 examples -> 3, 2, 1 batches
        assert probe.attributes["v"].get() == 321

    def test_deterministic_order(self) -> None:
        """
        Tests that the same seed replays the identical order.
        """

        app: Nadata = self._app()

        results: list[Any] = []

        for _ in range(2):

            app.execute_data(data_text=(
                "p = P();\n"
                "rows = " + repr([[float(i)] for i in range(10)]) + ";\n"
                "b = get_batch_of_partition(rows, 7, [1.0], 3, 0, 0);\n"
                "p.v = b[0][0];\n"
            ))

            probe: Any = app.query(struct_name="P").last()
            results.append(probe.attributes["v"].get())

        assert results[0] == results[1]

    def test_invalid_weights_rejected(self) -> None:
        """
        Tests weight validation.
        """

        app: Nadata = self._app()

        with pytest.raises(ValueError, match="sum"):
            app.execute_data(data_text=(
                "p = P();\n"
                "rows = [[1.0]];\n"
                "n = get_partition_iteration_count(rows, 7, [0.5, 0.2], 1, 0);\n"
            ))


class TestActivations:
    """
    Tests activation member behavior and numerical stability.
    """

    def _probe_out(self, schema: str, body: str) -> Any:
        """
        Runs a script against a fresh schema and returns the probe output.

        Args:
            schema (str): Probe schema body.
            body (str): Script statements producing 'out'.

        Returns:
            Any: The 'out' attribute of the newest probe.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=schema)
        app.execute_data(data_text=body)

        probe: Any = app.query(struct_name="P").last()

        return probe.attributes["out"].get()

    _P3: str = (
        "class P {\n"
        "  w: Tensor[Float32, 3] = [0.5, 1.5, -1.0];\n"
        "  out: Float32 = 0.0;\n"
        "}"
    )

    def test_sigmoid_properties(self) -> None:
        """
        Tests sigmoid values on known inputs.
        """

        values: Any = self._probe_out(
            schema=self._P3,
            body=(
                "p = P();\n"
                "t: Tensor[Float32, 3] = p.w.sigmoid();\n"
                "p.out = t[0] * 10000.0 + t[1] + t[2];\n"
            ),
        )

        # sigmoid(0.5), sigmoid(1.5), sigmoid(-1.0) at full precision
        assert values == pytest.approx(
            0.62245933 * 10000.0 + 0.81757448 + 0.26894143, abs=1e-3
        )

    def test_sigmoid_extreme_stability(self) -> None:
        """
        Tests that extreme inputs do not overflow.
        """

        values: Any = self._probe_out(
            schema=(
                "class P {\n"
                "  w: Tensor[Float32, 3] = [50.0, -50.0, 0.0];\n"
                "  out: Float32 = 0.0;\n"
                "}"
            ),
            body=(
                "p = P();\n"
                "t: Tensor[Float32, 3] = p.w.sigmoid();\n"
                "p.out = t[0] + t[1] + t[2];\n"
            ),
        )

        # ~1 + ~0 + 0.5, no overflow
        assert values == pytest.approx(1.5, abs=1e-3)

    def test_softmax_rows_and_stability(self) -> None:
        """
        Tests softmax row normalization and max-shift stability.
        """

        values: Any = self._probe_out(
            schema=(
                "class P {\n"
                "  w: Tensor[Float32, 2, 3] ="
                " [1000.0, 1001.0, 1002.0, 0.0, 0.0, 0.0];\n"
                "  out: Float32 = 0.0;\n"
                "}"
            ),
            body=(
                "p = P();\n"
                "t: Tensor[Float32, 2, 3] = p.w.softmax();\n"
                "s0: Float32 = t[0, 0] + t[0, 1] + t[0, 2];\n"
                "s1: Float32 = t[1, 0] + t[1, 1] + t[1, 2];\n"
                "p.out = s0 + s1;\n"
            ),
        )

        # Both rows normalize to 1 even with a 1000-scale row
        assert values == pytest.approx(2.0, abs=1e-3)

    def test_gelu_zero(self) -> None:
        """
        Tests that gelu(0) is 0.
        """

        values: Any = self._probe_out(
            schema=(
                "class P {\n"
                "  w: Tensor[Float32, 1] = [0.0];\n"
                "  out: Float32 = 0.0;\n"
                "}"
            ),
            body=(
                "p = P();\n"
                "t: Tensor[Float32, 1] = p.w.gelu();\n"
                "p.out = t[0];\n"
            ),
        )

        assert values == 0.0


class TestAdam:
    """
    Tests the Adam optimizer.
    """

    def _probe(self, body: str) -> Any:
        """
        Runs a training body against a 1-cell probe schema.

        Args:
            body (str): Script statements.

        Returns:
            Any: The 'out' attribute of the newest probe.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P {\n"
            "  w: Tensor[Float32, 1] = [1.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        ))

        app.execute_data(data_text=body)

        probe: Any = app.query(struct_name="P").last()

        return probe.attributes["out"].get()

    def test_first_step_moves_by_lr(self) -> None:
        """
        Tests the bias-corrected first step: w - lr * sign(grad).
        """

        value: Any = self._probe(body=(
            "p = P();\n"
            "opt = Adam(lr = 0.1);\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
            "opt.step(p);\n"
            "p.out = p.w[0];\n"
        ))

        # First step: mhat = grad, vhat = grad^2 -> w - lr * grad/|grad|
        assert value == pytest.approx(0.9)

    def test_adam_converges(self) -> None:
        """
        Tests that Adam drives w to the minimum of (w - 2)^2.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P {\n"
            "  w: Tensor[Float32, 1] = [0.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        ))

        app.execute_data(data_text=(
            "p = P();\n"
            "opt = Adam(lr = 0.1);\n"
            "loss: Float32 = 0.0;\n"
            "for (i: Int64 = 0 :: i < 60 :: i++) {\n"
            "  loss = ((p.w - 2.0) * (p.w - 2.0)).sum();\n"
            "  backward(loss);\n"
            "  opt.step(p);\n"
            "}\n"
            "p.out = p.w[0];\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        # Adam's normalized steps oscillate around the optimum with
        # amplitude ~lr; it must land within one step of 2.0
        assert probe.attributes["out"].get() == pytest.approx(2.0, abs=0.11)

    def test_adam_state_persists_across_steps(self) -> None:
        """
        Tests that moments accumulate rather than reset every step.
        """

        value: Any = self._probe(body=(
            "p = P();\n"
            "opt = Adam(lr = 0.1);\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
            "opt.step(p);\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
            "opt.step(p);\n"
            "p.out = p.w[0];\n"
        ))

        # After step 1 w = 0.9; step 2 moves less than a full lr
        assert 0.75 < value < 0.9


class TestTraining:
    """
    Tests end-to-end SGD training through the engine.
    """

    def test_linear_regression_training(self) -> None:
        """
        Tests that a linear model learns a fixed target via the data
        pipeline, with the final loss pinned.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class Model {\n"
            "  w: Tensor[Float32, 2, 1] = randn(7);\n"
            "  b: Tensor[Float32, 1] = [0.0];\n"
            "  out: Float32 = 0.0;\n"
            "\n"
            "  @predict(x: Tensor[Float32, 2]) -> Tensor[Float32, 1] {\n"
            "    return x.matmul(self.w) + self.b;\n"
            "  }\n"
            "}"
        ))

        rng: random.Random = random.Random(0)
        dataset: list[list[float]] = []

        for _ in range(12):
            x0: float = rng.uniform(-2, 2)
            x1: float = rng.uniform(-2, 2)
            dataset.append([x0, x1, 3.0 * x0 - 2.0 * x1])

        app.execute_data(data_text=(
            "p = Model();\n"
            "opt = SGD(lr = 0.05);\n"
            "rows = " + repr(dataset) + ";\n"
            "total: Int64 = get_partition_iteration_count(rows, 42, [1.0], 4, 0);\n"
            "loss_acc: Float32 = 0.0;\n"
            "for (epoch: Int64 = 0 :: epoch < 8 :: epoch++) {\n"
            "  loss_acc = 0.0;\n"
            "  for (it: Int64 = 0 :: it < total :: it++) {\n"
            "    batch: Array[Array[Float32]] ="
            " get_batch_of_partition(rows, 42, [1.0], 4, 0, it);\n"
            "    for (r: Array[Float32] in batch) {\n"
            "      x: Tensor[Float32, 2] = [r[0], r[1]];\n"
            "      y: Tensor[Float32, 1] = [r[2]];\n"
            "      pred: Tensor[Float32, 1] = p.predict(x);\n"
            "      diff: Tensor[Float32, 1] = pred - y;\n"
            "      loss: Float32 = (diff * diff).mean();\n"
            "      loss_acc = loss_acc + loss;\n"
            "      backward(loss);\n"
            "    }\n"
            "    opt.step(p);\n"
            "  }\n"
            "}\n"
            "p.out = loss_acc / 8.0;\n"
        ))

        probe: Any = app.query(struct_name="Model").last()

        assert probe.attributes["out"].get() < 0.01

    def test_mlp_with_gelu_and_adam(self) -> None:
        """
        Tests a one-hidden-layer network (gelu activation) trained with
        Adam through the data pipeline, final loss pinned.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class Net {\n"
            "  w1: Tensor[Float32, 2, 4] = randn(11);\n"
            "  b1: Tensor[Float32, 4] = [0.0, 0.0, 0.0, 0.0];\n"
            "  w2: Tensor[Float32, 4, 1] = randn(12);\n"
            "  b2: Tensor[Float32, 1] = [0.0];\n"
            "  out: Float32 = 0.0;\n"
            "\n"
            "  @predict(x: Tensor[Float32, 2]) -> Tensor[Float32, 1] {\n"
            "    h: Tensor[Float32, 4] = (x.matmul(self.w1) + self.b1).gelu();\n"
            "    return h.matmul(self.w2) + self.b2;\n"
            "  }\n"
            "}"
        ))

        rng: random.Random = random.Random(1)
        dataset: list[list[float]] = []

        for _ in range(12):
            x0: float = rng.uniform(-2, 2)
            x1: float = rng.uniform(-2, 2)
            dataset.append([x0, x1, 3.0 * x0 - 2.0 * x1])

        app.execute_data(data_text=(
            "n = Net();\n"
            "opt = Adam(lr = 0.05);\n"
            "rows = " + repr(dataset) + ";\n"
            "total: Int64 = get_partition_iteration_count(rows, 7, [1.0], 4, 0);\n"
            "loss_acc: Float32 = 0.0;\n"
            "for (epoch: Int64 = 0 :: epoch < 80 :: epoch++) {\n"
            "  loss_acc = 0.0;\n"
            "  for (it: Int64 = 0 :: it < total :: it++) {\n"
            "    batch: Array[Array[Float32]] ="
            " get_batch_of_partition(rows, 7, [1.0], 4, 0, it);\n"
            "    for (r: Array[Float32] in batch) {\n"
            "      x: Tensor[Float32, 2] = [r[0], r[1]];\n"
            "      y: Tensor[Float32, 1] = [r[2]];\n"
            "      pred: Tensor[Float32, 1] = n.predict(x);\n"
            "      diff: Tensor[Float32, 1] = pred - y;\n"
            "      loss: Float32 = (diff * diff).mean();\n"
            "      loss_acc = loss_acc + loss;\n"
            "      backward(loss);\n"
            "    }\n"
            "    opt.step(n);\n"
            "  }\n"
            "}\n"
            "n.out = loss_acc / 80.0;\n"
        ))

        probe: Any = app.query(struct_name="Net").last()

        assert probe.attributes["out"].get() < 0.01

    def test_sgd_zero_grad(self) -> None:
        """
        Tests zero_grad clearing deposited gradients.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P {\n"
            "  w: Tensor[Float32, 2] = [1.0, 2.0];\n"
            "  g: Tensor[Float32, 2] = [0.0, 0.0];\n"
            "}"
        ))

        app.execute_data(data_text=(
            "p = P();\n"
            "opt = SGD(lr = 0.1);\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
            "opt.zero_grad(p);\n"
            "p.g = p.w.grad();\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        assert probe.attributes["g"].get() == [0.0, 0.0]

    def test_sgd_updates_parameters(self) -> None:
        """
        Tests that step() applies the gradient descent update.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P {\n"
            "  w: Tensor[Float32, 1] = [1.0];\n"
            "  out: Float32 = 0.0;\n"
            "}"
        ))

        app.execute_data(data_text=(
            "p = P();\n"
            "opt = SGD(lr = 0.1);\n"
            "loss = (p.w * p.w).sum();\n"
            "backward(loss);\n"
            "opt.step(p);\n"
            "p.out = p.w[0];\n"
        ))

        probe: Any = app.query(struct_name="P").last()

        # w = 1.0, grad = 2.0 -> w - 0.1 * 2.0 = 0.8
        assert probe.attributes["out"].get() == pytest.approx(0.8)
