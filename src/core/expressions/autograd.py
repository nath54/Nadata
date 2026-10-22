"""
Reverse-mode automatic differentiation for tensor values.

A Tape records one entry per differentiable operation performed during a
script execution pass. Each Record knows the operation's backward rule,
its input and output Values (by identity), small plain-data facts needed
by the rule, and the input version counters at record time.

Calling backward(loss) walks the records in reverse, accumulating
per-tensor gradients into a buffer keyed by value identity, and finally
deposits gradients onto every recorded input flagged learnable.

The engine stays dependency-free: every rule is index arithmetic over
flat row-major payloads, and the only imports beyond the standard
library are the engine's own modules.
"""

# Import Modules
from __future__ import annotations
import math
import random
from collections.abc import Callable, Iterator
from typing import Any, Optional, cast

from src.core.expressions.scope import ScopeAwareFunction
from src.core.types.types import (
    TypeArray,
    TypeBool,
    TypeFloat,
    TypeInt,
    TypeTensor,
)
from src.core.values.values import (
    Value,
    ValueArray,
    ValueExternal,
    ValueInt,
    ValueTensor,
)

# A backward rule maps the output gradient, the recorded input Values,
# and the saved plain-data facts to one gradient list per input (None
# for inputs that earn no gradient).
BackwardRule = Callable[
    [list[float], list[Value], dict[str, Any]],
    list[Optional[list[float]]],
]

# Weight-sum tolerance for partition validation (Float64 accumulation
# makes exact comparison false-positive)
WEIGHT_TOLERANCE: float = 1e-9


def _is_trackable_tensor(value: Value) -> bool:
    """
    Returns whether a Value is a Float tensor eligible for tracking.

    Args:
        value (Value): Candidate Value.

    Returns:
        bool: True for Float-element ValueTensors; Int/Bool/Str tensors
            are constants in the graph.
    """

    return (
        isinstance(value, ValueTensor)
        and isinstance(value.value_type.value_type, TypeFloat)
    )


def _extents(tensor: ValueTensor) -> list[int]:
    """
    Returns the concrete per-axis extents of a tensor.

    Args:
        tensor (ValueTensor): Tensor to inspect.

    Returns:
        list[int]: Extents in axis order.
    """

    if tensor.concrete_dims is not None:
        return list(tensor.concrete_dims)

    return [dim for dim in tensor.value_type.dims if isinstance(dim, int)]


def _shape_len(shape: list[int]) -> int:
    """
    Returns the number of cells described by a shape.

    Args:
        shape (list[int]): Per-axis extents.

    Returns:
        int: Total cell count.
    """

    total: int = 1

    for dim in shape:
        total *= dim

    return total


def _strides(shape: list[int]) -> list[int]:
    """
    Returns the row-major strides of a shape.

    Args:
        shape (list[int]): Per-axis extents.

    Returns:
        list[int]: One stride per axis (last stride is 1).
    """

    strides: list[int] = [1] * len(shape)
    running: int = 1

    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        running *= shape[axis]

    return strides


def _raw_cells(value: Value, shape: list[int]) -> list[Any]:
    """
    Normalizes an input Value to a flat cell list.

    Args:
        value (Value): Recorded input (tensor payload or scalar).
        shape (list[int]): The shape recorded for this input (empty for a
            scalar).

    Returns:
        list[Any]: Flat cell list.
    """

    raw: Any = value.get()

    if not shape:
        return [raw]

    return list(raw)


def _unbroadcast(grad: list[float], in_shape: list[int]) -> list[float]:
    """
    Reduces an output gradient to one input's shape.

    Only two broadcast cases exist in the engine: identical shapes and
    scalar broadcast, so the reduction is identity or a plain sum.

    Args:
        grad (list[float]): Output-shaped gradient.
        in_shape (list[int]): Input shape (empty for a scalar).

    Returns:
        list[float]: Input-shaped gradient.

    Raises:
        ValueError: If the input shape is neither the output shape nor a
            scalar (unsupported broadcast).
    """

    if not in_shape:
        return [sum(grad)]

    if _shape_len(in_shape) == len(grad):
        return list(grad)

    raise ValueError(
        f"Unsupported broadcast in backward: input shape {in_shape} vs "
        f"{len(grad)} gradient cell(s)"
    )


def _transpose2_cells(cells: list[Any], shape: list[int]) -> list[Any]:
    """
    Transposes a flat 2-D payload.

    Args:
        cells (list[Any]): Flat row-major payload of shape [rows, cols].
        shape (list[int]): The [rows, cols] shape.

    Returns:
        list[Any]: Flat payload of the transposed shape.
    """

    rows: int = shape[0]
    cols: int = shape[1]

    return [cells[col * cols + row] for row in range(cols) for col in range(rows)]


def _matmul_flat(
    a_cells: list[Any],
    a_shape: list[int],
    b_cells: list[Any],
    b_shape: list[int],
) -> tuple[list[Any], list[int]]:
    """
    Computes a matrix product over flat payloads (all four rank cases).

    Args:
        a_cells (list[Any]): Left operand flat payload.
        a_shape (list[int]): Left operand shape.
        b_cells (list[Any]): Right operand flat payload.
        b_shape (list[int]): Right operand shape.

    Returns:
        tuple[list[Any], list[int]]: Result payload and shape.

    Raises:
        ValueError: On an inner dimension mismatch.
    """

    inner: int = a_shape[-1]

    if inner != b_shape[0]:
        raise ValueError(
            f"matmul inner dimension mismatch: {a_shape} vs {b_shape}"
        )

    # 1-D x 1-D: scalar dot product
    if len(a_shape) == 1 and len(b_shape) == 1:
        return (
            [sum(a_cells[i] * b_cells[i] for i in range(inner))],
            [],
        )

    # 1-D x 2-D: vector-matrix product
    if len(a_shape) == 1:
        cols: int = b_shape[1]

        return (
            [
                sum(
                    a_cells[i] * b_cells[i * cols + col]
                    for i in range(inner)
                )
                for col in range(cols)
            ],
            [cols],
        )

    rows: int = a_shape[0]

    # 2-D x 1-D: matrix-vector product
    if len(b_shape) == 1:
        return (
            [
                sum(
                    a_cells[row * inner + i] * b_cells[i]
                    for i in range(inner)
                )
                for row in range(rows)
            ],
            [rows],
        )

    # 2-D x 2-D: full matrix product
    cols = b_shape[1]

    return (
        [
            sum(
                a_cells[row * inner + i] * b_cells[i * cols + col]
                for i in range(inner)
            )
            for row in range(rows)
            for col in range(cols)
        ],
        [rows, cols],
    )


class Record:
    """
    One recorded operation on the tape.

    Attributes:
        rule (BackwardRule): Backward rule for the operation.
        rule_name (str): Registry name (for error messages).
        inputs (list[Value]): Consumed Values by identity.
        output (Value): Produced Value by identity.
        saved (dict[str, Any]): Plain-data facts the rule needs.
        versions (list[int]): Input version counters at record time
            (-1 for non-tensor inputs).
    """

    def __init__(
        self,
        rule: BackwardRule,
        rule_name: str,
        inputs: list[Value],
        output: Value,
        saved: dict[str, Any],
        versions: list[int],
    ) -> None:
        """
        Initializes the Record instance.

        Args:
            rule (BackwardRule): Backward rule.
            rule_name (str): Registry name.
            inputs (list[Value]): Consumed Values.
            output (Value): Produced Value.
            saved (dict[str, Any]): Saved plain-data facts.
            versions (list[int]): Input version counters.
        """

        self.rule: BackwardRule = rule
        self.rule_name: str = rule_name
        self.inputs: list[Value] = inputs
        self.output: Value = output
        self.saved: dict[str, Any] = saved
        self.versions: list[int] = versions


class Tape:
    """
    Ordered record list for one script execution pass.

    Attributes:
        records (list[Record]): Records in execution (topological) order.
    """

    def __init__(self) -> None:
        """
        Initializes the Tape instance.
        """

        self.records: list[Record] = []
        self._scalar_outputs: set[int] = set()

    def maybe_record(
        self,
        rule_name: str,
        inputs: list[Value],
        output: Value,
        saved: dict[str, Any],
    ) -> None:
        """
        Records an operation if any Float input is tracked.

        The output's flags follow the inputs: tracked inherits, and a
        produced value is never learnable by default.

        Args:
            rule_name (str): Backward rule registry name.
            inputs (list[Value]): Consumed Values (tensors and scalars).
            output (Value): Produced Value.
            saved (dict[str, Any]): Plain-data facts for the rule.
        """

        tracked: bool = any(
            isinstance(value, ValueTensor)
            and isinstance(value.value_type.value_type, TypeFloat)
            and value.tracked
            for value in inputs
        )

        if isinstance(output, ValueTensor):
            output.tracked = tracked
            output.learnable = False

        if not tracked:
            return None

        self.records.append(
            Record(
                rule=BACKWARD_RULES[rule_name],
                rule_name=rule_name,
                inputs=list(inputs),
                output=output,
                saved=dict(saved),
                versions=[
                    value.version if isinstance(value, ValueTensor) else -1
                    for value in inputs
                ],
            )
        )

        # Scalar outputs (reductions) seed future scalar-chain records
        if not isinstance(output, ValueTensor):
            self._scalar_outputs.add(id(output))

        return None

    def maybe_record_scalar(
        self,
        op: str,
        inputs: list[Value],
        output: Value,
    ) -> None:
        """
        Records a scalar float arithmetic operation on the autograd path.

        A scalar op participates in the graph when at least one Float
        input is the output of an earlier record (e.g. a reduction over a
        tensor); constants and schema scalars flow through unchanged.
        Gradients accumulate in length-one buffers so the backward walk
        treats scalars and tensors uniformly.

        Args:
            op (str): Arithmetic operator ('+', '-', '*', '/').
            inputs (list[Value]): The two operand Values.
            output (Value): Produced scalar Value.
        """

        rules: dict[str, str] = {
            "+": "s_add",
            "-": "s_sub",
            "*": "s_mul",
            "/": "s_div",
        }

        if op not in rules:
            return None

        live: bool = any(
            isinstance(value.value_type, TypeFloat)
            and id(value) in self._scalar_outputs
            for value in inputs
        )

        if not live:
            return None

        self.records.append(
            Record(
                rule=BACKWARD_RULES[rules[op]],
                rule_name=rules[op],
                inputs=list(inputs),
                output=output,
                saved={},
                versions=[-1, -1],
            )
        )

        self._scalar_outputs.add(id(output))

        return None

    def backward(self, loss: Value) -> None:
        """
        Walks the records in reverse and deposits parameter gradients.

        Args:
            loss (Value): Terminal scalar or tensor Value to seed with
                ones.

        Raises:
            RuntimeError: If a recorded input was mutated between forward
                and backward.
        """

        records: list[Record] = self.records
        self.records = []

        loss_len: int = (
            len(loss.cells) if isinstance(loss, ValueTensor) else 1
        )

        buffers: dict[int, list[float]] = {id(loss): [1.0] * loss_len}

        for record in reversed(records):

            grad: Optional[list[float]] = buffers.get(id(record.output))

            if grad is None:
                continue

            # Detect in-place mutation of tracked inputs after recording
            for index, input_value in enumerate(record.inputs):

                if (
                    isinstance(input_value, ValueTensor)
                    and record.versions[index] != -1
                    and input_value.version != record.versions[index]
                ):
                    raise RuntimeError(
                        f"Tensor mutated between forward and backward "
                        f"(rule '{record.rule_name}'); copy or detach "
                        f"before writing in place"
                    )

            input_grads: list[Optional[list[float]]] = record.rule(
                grad, record.inputs, record.saved
            )

            for index, input_grad in enumerate(input_grads):

                if input_grad is None:
                    continue

                key: int = id(record.inputs[index])
                existing: Optional[list[float]] = buffers.get(key)

                if existing is None:
                    buffers[key] = list(input_grad)

                else:
                    buffers[key] = [
                        old + new for old, new in zip(existing, input_grad)
                    ]

        # Deposit onto learnable recorded inputs (parameters)
        deposited: set[int] = set()

        for record in records:

            for index, input_value in enumerate(record.inputs):

                if not isinstance(input_value, ValueTensor):
                    continue

                if not input_value.learnable or id(input_value) in deposited:
                    continue

                accumulated: Optional[list[float]] = buffers.get(
                    id(input_value)
                )

                if accumulated is None:
                    continue

                if input_value.grad is None:
                    input_value.grad = accumulated

                else:
                    input_value.grad = [
                        old + new
                        for old, new in zip(input_value.grad, accumulated)
                    ]

                deposited.add(id(input_value))

        return None


# --- Backward rules ---
# Each rule maps (output_grad, inputs, saved) to one gradient list per
# input; None marks an input that earns no gradient.


def _backward_add(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    add: both inputs pass the gradient through (unbroadcast).
    """

    return [
        _unbroadcast(grad, saved["shapes"][0]),
        _unbroadcast(grad, saved["shapes"][1]),
    ]


def _backward_sub(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    sub: left passes through, right negates.
    """

    return [
        _unbroadcast(grad, saved["shapes"][0]),
        [
            -cell
            for cell in _unbroadcast(grad, saved["shapes"][1])
        ],
    ]


def _elementwise_product_grads(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    Shared mul/div machinery: gradient of one operand given the other.

    Args:
        grad (list[float]): Output-shaped gradient.
        inputs (list[Value]): The two operands.
        saved (dict[str, Any]): Input shapes.

    Returns:
        list[list[float]]: [left_grad, right_grad].
    """

    shapes: list[list[int]] = saved["shapes"]
    left_cells: list[Any] = _raw_cells(inputs[0], shapes[0])
    right_cells: list[Any] = _raw_cells(inputs[1], shapes[1])

    left_grad: list[float]
    right_grad: list[float]

    if not shapes[0] and not shapes[1]:
        left_grad = [sum(grad)]
        right_grad = [sum(grad)]

    elif not shapes[0]:
        left_grad = [sum(grad[i] * right_cells[i] for i in range(len(grad)))]
        right_grad = [grad[i] * left_cells[0] for i in range(len(grad))]

    elif not shapes[1]:
        left_grad = [grad[i] * right_cells[0] for i in range(len(grad))]
        right_grad = [sum(grad[i] * left_cells[i] for i in range(len(grad)))]

    else:
        left_grad = [grad[i] * right_cells[i] for i in range(len(grad))]
        right_grad = [grad[i] * left_cells[i] for i in range(len(grad))]

    return [left_grad, right_grad]


def _backward_mul(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    mul: each input's gradient scales by the other operand.
    """

    return _elementwise_product_grads(grad, inputs, saved)


def _backward_div(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    div: d(a/b) = da/b - db*a/b^2 elementwise (with broadcasting).
    """

    shapes: list[list[int]] = saved["shapes"]
    a_cells: list[Any] = _raw_cells(inputs[0], shapes[0])
    b_cells: list[Any] = _raw_cells(inputs[1], shapes[1])

    left_grad: list[float]
    right_grad: list[float]

    if not shapes[0] and not shapes[1]:
        left_grad = [sum(grad)]
        right_grad = [sum(-grad[i] * a_cells[0] / (b_cells[0] ** 2)
                          for i in range(len(grad)))]

    elif not shapes[0]:
        left_grad = [sum(grad[i] / b_cells[i] for i in range(len(grad)))]
        right_grad = [
            -grad[i] * a_cells[0] / (b_cells[i] ** 2)
            for i in range(len(grad))
        ]

    elif not shapes[1]:
        left_grad = [grad[i] / b_cells[0] for i in range(len(grad))]
        right_grad = [
            sum(-grad[i] * a_cells[i] / (b_cells[0] ** 2)
                for i in range(len(grad)))
        ]

    else:
        left_grad = [grad[i] / b_cells[i] for i in range(len(grad))]
        right_grad = [
            -grad[i] * a_cells[i] / (b_cells[i] ** 2)
            for i in range(len(grad))
        ]

    return [left_grad, right_grad]


def _backward_s_add(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    Scalar add: both operands pass the gradient through.
    """

    return [grad, grad]


def _backward_s_sub(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    Scalar sub: left passes through, right negates.
    """

    return [grad, [-cell for cell in grad]]


def _backward_s_mul(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    Scalar mul: each operand's gradient scales by the other.
    """

    left: Any = inputs[0].get()
    right: Any = inputs[1].get()

    return [[grad[0] * right], [grad[0] * left]]


def _backward_s_div(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    Scalar div: d(a/b) = da/b - db*a/b^2.
    """

    left: Any = inputs[0].get()
    right: Any = inputs[1].get()

    return [
        [grad[0] / right],
        [-grad[0] * left / (right * right)],
    ]


def _backward_matmul(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    matmul: transposed products for 2-D operands, outer products for
    the vector cases, scaled operand for the dot product.
    """

    a_shape: list[int] = saved["shapes"][0]
    b_shape: list[int] = saved["shapes"][1]
    a_cells: list[Any] = _raw_cells(inputs[0], a_shape)
    b_cells: list[Any] = _raw_cells(inputs[1], b_shape)

    # 1-D x 1-D dot product: da = g * b, db = g * a
    if len(a_shape) == 1 and len(b_shape) == 1:
        return [
            [grad[0] * b_cells[i] for i in range(len(a_cells))],
            [grad[0] * a_cells[i] for i in range(len(b_cells))],
        ]

    # 1-D x 2-D vector-matrix: da = g @ b^T, db = outer(a, g)
    if len(a_shape) == 1:
        da_cells, _ = _matmul_flat(
            grad, [len(grad)], _transpose2_cells(b_cells, b_shape),
            [b_shape[1], b_shape[0]],
        )
        db_cells: list[Any] = [
            a_cells[i] * grad[col]
            for i in range(b_shape[0])
            for col in range(b_shape[1])
        ]
        return [list(da_cells), db_cells]

    # 2-D x 1-D matrix-vector: da = outer(g, b), db = a^T @ g
    if len(b_shape) == 1:
        da_cells, _ = _matmul_flat(
            grad,
            [a_shape[0], 1],
            list(b_cells),
            [1, b_shape[0]],
        )
        db_cells = [
            sum(a_cells[row * a_shape[1] + i] * grad[row]
                for row in range(a_shape[0]))
            for i in range(b_shape[0])
        ]
        return [list(da_cells), db_cells]

    # 2-D x 2-D: da = g @ b^T, db = a^T @ g
    da_cells, _ = _matmul_flat(
        grad,
        [a_shape[0], b_shape[1]],
        _transpose2_cells(b_cells, b_shape),
        [b_shape[1], b_shape[0]],
    )
    db_cells, _ = _matmul_flat(
        _transpose2_cells(a_cells, a_shape),
        [a_shape[1], a_shape[0]],
        grad,
        [a_shape[0], b_shape[1]],
    )
    return [list(da_cells), db_cells]


def _backward_sum(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    sum: the scalar gradient broadcasts back over every cell.
    """

    count: int = _shape_len(saved["shapes"][0])

    return [[grad[0]] * count]


def _backward_mean(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    mean: the scalar gradient divides by the cell count.
    """

    count: int = _shape_len(saved["shapes"][0])

    return [[grad[0] / count] * count]


def _backward_maxmin(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    max/min: the whole gradient routes to the extreme cell.
    """

    result: list[float] = [0.0] * _shape_len(saved["shapes"][0])
    result[saved["argmax"]] = grad[0]

    return [result]


def _backward_transpose(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    transpose: the gradient is the transpose of the output gradient.
    """

    return [_permuted(grad, saved["out_shape"], [1, 0])]


def _backward_permute(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    permute: the gradient applies the inverse permutation.
    """

    perm: list[int] = saved["perm"]
    inverse: list[int] = [0] * len(perm)

    for axis, source in enumerate(perm):
        inverse[source] = axis

    return [_permuted(grad, saved["out_shape"], inverse)]


def _permuted(
    cells: list[Any],
    shape: list[int],
    perm: list[int],
) -> list[Any]:
    """
    Permutes a flat row-major payload by an axis permutation.

    Args:
        cells (list[Any]): Source flat payload.
        shape (list[int]): Source shape.
        perm (list[int]): Output axis d addresses source axis perm[d].

    Returns:
        list[Any]: Permuted flat payload.
    """

    rank: int = len(shape)
    src_strides: list[int] = _strides(shape)

    out_shape: list[int] = [shape[axis] for axis in perm]
    out_strides: list[int] = _strides(out_shape)

    result: list[Any] = [None] * len(cells)

    for out_flat in range(len(cells)):

        remaining: int = out_flat
        src_flat: int = 0

        for axis in range(rank):
            coordinate: int = remaining // out_strides[axis]
            remaining %= out_strides[axis]
            src_flat += coordinate * src_strides[perm[axis]]

        result[out_flat] = cells[src_flat]

    return result


def _backward_reshape(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    reshape/flatten: row-major order is preserved, so the gradient is
    the same flat payload.
    """

    return [list(grad)]


def _backward_concat(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    concat: the gradient splits along the join axis.
    """

    a_shape: list[int] = saved["shapes"][0]
    b_shape: list[int] = saved["shapes"][1]
    axis: int = saved["axis"]

    a_len: int = _shape_len(a_shape)

    if axis == 0 or len(a_shape) == 1:
        return [list(grad[:a_len]), list(grad[a_len:])]

    a_rows: int = a_shape[0]
    a_cols: int = a_shape[1]
    b_cols: int = b_shape[1]

    a_grad: list[float] = []
    b_grad: list[float] = []

    for row in range(a_rows):

        base: int = row * (a_cols + b_cols)

        a_grad.extend(grad[base:base + a_cols])
        b_grad.extend(grad[base + a_cols:base + a_cols + b_cols])

    return [a_grad, b_grad]


def _backward_gather(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    gather: scatter-add each output row into its source row (duplicate
    indices accumulate - the embedding gradient).
    """

    table_shape: list[int] = saved["shapes"][0]
    rest: int = _shape_len(table_shape[1:]) if len(table_shape) > 1 else 1

    table_grad: list[float] = [0.0] * _shape_len(table_shape)

    for index, source_row in enumerate(saved["indices"]):
        for offset in range(rest):
            table_grad[source_row * rest + offset] += grad[
                index * rest + offset
            ]

    return [table_grad, None]


def _backward_slice(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    slice read: scatter the gradient block back into a zero tensor of
    the source shape.

    The saved spec holds one (is_slice, start, stop) tuple per source
    dimension; non-sliced dimensions were fixed integer indices saved
    in 'indices'.
    """

    src_shape: list[int] = saved["shapes"][0]
    spec: list[tuple[bool, int, int]] = saved["spec"]
    indices: list[int] = saved["indices"]

    out_shape: list[int] = [
        stop - start for is_slice, start, stop in spec if is_slice
    ]
    out_strides: list[int] = _strides(out_shape)

    src_grad: list[float] = [0.0] * _shape_len(src_shape)
    src_strides: list[int] = _strides(src_shape)

    slice_axes: list[int] = [
        axis for axis, (is_slice, _, _) in enumerate(spec) if is_slice
    ]

    for out_flat in range(len(grad)):

        remaining: int = out_flat
        src_flat: int = 0
        slice_position: int = 0

        for axis, (is_slice, start, _) in enumerate(spec):

            if is_slice:
                coordinate: int = (
                    remaining // out_strides[slice_position]
                )
                remaining %= out_strides[slice_position]
                slice_position += 1
                src_flat += (start + coordinate) * src_strides[axis]

            else:
                src_flat += indices[axis] * src_strides[axis]

        src_grad[src_flat] += grad[out_flat]

    return [src_grad]


def _backward_relu(
    grad: list[float],
    inputs: list[Value],
    saved: dict[str, Any],
) -> list[Optional[list[float]]]:
    """
    relu: the gradient passes where the input was positive.
    """

    input_cells: list[Any] = _raw_cells(inputs[0], saved["shapes"][0])

    return [
        [
            grad[i] if input_cells[i] > 0 else 0.0
            for i in range(len(grad))
        ]
    ]


BACKWARD_RULES: dict[str, BackwardRule] = {
    "add": _backward_add,
    "sub": _backward_sub,
    "mul": _backward_mul,
    "div": _backward_div,
    "matmul": _backward_matmul,
    "sum": _backward_sum,
    "mean": _backward_mean,
    "max": _backward_maxmin,
    "min": _backward_maxmin,
    "transpose": _backward_transpose,
    "permute": _backward_permute,
    "reshape": _backward_reshape,
    "flatten": _backward_reshape,
    "concat": _backward_concat,
    "gather": _backward_gather,
    "slice": _backward_slice,
    "relu": _backward_relu,
    "s_add": _backward_s_add,
    "s_sub": _backward_s_sub,
    "s_mul": _backward_s_mul,
    "s_div": _backward_s_div,
}

# Members recorded at the tensor member-call site, mapped to their rule
RECORDABLE_MEMBERS: dict[str, str] = {
    "matmul": "matmul",
    "sum": "sum",
    "mean": "mean",
    "max": "max",
    "min": "min",
    "transpose": "transpose",
    "permute": "permute",
    "reshape": "reshape",
    "flatten": "flatten",
    "concat": "concat",
    "gather": "gather",
    "relu": "relu",
}


def maybe_record_member(
    member: str,
    receiver: Value,
    args: list[Value],
    result: Value,
    tape: Optional[Tape],
) -> None:
    """
    Records a tensor member call if it is differentiable and tracked.

    Args:
        member (str): Called member name.
        receiver (Value): The tensor Value the member was called on.
        args (list[Value]): Evaluated call arguments.
        result (Value): Produced Value.
        tape (Optional[Tape]): Live tape, or None outside script
            execution.
    """

    if tape is None or member not in RECORDABLE_MEMBERS:
        return None

    saved: dict[str, Any] = {
        "shapes": [
            _extents(receiver) if isinstance(receiver, ValueTensor) else []
        ]
    }
    inputs: list[Value] = [receiver]

    for argument in args:
        saved["shapes"].append(
            _extents(argument) if isinstance(argument, ValueTensor) else []
        )
        inputs.append(argument)

    _fill_member_saved(
        member=member, receiver=receiver, args=args, saved=saved
    )

    tape.maybe_record(
        rule_name=RECORDABLE_MEMBERS[member],
        inputs=inputs,
        output=result,
        saved=saved,
    )

    return None


def _fill_member_saved(
    member: str,
    receiver: Any,
    args: list[Value],
    saved: dict[str, Any],
) -> None:
    """
    Fills the op-specific plain-data facts a member rule needs.

    Args:
        member (str): Called member name.
        receiver (Value): Receiver tensor.
        args (list[Value]): Evaluated call arguments.
        saved (dict[str, Any]): Saved-facts dict to extend.
    """

    if member in ("max", "min"):
        saved["argmax"] = _extreme_index(
            cells=receiver.cells, member=member
        )

    elif member == "transpose":
        saved["out_shape"] = list(reversed(_extents(receiver)))

    elif member == "concat":
        saved["axis"] = args[1].get() if len(args) > 1 else 0

    elif member == "permute":
        saved["perm"] = [
            argument.get() for argument in args
        ]
        saved["out_shape"] = [
            _extents(receiver)[axis] for axis in saved["perm"]
        ]

    elif member == "gather":
        saved["indices"] = _gather_indices(args=args)

    return None


def _extreme_index(cells: list[Any], member: str) -> int:
    """
    Returns the index of the extreme cell (first on ties).

    Args:
        cells (list[Any]): Flat payload.
        member (str): 'max' or 'min'.

    Returns:
        int: Extreme index.
    """

    best: int = 0

    for position in range(1, len(cells)):

        if member == "max":
            if cells[position] > cells[best]:
                best = position

        elif cells[position] < cells[best]:
            best = position

    return best


def _gather_indices(args: list[Value]) -> list[int]:
    """
    Extracts the flat integer indices of a gather call.

    Args:
        args (list[Value]): Call arguments (indices tensor or array).

    Returns:
        list[int]: Flat indices.

    Raises:
        TypeError: If the argument shape is unexpected.
    """

    argument: Value = args[0]
    raw: Any = argument.get()

    if isinstance(argument, ValueTensor):
        return [int(cell) for cell in raw]

    if isinstance(raw, list):
        return [int(cell) for cell in raw]

    raise TypeError("gather indices must be a tensor or an array")


def maybe_record_binary(
    op: str,
    left: Value,
    right: Value,
    result: Value,
    tape: Optional[Tape],
) -> None:
    """
    Records a tensor arithmetic operation if it is differentiable.

    Args:
        op (str): Arithmetic operator ('+', '-', '*', '/').
        left (Value): Left operand Value.
        right (Value): Right operand Value.
        result (Value): Produced Value.
        tape (Optional[Tape]): Live tape, or None.
    """

    if tape is None:
        return None

    rule_name: str = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
    }[op]

    shapes: list[list[int]] = []

    for operand in (left, right):

        if isinstance(operand, ValueTensor):
            shapes.append(_extents(operand))

        else:
            shapes.append([])

    tape.maybe_record(
        rule_name=rule_name,
        inputs=[left, right],
        output=result,
        saved={"shapes": shapes},
    )

    return None


def maybe_record_slice(
    target: ValueTensor,
    spec: list[tuple[bool, int, int]],
    indices: list[int],
    result: Value,
    tape: Optional[Tape],
) -> None:
    """
    Records a sliced tensor read if the source is tracked.

    Args:
        target (ValueTensor): Source tensor.
        spec (list[tuple[bool, int, int]]): Per-dimension
            (is_slice, start, stop) of the read.
        indices (list[int]): Fixed integer index per non-sliced
            dimension.
        result (Value): Produced sub-tensor.
        tape (Optional[Tape]): Live tape, or None.
    """

    if tape is None:
        return None

    tape.maybe_record(
        rule_name="slice",
        inputs=[target],
        output=result,
        saved={
            "shapes": [_extents(target)],
            "spec": spec,
            "indices": indices,
        },
    )

    return None


class SetLearnableFunction(ScopeAwareFunction):
    """
    Native set_learnable(elt, value): recursively sets the learnable
    flag (and tracking, see below) on tensors.

    Setting learnable also controls tracking when the value is fresh:
    learnable=True implies tracked=True, learnable=False implies
    tracked=False, so one call prunes or adopts a whole subgraph.
    Ops-produced values keep their own tracked state.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> None:
        """
        Applies the flag recursively.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The element and the boolean flag.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("set_learnable takes exactly two arguments")

        flag: Any = args[1].get()

        if not isinstance(flag, bool):
            raise TypeError("set_learnable flag must be a boolean")

        _apply_flag(element=_flag_target(args[0]), learnable=flag)

        return None


class SetTrackableFunction(ScopeAwareFunction):
    """
    Native set_trackable(elt, value): recursively sets the tracking
    flag on tensors without touching learnability.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> None:
        """
        Applies the flag recursively.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The element and the boolean flag.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("set_trackable takes exactly two arguments")

        flag: Any = args[1].get()

        if not isinstance(flag, bool):
            raise TypeError("set_trackable flag must be a boolean")

        _apply_flag(
            element=_flag_target(args[0]), learnable=None, tracked=flag
        )

        return None


def _flag_target(value: Value) -> Any:
    """
    Unwraps a setter argument to its flaggable payload.

    Args:
        value (Value): The argument Value.

    Returns:
        Any: The wrapped struct instance, tensor, or container.
    """

    if isinstance(value, ValueExternal):
        return value.payload

    return value


def _apply_flag(
    element: Any,
    learnable: Optional[bool],
    tracked: Optional[bool] = None,
) -> None:
    """
    Recursively applies flag updates over structs, arrays, and tensors.

    Args:
        element (Any): Struct instance, array, tensor, or nested
            container.
        learnable (Optional[bool]): New learnable flag, or None to keep.
        tracked (Optional[bool]): New tracked flag, or None to keep.
            When learnable is set explicitly, tracking follows it.
    """

    if learnable is not None:
        tracked = learnable

    if isinstance(element, ValueTensor):
        if learnable is not None:
            element.learnable = learnable

        if tracked is not None:
            element.tracked = tracked

        return None

    if isinstance(element, list):
        for item in element:
            _apply_flag(element=item, learnable=learnable, tracked=tracked)

        return None

    attributes: Optional[dict[str, Any]] = getattr(
        element, "attributes", None
    )

    if isinstance(attributes, dict):

        struct_def: Any = getattr(element, "struct", None)

        if struct_def is not None:

            # Materialize lazily-defaulted attributes so the flags land
            # on the stored tensors (and on future reads of them).
            for name in struct_def.get_all_attributes():
                element.ensure_attribute(name=name)

        for value in attributes.values():
            _apply_flag(element=value, learnable=learnable, tracked=tracked)

    return None


class DetachFunction(ScopeAwareFunction):
    """
    Native detach(x): returns an untracked copy of a tensor.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Value:
        """
        Copies the tensor with tracking and learnability off.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The tensor Value.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Value: Independent untracked copy.

        Raises:
            TypeError: On wrong arity or a non-tensor argument.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("detach takes exactly one argument")

        tensor: Any = args[0]

        if not isinstance(tensor, ValueTensor):
            raise TypeError("detach expects a tensor")

        return ValueTensor(
            value_type=tensor.value_type,
            cells=list(tensor.cells),
            tracked=False,
            learnable=False,
        )


class BackwardFunction(ScopeAwareFunction):
    """
    Native backward(loss): differentiates the tape up to the loss.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> None:
        """
        Runs the backward walk on the live tape.

        Args:
            scope (Any): Live execution scope (provides the tape).
            args (list[Value]): The loss Value.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On wrong arity.
            RuntimeError: If no tape is live.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("backward takes exactly one argument")

        tape: Optional[Any] = scope.tape

        if tape is None:
            raise RuntimeError(
                "backward() requires a live script execution tape"
            )

        tape.backward(loss=args[0])

        return None


def array_to_tensor(*args: Value, **kwargs: Value) -> ValueTensor:
    """
    Native array_to_tensor(arr, learnable, trackable): converts a
    (possibly nested) array into a tensor.

    The shape derives from the nesting; ragged nesting raises
    ValueError. The element type comes from the array's declared type
    or the first cell. Assignment to a tensor attribute validates the
    declared shape and raises ValueError on mismatch.
    """

    if len(args) != 1:
        raise TypeError("array_to_tensor takes exactly one array")

    learnable, tracked = _flag_kwargs(kwargs=kwargs)

    raw: Any = args[0].get()

    if not isinstance(raw, list):
        raise TypeError("array_to_tensor expects an array")

    cells, shape = _tensor_payload(nested=raw)

    element_type: Any = _array_element_type(array=args[0], cells=cells)

    return ValueTensor(
        value_type=TypeTensor(
            value_type=element_type,
            dims=cast("list[int | str]", shape),
        ),
        cells=cells,
        tracked=tracked,
        learnable=learnable,
    )


def tensor_to_array(*args: Value, **kwargs: Value) -> ValueArray:
    """
    Native tensor_to_array(t): converts a tensor into nested
    row-major arrays (rank 1 stays flat, rank 2 becomes an array of
    rows, and so on).
    """

    if len(args) != 1 or not isinstance(args[0], ValueTensor):
        raise TypeError("tensor_to_array takes exactly one tensor")

    tensor: ValueTensor = args[0]
    shape: list[int] = _extents(tensor)

    nested: Any = _nest(
        cells=list(tensor.cells), shape=shape, offset=0
    )

    return ValueArray(
        value_type=_nested_array_type(
            element_type=tensor.value_type.value_type, rank=len(shape)
        ),
        elements=nested if shape != [0] else [],
    )


def _nested_array_type(element_type: Any, rank: int) -> Any:
    """
    Builds the nested TypeArray for a tensor conversion.

    Args:
        element_type (Any): Tensor element type.
        rank (int): Tensor rank.

    Returns:
        Any: TypeArray nested rank times.
    """

    result: Any = element_type

    for _ in range(rank):
        result = TypeArray(value_type=result)

    return result


def _nest(cells: list[Any], shape: list[int], offset: int) -> Any:
    """
    Builds a nested list from a flat row-major payload.

    Args:
        cells (list[Any]): Flat payload.
        shape (list[int]): Remaining shape to nest.
        offset (int): Start offset into the payload.

    Returns:
        Any: Nested list (or a single cell for an empty shape).
    """

    if not shape:
        return cells[offset]

    if len(shape) == 1:
        return list(cells[offset:offset + shape[0]])

    stride: int = _shape_len(shape[1:])
    result: list[Any] = []

    for index in range(shape[0]):
        result.append(
            _nest(
                cells=cells,
                shape=shape[1:],
                offset=offset + index * stride,
            )
        )

    return result


def _flag_kwargs(kwargs: dict[str, Value]) -> tuple[bool, bool]:
    """
    Reads the optional learnable/trackable conversion flags.

    Args:
        kwargs (dict[str, Value]): Keyword arguments.

    Returns:
        tuple[bool, bool]: (learnable, tracked).

    Raises:
        TypeError: On unknown or non-boolean flags.
    """

    learnable: bool = True
    tracked: bool = True

    for key, value in kwargs.items():

        raw: Any = value.get()

        if not isinstance(raw, bool):
            raise TypeError(f"Flag '{key}' must be a boolean")

        if key == "learnable":
            learnable = raw

        elif key == "trackable":
            tracked = raw

        else:
            raise TypeError(f"Unknown flag '{key}'")

    return learnable, tracked


def _tensor_payload(nested: list[Any]) -> tuple[list[Any], list[int]]:
    """
    Flattens a nested array and derives its shape.

    Args:
        nested (list[Any]): Possibly nested array payload.

    Returns:
        tuple[list[Any], list[int]]: Flat cells and derived shape.

    Raises:
        ValueError: On ragged nesting.
    """

    if not nested:
        raise ValueError("array_to_tensor cannot build an empty tensor")

    if not isinstance(nested[0], list):
        return list(nested), [len(nested)]

    rows: int = len(nested)

    for row in nested:

        if not isinstance(row, list) or len(row) != len(nested[0]):
            raise ValueError(
                "array_to_tensor nesting is ragged; all sub-arrays must "
                "match the tensor shape exactly"
            )

    flat: list[Any] = []
    sub_shape: list[int] = []

    for row in nested:

        sub_cells, sub_shape = _tensor_payload(nested=row)
        flat.extend(sub_cells)

    return flat, [rows] + sub_shape


def _array_element_type(array: Value, cells: list[Any]) -> Any:
    """
    Determines the tensor element type of an array argument.

    Args:
        array (Value): The array Value (may carry a declared type).
        cells (list[Any]): Flat cells.

    Returns:
        Any: The element Type.
    """

    declared: Any = getattr(array, "value_type", None)

    if declared is not None and hasattr(declared, "element_type"):
        return declared.element_type

    first: Any = cells[0]

    if isinstance(first, bool):
        from src.core.types.types import TypeBool
        return TypeBool()

    if isinstance(first, int):
        from src.core.types.types import TypeInt
        return TypeInt(num_bits=64, signed=True)

    from src.core.types.types import TypeFloat
    return TypeFloat(num_bits=32)


def _partition_bounds(
    length: int,
    weights: list[float],
) -> list[int]:
    """
    Computes partition boundaries with largest-remainder rounding.

    Args:
        length (int): Total element count.
        weights (list[float]): Positive weights summing to 1 (within
            tolerance).

    Returns:
        list[int]: len(weights) + 1 monotone boundaries starting at 0.

    Raises:
        ValueError: On invalid weights.
    """

    if not weights:
        raise ValueError("partition requires at least one weight")

    for weight in weights:
        if weight <= 0:
            raise ValueError(
                f"Partition weights must be positive, got {weight}"
            )

    total: float = sum(weights)

    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(
            f"Partition weights must sum to 1, got {total}"
        )

    exact: list[float] = [length * weight for weight in weights]
    floors: list[int] = [int(value) for value in exact]
    remaining: int = length - sum(floors)

    order: list[int] = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - floors[index]), index),
    )

    for index in order[:remaining]:
        floors[index] += 1

    bounds: list[int] = [0]

    for size in floors:
        bounds.append(bounds[-1] + size)

    return bounds


def _partition_raw_args(
    args: list[Value],
) -> tuple[list[Any], int, list[float], int, int]:
    """
    Unwraps the shared partition function arguments.

    Args:
        args (list[Value]): Positional arguments.
        kwargs (dict[str, Value]): Keyword arguments (rejected).

    Returns:
        tuple: (data list, seed, weights, partition index).

    Raises:
        TypeError: On wrong arity or kinds.
        ValueError: On invalid values.
    """

    if len(args) != 5:
        raise TypeError(
            "Expected (arr, seed, partition, batch_size, crt_partition)"
        )

    data: Any = args[0].get()
    seed: Any = args[1].get()
    raw_weights: Any = args[2].get()
    batch_size: Any = args[3].get()
    partition: Any = args[4].get()

    if not isinstance(data, list):
        raise TypeError("arr must be an array")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if not isinstance(raw_weights, list) or not raw_weights:
        raise TypeError("partition must be a non-empty weight array")

    weights: list[float] = []

    for weight in raw_weights:

        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("partition weights must be numeric")

        weights.append(float(weight))

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    if isinstance(partition, bool) or not isinstance(partition, int):
        raise TypeError("crt_partition must be an integer")

    return data, seed, weights, batch_size, partition


def get_batch_of_partition(*args: Value, **kwargs: Value) -> ValueArray:
    """
    Native get_batch_of_partition(arr, seed, partition, batch_size,
    crt_partition, iteration): returns one batch of a data partition.

    A single permutation of arr is derived from the seed and split into
    weighted partitions (largest-remainder rounding); iteration counts
    draws for THIS partition. The tail batch may be short; an empty
    array signals exhaustion (reset iteration to 0 for a new epoch -
    the order replays identically).
    """

    if kwargs:
        raise TypeError("get_batch_of_partition takes positional args only")

    if len(args) != 6:
        raise TypeError(
            "Expected (arr, seed, partition, batch_size, crt_partition, "
            "iteration)"
        )

    data, seed, weights, batch_size, partition = _partition_raw_args(
        args=list(args[:5])
    )

    iteration: Any = args[5].get()

    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("iteration must be an integer")

    if iteration < 0:
        raise ValueError(f"iteration must be >= 0, got {iteration}")

    if partition < 0 or partition >= len(weights):
        raise ValueError(
            f"crt_partition {partition} out of range for "
            f"{len(weights)} partition(s)"
        )

    bounds: list[int] = _partition_bounds(
        length=len(data), weights=weights
    )
    start: int = bounds[partition]
    stop: int = bounds[partition + 1]

    order: list[int] = _shuffled_indices(length=len(data), seed=seed)

    position: int = iteration * batch_size

    if position >= stop - start:
        return ValueArray(
            value_type=_batch_element_type(data=data), elements=[]
        )

    batch_indices: list[int] = order[
        start + position:start + position + batch_size
    ]

    return ValueArray(
        value_type=_batch_element_type(data=data),
        elements=[data[index] for index in batch_indices],
    )


def _batch_element_type(data: list[Any]) -> Any:
    """
    Infers the array element type of a partition batch.

    Args:
        data (list[Any]): The dataset elements.

    Returns:
        Any: TypeArray over the inferred element type.
    """

    return TypeArray(
        value_type=_infer_element_type(raw=data[0]) if data else TypeBool()
    )


def _infer_element_type(raw: Any) -> Any:
    """
    Infers a schema element type from a raw Python value.

    Args:
        raw (Any): Sample value.

    Returns:
        Any: The matching element Type.
    """

    if isinstance(raw, bool):
        return TypeBool()

    if isinstance(raw, int):
        return TypeInt(num_bits=64, signed=True)

    if isinstance(raw, float):
        return TypeFloat(num_bits=32)

    if isinstance(raw, list):
        return TypeArray(value_type=_infer_element_type(raw=raw[0]))

    return TypeBool()


def get_partition_iteration_count(
    *args: Value, **kwargs: Value
) -> Value:
    """
    Native get_partition_iteration_count(arr, seed, partition,
    batch_size, crt_partition): returns the total number of batches a
    partition yields (ceil of its size over batch_size).
    """

    if kwargs:
        raise TypeError(
            "get_partition_iteration_count takes positional args only"
        )

    if len(args) != 5:
        raise TypeError(
            "Expected (arr, seed, partition, batch_size, crt_partition)"
        )

    data, seed, weights, batch_size, partition = _partition_raw_args(
        args=list(args)
    )

    if partition < 0 or partition >= len(weights):
        raise ValueError(
            f"crt_partition {partition} out of range for "
            f"{len(weights)} partition(s)"
        )

    bounds: list[int] = _partition_bounds(
        length=len(data), weights=weights
    )
    size: int = bounds[partition + 1] - bounds[partition]

    return ValueInt(
        value_type=TypeInt(num_bits=64, signed=True),
        value=(size + batch_size - 1) // batch_size,
    )


def _shuffled_indices(length: int, seed: int) -> list[int]:
    """
    Returns the seed-fixed permutation of range(length).

    Args:
        length (int): Element count.
        seed (int): Deterministic seed.

    Returns:
        list[int]: The permutation.
    """

    indices: list[int] = list(range(length))
    random.Random(seed).shuffle(indices)

    return indices


class Optimizer:
    """
    Base class for gradient-based optimizers.

    Subclasses implement step(); zero_grad and parameter discovery are
    shared. Hyperparameters are plain mutable attributes, so schedules
    are ordinary assignments in the training loop.

    Attributes:
        state (dict[tuple[str, ...], dict[str, Any]]): Per-locator
            optimizer state (momentum buffers, moments, step counts).
    """

    def __init__(self) -> None:
        """
        Initializes the Optimizer instance.
        """

        self.state: dict[tuple[str, ...], dict[str, Any]] = {}

    def step(self, model: Any) -> None:
        """
        Applies one update to every parameter holding a gradient.

        Args:
            model (Any): Struct instance to update.

        Raises:
            NotImplementedError: Always; subclasses override.
        """

        raise NotImplementedError("Optimizers must implement step().")

    def zero_grad(self, model: Any) -> None:
        """
        Clears every accumulated gradient in the model.

        Args:
            model (Any): Struct instance to clear.
        """

        for _, tensor in _param_walk(instance=self._unwrap(model)):
            tensor.grad = None

        return None

    @staticmethod
    def _unwrap(model: Any) -> Any:
        """
        Unwraps a Value or ValueExternal to the raw instance.

        Args:
            model (Any): Model reference as passed from a script.

        Returns:
            Any: The StructInstance payload.
        """

        if isinstance(model, Value):
            model = model.get()

        return model

    def _parameters(self, model: Any) -> list[
        tuple[tuple[str, ...], ValueTensor]
    ]:
        """
        Discovers the Float tensor parameters of a model.

        Args:
            model (Any): Model reference as passed from a script.

        Returns:
            list[tuple[tuple[str, ...], ValueTensor]]: Locator/tensor
                pairs.
        """

        return list(_param_walk(instance=self._unwrap(model=model)))


class SGD(Optimizer):
    """
    Stochastic gradient descent with optional momentum.

    Attributes:
        lr (float): Learning rate.
        momentum (float): Momentum factor (0 disables).
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        """
        Initializes the SGD instance.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (float, optional): Momentum factor. Defaults to 0.0.
        """

        # Initialize parent class
        super().__init__()

        self.lr: float = lr
        self.momentum: float = momentum

    def step(self, model: Any) -> None:
        """
        Applies one SGD update and consumes the deposited gradients.

        Args:
            model (Any): Struct instance to update.
        """

        for locator, tensor in self._parameters(model=model):

            if tensor.grad is None:
                continue

            delta: list[float] = list(tensor.grad)

            if self.momentum:

                state: dict[str, Any] = self.state.setdefault(
                    locator, {"velocity": [0.0] * len(tensor.cells)}
                )
                velocity: list[Any] = state["velocity"]

                velocity[:] = [
                    self.momentum * old + new
                    for old, new in zip(velocity, delta)
                ]
                delta = list(velocity)

            tensor.cells = [
                cell - self.lr * update
                for cell, update in zip(tensor.cells, delta)
            ]
            tensor.grad = None

        return None


def _param_walk(instance: Any) -> Iterator[tuple[tuple[str, ...], ValueTensor]]:
    """
    Yields (locator, tensor) for every Float tensor attribute.

    Args:
        instance (Any): Struct instance to walk.

    Yields:
        tuple[tuple[str, ...], ValueTensor]: Attribute path and tensor.
    """

    attributes: Optional[dict[str, Any]] = getattr(
        instance, "attributes", None
    )

    if not isinstance(attributes, dict):
        return

    for name in list(attributes.keys()):

        if hasattr(instance, "ensure_attribute"):
            instance.ensure_attribute(name=name)

        value: Any = attributes.get(name)

        if isinstance(value, ValueTensor) and isinstance(
            value.value_type.value_type, TypeFloat
        ):
            yield (name,), value

        elif hasattr(value, "attributes"):
            for sub_locator, tensor in _param_walk(instance=value):
                yield (name,) + sub_locator, tensor

        elif isinstance(value, ValueArray):
            for index, item in enumerate(value.get()):
                if isinstance(item, ValueTensor) and isinstance(
                    item.value_type.value_type, TypeFloat
                ):
                    yield ((name, str(index)), item)

    return
