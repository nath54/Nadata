"""
Golden output tests: run every bundled example project through the CLI and
assert its printed output.

Each example's prints double as its specification; when an example's output
changes on purpose, update the EXPECTED_OUTPUTS entry in the same commit.
"""

# Import Modules
import pytest

from src.cli import main

# Project folder -> substrings that must appear in the run output, in order
EXPECTED_OUTPUTS: dict[str, list[str]] = {
    "ex0_simple": [
        "Age difference between Tom and Jean: DateTime(y=-31, m=0, d=0)",
        "Is Tom an adult: True",
        "Is Jean an adult: True",
        "Tom id: 0",
        "Adult Parisians count: 3",
        "Adult Parisians: [<Human id=0>, <Human id=1>, <Human id=2>]",
        "Loaded 6 struct(s), 6 instance(s), 2 function(s).",
    ],
    "ex1_primitives": [
        "Sensor 1 is saturated: True",
        "Sensor 2 is saturated: False",
        "Sensor 1 adjusted temp: -34.5",
        "Hot active sensors count: 1",
        "Hot active sensors: [<NumericSensors id=2>]",
        "Loaded 1 struct(s), 3 instance(s), 0 function(s).",
    ],
    "ex2_inheritance": [
        "Primary DB Description: srv-db-01",
        "Is DB Overloaded at 92%: True",
        "Total polymorphic nodes count: 3",
        "Primary database servers: [<DatabaseServer id=0>]",
        "Fast heavy DBs count: 2",
        "Loaded 6 struct(s), 3 instance(s), 0 function(s).",
    ],
    "ex3_collections": [
        "Dark theme has opaque primary: True",
        "Hackathon duration: DateTime(y=0, m=0, d=2, H=9, M=0, S=0, MS=0)",
        "Dark palettes found: 1",
        "Future events: [<EventSchedule id=0>]",
        "Loaded 3 struct(s), 4 instance(s), 0 function(s).",
    ],
    "ex4_expressions": [
        "Input 10.0 transformed by Standard Scaler: 34.0",
        "Input 10.0 transformed by Bypass Scaler: 10.0",
        "Classify -5.0: NEGATIVE",
        "Classify 0.0: ZERO",
        "Classify 45.0: NORMAL",
        "Classify 150.0: HIGH",
        "Boolean logic test (true, false, false): True",
        "Boolean logic test (false, true, false): False",
        "Boolean logic test (false, false, true): True",
        "Heavy pipelines: [<MathPipeline id=0>]",
        "Loaded 1 struct(s), 2 instance(s), 0 function(s).",
    ],
    "ex5_edge_cases": [
        "Item 1 noop call: True",
        "Item 1 coordinate sum with shadow parameters: 15",
        "Zero items count: 2",
        "Loaded 3 struct(s), 6 instance(s), 0 function(s).",
    ],
    "ex6_loops_collections": [
        "boiler total: 104.5",
        "boiler average: 20.9",
        "boiler max: 22.3",
        "boiler above 21: 1",
        "boiler gain: 1.029",
        "reactor total: 906.0",
        "first reading: 20.5",
        "last reading: 20.9",
        "drift factor: 0.98",
        "sum 1..100: 4950",
        "fib pair: 10946 17711",
        "evens found: 5",
        "table sum: 36",
        "stock items: bolts nuts washers",
        "stock units: 400",
        "adjusted total: 102.41",
        "guarded runs: 40",
        "Loaded 1 struct(s), 2 instance(s), 0 function(s).",
    ],
    "ex7_queues_sets": [
        "backup jobs: 3",
        "first job: index",
        "peek job: compact",
        "jobs left: 2",
        "main next: analyze",
        "main jobs after drain: 0",
        "idle jobs (fresh instance): 0",
        "incidents queued: 3",
        "most urgent: fire",
        "peek after fire: cleanup",
        "second incident: cleanup",
        "third incident: audit",
        "incidents left: 0",
        "lightest load: nightly",
        "heaviest load: hourly",
        "label count: 2",
        "has red: True",
        "has green: False",
        "has red after discard: False",
        "backup label count: 0",
        "backup labels: 2",
        "has cache: True",
        "jobs iterated: 2",
        "first worker: alice",
        "second worker: bob",
        "workers left: 0",
        "Loaded 2 struct(s), 5 instance(s), 0 function(s).",
    ],
    "ex8_grids_tensors": [
        "terrain cells: 3",
        "at -3 7: lava",
        "has 5 5: False",
        "square neighbor count: 4",
        "with temporary: 4",
        "after remove: 3",
        "cells visited: 3",
        "hex neighbor count: 6",
        "hex occupied next to (1,1): 1",
        "cube neighbor count: 6",
        "cube occupied next to core: 1",
        "kernel size: 9",
        "kernel at 0 2: -1.0",
        "kernel at 2 2: 2.0",
        "kernel rank: 2",
        "weights size: 8",
        "weights at 1 1 1: 8.0",
        "kernel after write: 9.5",
        "kernel sum: 11.75",
        "Loaded 1 struct(s), 1 instance(s), 0 function(s).",
    ],
    "ex9_constants": [
        "small batch: 2",
        "small dim: 4",
        "wide batch: 3",
        "wide dim: 2",
        "small q cells: 8",
        "wide q cells: 6",
        "q rank: 2",
        "small q sum: 36.0",
        "wide q sum: 6.0",
        "small q sum after write: 45.0",
        "small batch again: 2",
        "small label: renamed",
        "Loaded 1 struct(s), 2 instance(s), 0 function(s).",
    ],
}

# Every bundled example must be covered here
ALL_EXAMPLES: list[str] = [
    "ex0_simple",
    "ex1_primitives",
    "ex2_inheritance",
    "ex3_collections",
    "ex4_expressions",
    "ex5_edge_cases",
    "ex6_loops_collections",
    "ex7_queues_sets",
    "ex8_grids_tensors",
    "ex9_constants",
]


class TestExampleOutputs:
    """
    Runs each example through the CLI and asserts its golden output.
    """

    def test_all_examples_are_covered(self) -> None:
        """
        Tests that the expected outputs table covers every example folder.
        """

        import os

        examples_dir: str = os.path.join("examples")

        folders: list[str] = sorted(
            name
            for name in os.listdir(examples_dir)
            if os.path.isdir(os.path.join(examples_dir, name))
            and name.startswith("ex")
        )

        assert sorted(EXPECTED_OUTPUTS.keys()) == folders

    @pytest.mark.parametrize("example", ALL_EXAMPLES)
    def test_example_output(self, example: str, capsys: pytest.CaptureFixture) -> None:
        """
        Tests one example's printed output against its golden expectations.

        Args:
            example (str): Example folder name under examples/.
            capsys (pytest.CaptureFixture): Output capture fixture.
        """

        exit_code: int = main(argv=["run", f"examples/{example}"])

        captured: str = capsys.readouterr().out

        assert exit_code == 0

        # Every expected line must appear, in declaration order
        position: int = 0

        for expected in EXPECTED_OUTPUTS[example]:
            found: int = captured.find(expected, position)

            assert found != -1, f"'{expected}' missing from output of {example}"

            position = found + len(expected)

    def test_examples_run_via_console_script(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        Tests that a dump / reload of an example preserves its script output.
        """

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path: str = os.path.join(temp_dir, "golden.ndb")

            exit_code: int = main(argv=["dump", "examples/ex6_loops_collections", storage_path])

            assert exit_code == 0

            # Reload and re-execute a representative script line
            from src.api import Nadata

            reloaded: Nadata = Nadata.open(storage_path=storage_path)

            reloaded.execute_data('print("reload total:", boiler.total());')

            assert "reload total: 104.5" in capsys.readouterr().out
