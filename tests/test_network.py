"""
Tests for Phase E networks and routing: network_index over user
structs, endpoint snapping, mode filtering, distance / time costs,
path members, and incremental maintenance.
"""

# Import Modules
import math
from typing import Any

import pytest

from nadata.api import Nadata
from nadata.core.database.database import NaDataDatabase
from nadata.core.types.home_types.coordinate import Coordinate
from nadata.core.types.home_types.geometry import Polyline
from nadata.core.types.home_types.planet import DEFAULT_PLANET, set_current_planet

R: float = 6371008.8

SCHEMA: str = """
class Node {
    name: Str;
    position: Coordinate;
}
class Edge {
    label: Str;
    from_node: Node;
    to_node: Node;
    modes: Array[Str] = [];
    speed: Dict[Str, Float64] = {};
    shape: Polyline;
}
"""



def _ring_app() -> Any:
    """
    Builds a square network: a - b - c - d - a, all edges drivable
    and walkable except the diagonal bd (car only) and ad (walking
    only). bd is the hypotenuse.

    Returns:
        Any: The Nadata facade with the network built.
    """

    app: Any = Nadata(database=NaDataDatabase())

    app.execute_schema(schema_text=SCHEMA)

    app.execute_data(
        "a = Node(name=\"a\", position=Coordinate(0, 0, 6371008.8));"
        "b = Node(name=\"b\", position=Coordinate(0, 1, 6371008.8));"
        "c = Node(name=\"c\", position=Coordinate(1, 1, 6371008.8));"
        "d = Node(name=\"d\", position=Coordinate(1, 0, 6371008.8));"
        "e_ab = Edge(label=\"ab\", from_node=a, to_node=b,"
        " modes=[\"car\", \"walk\"],"
        " speed={ \"car\": 50.0, \"walk\": 5.0 },"
        " shape=Polyline([Coordinate(0, 0, 6371008.8),"
        " Coordinate(0, 1, 6371008.8)]));"
        "e_bc = Edge(label=\"bc\", from_node=b, to_node=c,"
        " modes=[\"car\", \"walk\"],"
        " speed={ \"car\": 50.0, \"walk\": 5.0 },"
        " shape=Polyline([Coordinate(0, 1, 6371008.8),"
        " Coordinate(1, 1, 6371008.8)]));"
        "e_cd = Edge(label=\"cd\", from_node=c, to_node=d,"
        " modes=[\"car\", \"walk\"],"
        " speed={ \"car\": 50.0, \"walk\": 5.0 },"
        " shape=Polyline([Coordinate(1, 1, 6371008.8),"
        " Coordinate(1, 0, 6371008.8)]));"
        "e_bd = Edge(label=\"bd\", from_node=b, to_node=d,"
        " modes=[\"car\"], speed={ \"car\": 90.0 },"
        " shape=Polyline([Coordinate(0, 1, 6371008.8),"
        " Coordinate(1, 0, 6371008.8)]));"
        "e_ad = Edge(label=\"ad\", from_node=a, to_node=d,"
        " modes=[\"walk\"], speed={ \"walk\": 5.0 },"
        " shape=Polyline([Coordinate(0, 0, 6371008.8),"
        " Coordinate(1, 0, 6371008.8)]));"
        "network_index(Node, Edge);"
    )

    return app



def _node_names(path: Any) -> list[str]:
    """
    Extracts the node names along a path.

    Args:
        path (Any): The Path result.

    Returns:
        list[str]: Node names in traversal order.
    """

    return [node.attributes["name"].get() for node in path.nodes]



class TestRouting:
    """
    route() over the square network.
    """

    def test_mode_filtering(self) -> None:
        """
        Car routes may use the car-only hypotenuse; walk routes
        cannot.
        """

        app: Any = _ring_app()

        node_table: Any = app.database.tables_by_struct_name["Node"]

        network: Any = node_table.networks[0]

        start: Coordinate = Coordinate(0.0, 0.0, R)
        end: Coordinate = Coordinate(1.0, 0.0, R)

        car_path: Any = network.route(start=start, end=end, mode="car")
        walk_path: Any = network.route(start=start, end=end, mode="walk")

        # Car: a -> b -> (hypotenuse) -> d
        assert _node_names(car_path) == ["a", "b", "d"]

        # Walk: a -> d directly (the walking edge)
        assert _node_names(walk_path) == ["a", "d"]

        # One degree of latitude is about 111.195 km
        assert walk_path.cost == pytest.approx(111195.0, rel=0.001)

    def test_distance_cost_uses_shape(self) -> None:
        """
        The distance cost follows the edge Polyline shape length.
        """

        app: Any = _ring_app()

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        path: Any = network.route(
            start=Coordinate(0.0, 0.0, R),
            end=Coordinate(1.0, 0.0, R),
            mode="walk",
        )

        expected: float = Polyline(
            vertices=[
                Coordinate(0, 0, R),
                Coordinate(1, 0, R),
            ]
        ).length()

        assert path.cost == pytest.approx(expected, rel=1e-6)

    def test_time_cost(self) -> None:
        """
        The time cost divides the length by the mode speed.
        """

        app: Any = _ring_app()

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        path: Any = network.route(
            start=Coordinate(0.0, 0.0, R),
            end=Coordinate(1.0, 0.0, R),
            mode="walk",
            cost="time",
        )

        length: float = Polyline(
            vertices=[Coordinate(0, 0, R), Coordinate(1, 0, R)]
        ).length()

        assert path.cost == pytest.approx(length / (5.0 / 3.6), rel=1e-6)

    def test_snapping(self) -> None:
        """
        Endpoints snap to the nearest nodes; the path starts and ends
        there.
        """

        app: Any = _ring_app()

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        path: Any = network.route(
            start=Coordinate(0.01, 0.01, R),
            end=Coordinate(0.99, 0.01, R),
            mode="walk",
        )

        names: list[str] = _node_names(path)

        assert names[0] == "a"
        assert names[-1] == "d"

    def test_no_path(self) -> None:
        """
        A mode with no connecting edges raises ValueError.
        """

        app: Any = _ring_app()

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        # "train" has no edges at all
        with pytest.raises(ValueError):
            network.route(
                start=Coordinate(0.0, 0.0, R),
                end=Coordinate(1.0, 0.0, R),
                mode="train",
            )

    def test_same_point_route(self) -> None:
        """
        Snapping both endpoints to the same node yields an empty
        path.
        """

        app: Any = _ring_app()

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        path: Any = network.route(
            start=Coordinate(0.0, 0.0, R),
            end=Coordinate(0.001, 0.001, R),
            mode="car",
        )

        assert path.nodes == [network.nodes_table.get(local_id=0)]
        assert path.edges == []
        assert path.cost == 0.0

    def test_unknown_cost_mode(self) -> None:
        """
        Unknown cost modes raise ValueError.
        """

        app: Any = _ring_app()

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        with pytest.raises(ValueError):
            network.route(
                start=Coordinate(0, 0, R),
                end=Coordinate(1, 0, R),
                mode="car",
                cost="happiness",
            )


class TestLanguageIntegration:
    """
    network_index and route from .nadata scripts.
    """

    def test_script_route(self) -> None:
        """
        The script-level route returns a path with duck-typed members.
        """

        app: Any = _ring_app()

        app.execute_data(
            "p = route(Coordinate(0.05, 0.05, 6371008.8),"
            " Coordinate(0.95, 0.05, 6371008.8), \"car\");"
            "print(\"first:\", p.nodes[0].name);"
            "print(\"second:\", p.nodes[1].name);"
            "count = 0;"
            "for (edge: Edge in p.edges) { count = count + 1; }"
            "print(\"hops:\", count);"
        )

        assert app.get_attribute(symbol="a", name="name") == "a"

    def test_route_requires_network(self) -> None:
        """
        route without a built network raises ValueError.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        with pytest.raises(ValueError):
            app.execute_data(
                "p = route(Coordinate(0, 0, 6371008.8),"
                " Coordinate(1, 0, 6371008.8), \"car\");"
            )

    def test_network_index_rejects_non_structs(self) -> None:
        """
        network_index rejects non-struct arguments.
        """

        from nadata.core.expressions.builtins import NetworkIndexFunction
        from nadata.core.values.values import ValueStr, make_value

        with pytest.raises(TypeError):
            NetworkIndexFunction().invoke(
                scope=None,
                args=[
                    make_value(
                        value_type=__import__(
                            "nadata.core.types.types", fromlist=["TypeStr"]
                        ).TypeStr(),
                        raw="nope",
                    ),
                    make_value(
                        value_type=__import__(
                            "nadata.core.types.types", fromlist=["TypeStr"]
                        ).TypeStr(),
                        raw="nope",
                    ),
                ],
                kwargs={},
            )


class TestMaintenance:
    """
    Incremental network updates.
    """

    def test_edge_insert_after_build(self) -> None:
        """
        Edges inserted after the build participate in routing.
        """

        app: Any = _ring_app()

        # A walk edge b -> d that beats the a -> d walk edge for the
        # a...d trip via a -> b -> d
        app.execute_data(
            "e_bd_walk = Edge(label=\"bdw\", from_node=b, to_node=d,"
            " modes=[\"walk\"], speed={ \"walk\": 5.0 },"
            " shape=Polyline([Coordinate(0, 1, 6371008.8),"
            " Coordinate(1, 0, 6371008.8)]));"
        )

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        path: Any = network.route(
            start=Coordinate(0.0, 0.0, R),
            end=Coordinate(1.0, 0.0, R),
            mode="walk",
        )

        # The hypotenuse (~156 km) is longer than the direct edge
        # (~111 km), so the direct edge still wins
        assert _node_names(path) == ["a", "d"]

    def test_node_removal(self) -> None:
        """
        Removing a node reroutes around it.
        """

        app: Any = _ring_app()

        table: Any = app.database.tables_by_struct_name["Node"]

        network: Any = table.networks[0]

        # Remove d: a car route toward its position now snaps to the
        # nearest survivor (c) and detours through b -> c
        table.index_remove_entity(local_id=3)

        path: Any = network.route(
            start=Coordinate(0.0, 0.0, R),
            end=Coordinate(0.999, 0.001, R),
            mode="car",
        )

        names: list[str] = _node_names(path)

        assert "d" not in names
        assert names[-1] == "c"

    def test_edge_removal(self) -> None:
        """
        Removing an edge forces the longer alternative.
        """

        app: Any = _ring_app()

        edge_table: Any = app.database.tables_by_struct_name["Edge"]

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        # Drop the car-only hypotenuse (local id 3)
        edge_table.index_remove_entity(local_id=3)

        path: Any = network.route(
            start=Coordinate(0.0, 0.0, R),
            end=Coordinate(1.0, 0.0, R),
            mode="car",
        )

        # Car now must go around: a -> b -> c -> d
        assert _node_names(path) == ["a", "b", "c", "d"]

    def test_edge_update_recosts(self) -> None:
        """
        Rewriting an edge's shape invalidates its cached cost.
        """

        app: Any = _ring_app()

        network: Any = app.database.tables_by_struct_name["Node"].networks[0]

        edge_table: Any = app.database.tables_by_struct_name["Edge"]

        edge: Any = edge_table.get(local_id=0)

        from nadata.core.values.values import ValuePolyline

        # Detour the a - b edge far east: the a -> b -> ... -> d car
        # route becomes much longer than a -> d is for walking, but
        # the car path (forced through b) length changes
        edge.set_attribute(
            name="shape",
            value=ValuePolyline(
                value=Polyline(
                    vertices=[
                        Coordinate(0, 0, R),
                        Coordinate(0, 30, R),
                        Coordinate(0, 1, R),
                    ]
                )
            ),
        )

        path: Any = network.route(
            start=Coordinate(0.0, 0.0, R),
            end=Coordinate(1.0, 0.0, R),
            mode="car",
        )

        # The detoured a - b edge (~3345 km) exceeds the direct walk
        # distance, but the car cannot walk; it must still route
        # through b and pay the detour
        assert _node_names(path) == ["a", "b", "d"]

        assert path.cost > 3_000_000.0


class TestUnfilteredEdges:
    """
    Edge structs without modes / speeds.
    """

    def test_no_modes_attribute(self) -> None:
        """
        Edges without a modes attribute carry every mode.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class N { name: Str; position: Coordinate; }\n"
            "class E { from_node: N; to_node: N; shape: Polyline; }\n"
        )

        app.execute_data(
            "a = N(name=\"a\", position=Coordinate(0, 0, 6371008.8));"
            "b = N(name=\"b\", position=Coordinate(0, 1, 6371008.8));"
            "e = E(from_node=a, to_node=b,"
            " shape=Polyline([Coordinate(0, 0, 6371008.8),"
            " Coordinate(0, 1, 6371008.8)]));"
            "network_index(N, E);"
        )

        network: Any = app.database.tables_by_struct_name["N"].networks[0]

        for mode in ("car", "walk", "train", "anything"):
            path: Any = network.route(
                start=Coordinate(0, 0, R),
                end=Coordinate(0, 1, R),
                mode=mode,
            )

            assert len(path.edges) == 1

    def test_missing_speed_falls_back(self) -> None:
        """
        A time route on an edge without the mode's speed falls back
        to the edge's fastest speed.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class N { name: Str; position: Coordinate; }\n"
            "class E { from_node: N; to_node: N; modes: Array[Str] = [];"
            " speed: Dict[Str, Float64] = {}; shape: Polyline; }\n"
        )

        app.execute_data(
            "a = N(name=\"a\", position=Coordinate(0, 0, 6371008.8));"
            "b = N(name=\"b\", position=Coordinate(0, 1, 6371008.8));"
            "e = E(from_node=a, to_node=b, modes=[\"car\"],"
            " speed={ \"car\": 36.0 },"
            " shape=Polyline([Coordinate(0, 0, 6371008.8),"
            " Coordinate(0, 1, 6371008.8)]));"
            "network_index(N, E);"
        )

        network: Any = app.database.tables_by_struct_name["N"].networks[0]

        path: Any = network.route(
            start=Coordinate(0, 0, R),
            end=Coordinate(0, 1, R),
            mode="car",
            cost="time",
        )

        # 111195 m at 36 km/h = 10 m/s -> 11119.5 s
        assert path.cost == pytest.approx(111195.0 / 10.0, rel=1e-3)


class TestTeardown:
    """
    Session state hygiene.
    """

    def teardown_method(self) -> None:
        """
        Restores the default session planet.

        Returns:
            None: Nothing.
        """

        set_current_planet(planet=DEFAULT_PLANET)
