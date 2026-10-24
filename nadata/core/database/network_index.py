"""
Network index and routing over user-defined node / edge structs.

A network binds a node struct (carrying a Coordinate position) and an
edge struct (carrying from/to references into the node struct, and
optionally modes, per-mode speeds, and a Polyline shape). Routing runs
A* over mode-filtered edges with geodesic costs; indoor graphs (rooms
as nodes, doors as edges) use the same machinery.

Like spatial indexes, networks are session scoped: they are rebuilt by
scripts after load, and maintained incrementally on insert / removal /
write of the involved tables.
"""

# Import Modules
import heapq
import math
from typing import Any, Optional

from nadata.core.database.entity_id import EntityId
from nadata.core.types.home_types.coordinate import Coordinate
from nadata.core.types.home_types.geometry import Polyline
from nadata.core.types.home_types.planet import get_current_planet

# km/h to m/s
_KMH_TO_MPS: float = 1.0 / 3.6

# Fallback speed for edges (and networks) without any speed data;
# keeps time-cost routing connected on unannotated graphs
_DEFAULT_SPEED_MPS: float = 50.0 * _KMH_TO_MPS



class Path:
    """
    Routing result: the traversed nodes and edges with the total cost.

    Attributes:
        nodes (list[Any]): The traversed node instances, from the
            snapped start to the snapped end.
        edges (list[Any]): The traversed edge instances, in order.
        cost (float): Total cost (meters for the distance mode,
            seconds for the time mode).
        mode (str): The routing mode the path was computed for.
    """

    def __init__(
        self,
        nodes: list[Any],
        edges: list[Any],
        cost: float,
        mode: str,
    ) -> None:
        """
        Initializes the Path instance.

        Args:
            nodes (list[Any]): Traversed node instances.
            edges (list[Any]): Traversed edge instances.
            cost (float): Total cost.
            mode (str): Routing mode.
        """

        self.nodes: list[Any] = nodes
        self.edges: list[Any] = edges
        self.cost: float = cost
        self.mode: str = mode

    def __repr__(self) -> str:
        """
        Returns string representation of the Path.

        Returns:
            str: Representation with the mode, hop count, and cost.
        """

        return (
            f"Path(mode={self.mode!r}, hops={len(self.edges)}, "
            f"cost={self.cost:.2f})"
        )



class NetworkIndex:
    """
    Session-scoped routing network over a node struct and an edge
    struct.

    Attributes:
        nodes_table (Any): The StructTable of node instances.
        edges_table (Any): The StructTable of edge instances.
        position_attribute (str): The node Coordinate attribute name.
        from_attribute (str): The edge attribute referencing the start
            node.
        to_attribute (str): The edge attribute referencing the end
            node.
        modes_attribute (Optional[str]): The edge Array[Str] modes
            attribute, or None when edges are unfiltered.
        speed_attribute (Optional[str]): The edge per-mode speed dict
            attribute (km/h), or None.
        shape_attribute (Optional[str]): The edge Polyline geometry
            attribute, or None.
        adjacency (dict[int, list[tuple[int, int]]]): Node local id to
            (edge local id, other node local id) pairs.
        node_ids (set[int]): Known node local ids.
    """

    def __init__(
        self,
        nodes_table: Any,
        edges_table: Any,
    ) -> None:
        """
        Initializes the NetworkIndex instance and detects attributes.

        Args:
            nodes_table (Any): The node StructTable.
            edges_table (Any): The edge StructTable.

        Raises:
            TypeError: If the structs lack the required attributes.
        """

        self.nodes_table: Any = nodes_table
        self.edges_table: Any = edges_table

        node_struct: Any = nodes_table.struct
        edge_struct: Any = edges_table.struct

        # The node position: the first Coordinate attribute
        self.position_attribute: Optional[str] = None

        for name, attr_def in dict(node_struct.get_all_attributes()).items():
            if repr(attr_def.attr_type) == "Coordinate":
                self.position_attribute = name

                break

        if self.position_attribute is None:
            raise TypeError(
                f"Struct '{node_struct.name}' has no Coordinate attribute "
                f"for network node positions"
            )

        # The edge endpoints: Struct-typed attributes matching the node
        # struct
        from_attr: Optional[str] = None
        to_attr: Optional[str] = None

        for name, attr_def in dict(edge_struct.get_all_attributes()).items():
            attr_type: Any = attr_def.attr_type

            is_node_ref: bool = (
                getattr(attr_type, "name", None) == node_struct.name
                and hasattr(attr_type, "get_all_attributes")
            )

            if is_node_ref:
                if from_attr is None:
                    from_attr = name
                else:
                    to_attr = name

        if from_attr is None or to_attr is None:
            raise TypeError(
                f"Struct '{edge_struct.name}' needs two attributes "
                f"referencing '{node_struct.name}' (from / to)"
            )

        self.from_attribute: str = from_attr
        self.to_attribute: str = to_attr

        # Optional overlays
        self.modes_attribute: Optional[str] = (
            "modes" if edge_struct.get_attribute(name="modes") is not None else None
        )
        self.speed_attribute: Optional[str] = (
            "speed" if edge_struct.get_attribute(name="speed") is not None else None
        )
        self.shape_attribute: Optional[str] = (
            "shape" if edge_struct.get_attribute(name="shape") is not None else None
        )

        self.adjacency: dict[int, list[tuple[int, int]]] = {}
        self.node_ids: set[int] = set()

        # Edge local ids whose endpoint references were not yet
        # populated at insert time (instance shells fill in later)
        self._pending_edges: set[int] = set()

        # Cost cache per edge local id: (cost mode, transport mode)
        # -> seconds or meters
        self._edge_costs: dict[int, dict[tuple[str, str], float]] = {}

        # Mode -> node local id -> connected component id (lazily
        # built for mode-aware snapping, cleared on every mutation)
        self._component_cache: dict[str, dict[int, int]] = {}

        # Fastest speed on the network (m/s) for the time heuristic
        # and for unannotated edges; computed at build time (an
        # underestimated max makes the heuristic inadmissible and
        # strands A*)
        self._max_speed_mps: float = 0.0

        # A cell index over node positions for endpoint snapping
        self._node_index: Any = None

    def _ensure_node_index(self) -> Any:
        """
        Returns the cell index over node positions, creating it once.

        Returns:
            Any: The SpatialIndex over the position attribute.
        """

        if self._node_index is None:
            existing: Optional[Any] = self.nodes_table.spatial_indexes.get(
                self.position_attribute
            )

            if existing is not None:
                self._node_index = existing
            else:
                self._node_index = self.nodes_table.create_spatial_index(
                    attribute=self.position_attribute, lod=10
                )

        return self._node_index

    def build(self) -> None:
        """
        Indexes all existing nodes and edges of both tables.

        Returns:
            None: Nothing.
        """

        self._ensure_node_index()

        self._component_cache.clear()

        for instance in self.nodes_table.instances.values():
            self._add_node(local_id=instance.entity_id.local_id)

        for instance in self.edges_table.instances.values():
            self._add_edge(instance=instance)
            self._absorb_speed(instance=instance)

        self._resolve_pending_edges()

    # --- Maintenance hooks ---

    def on_insert(self, instance: Any) -> None:
        """
        Adds a newly inserted node or edge instance.

        Args:
            instance (Any): The inserted StructInstance.
        """

        self._component_cache.clear()

        if instance.struct is self.nodes_table.struct:
            self._add_node(local_id=instance.entity_id.local_id)

            return None

        if instance.struct is self.edges_table.struct:
            self._add_edge(instance=instance)
            self._absorb_speed(instance=instance)

    def on_remove(self, local_id: int, table: Any) -> None:
        """
        Drops a local id from the network.

        Node and edge local ids live in separate per-table spaces, so
        the calling table disambiguates the removal.

        Args:
            local_id (int): The removed local id.
            table (Any): The StructTable the removal happened on.
        """

        self._component_cache.clear()

        if table is self.nodes_table:
            self._remove_node(local_id=local_id)

            return None

        self._remove_edge(local_id=local_id)

    def on_update(self, instance: Any) -> None:
        """
        Invalidates cached costs after an edge attribute write.

        Args:
            instance (Any): The written StructInstance.
        """

        if instance.struct is not self.edges_table.struct:
            return None

        self._component_cache.clear()

        local_id: int = instance.entity_id.local_id

        self._edge_costs.pop(local_id, None)

        self._remove_edge(local_id=local_id)
        self._add_edge(instance=instance)

    def _add_node(self, local_id: int) -> None:
        """
        Registers one node local id.

        Args:
            local_id (int): The node local id.
        """

        self.node_ids.add(local_id)

        self.adjacency.setdefault(local_id, [])

    def _remove_node(self, local_id: int) -> None:
        """
        Drops one node and every incident edge.

        Args:
            local_id (int): The node local id.
        """

        self.node_ids.discard(local_id)

        self.adjacency.pop(local_id, None)

        # Drop dangling edge entries pointing at the removed node
        for node_id, links in self.adjacency.items():
            self.adjacency[node_id] = [
                (edge_id, other)
                for edge_id, other in links
                if other != local_id
            ]

    def _remove_edge(self, local_id: int) -> None:
        """
        Drops one edge from the adjacency and the cost cache.

        Args:
            local_id (int): The edge local id.
        """

        self._edge_costs.pop(local_id, None)
        self._pending_edges.discard(local_id)

        for node_id in list(self.adjacency.keys()):
            self.adjacency[node_id] = [
                (edge_id, other)
                for edge_id, other in self.adjacency[node_id]
                if edge_id != local_id
            ]

    def _resolve_pending_edges(self) -> None:
        """
        Retries edges whose endpoints were missing when they arrived.

        Returns:
            None: Nothing.
        """

        for local_id in list(self._pending_edges):
            instance: Any = self.edges_table.instances.get(local_id)

            if instance is None:
                self._pending_edges.discard(local_id)

                continue

            self._pending_edges.discard(local_id)

            self._add_edge(instance=instance)

    def _add_edge(self, instance: Any) -> None:
        """
        Registers one edge instance in the adjacency.

        Edges whose endpoint references are not yet populated are
        parked as pending and retried before the next route.

        Args:
            instance (Any): The edge StructInstance.
        """

        local_id: int = instance.entity_id.local_id

        try:
            from_node: int = self._endpoint_local_id(
                instance=instance, attribute=self.from_attribute
            )
            to_node: int = self._endpoint_local_id(
                instance=instance, attribute=self.to_attribute
            )
        except TypeError:
            self._pending_edges.add(local_id)

            return None

        if from_node not in self.node_ids:
            self._add_node(local_id=from_node)

        if to_node not in self.node_ids:
            self._add_node(local_id=to_node)

        self.adjacency.setdefault(from_node, []).append(
            (instance.entity_id.local_id, to_node)
        )
        self.adjacency.setdefault(to_node, []).append(
            (instance.entity_id.local_id, from_node)
        )

    def _endpoint_local_id(self, instance: Any, attribute: str) -> int:
        """
        Resolves an endpoint reference to a node local id.

        Args:
            instance (Any): The edge StructInstance.
            attribute (str): The referencing attribute name.

        Returns:
            int: The node local id.

        Raises:
            TypeError: If the attribute does not reference an instance.
        """

        value: Any = instance.attributes.get(attribute)

        if value is None:
            raise TypeError(
                f"Edge instance is missing its '{attribute}' endpoint"
            )

        raw: Any = value.get() if hasattr(value, "get") else value

        local_id: int = raw.entity_id.local_id

        return local_id

    # --- Costs ---

    def _edge_instance(self, local_id: int) -> Any:
        """
        Returns an edge instance by local id.

        Args:
            local_id (int): The edge local id.

        Returns:
            Any: The StructInstance.
        """

        return self.edges_table.get(local_id=local_id)

    def _edge_length(self, edge: Any) -> float:
        """
        Returns the geodesic length of an edge in meters.

        Uses the shape Polyline when present; otherwise the direct
        geodesic distance between the endpoint positions.

        Args:
            edge (Any): The edge StructInstance.

        Returns:
            float: Length in meters.
        """

        if self.shape_attribute is not None:
            shape_value: Any = edge.attributes.get(self.shape_attribute)

            if shape_value is not None:
                raw: Any = shape_value.get() if hasattr(shape_value, "get") else (
                    shape_value
                )

                if isinstance(raw, Polyline):
                    return raw.length()

        return self._direct_length(edge=edge)

    def _direct_length(self, edge: Any) -> float:
        """
        Returns the direct geodesic distance between the endpoints.

        Args:
            edge (Any): The edge StructInstance.

        Returns:
            float: Distance in meters.
        """

        planet: Any = get_current_planet()

        start: Coordinate = self._node_position(
            local_id=self._endpoint_local_id(
                instance=edge, attribute=self.from_attribute
            )
        )
        end: Coordinate = self._node_position(
            local_id=self._endpoint_local_id(
                instance=edge, attribute=self.to_attribute
            )
        )

        return start.distance(other=end, planet=planet)

    def _node_position(self, local_id: int) -> Coordinate:
        """
        Returns the position Coordinate of a node.

        Args:
            local_id (int): The node local id.

        Returns:
            Coordinate: The position.

        Raises:
            TypeError: If the node has no position attribute.
        """

        node: Any = self.nodes_table.get(local_id=local_id)

        value: Any = node.attributes.get(self.position_attribute)

        if value is None:
            raise TypeError(
                f"Node {local_id} has no '{self.position_attribute}' value"
            )

        raw: Any = value.get() if hasattr(value, "get") else value

        position: Coordinate = raw

        return position

    def _edge_cost(
        self,
        edge_local_id: int,
        transport_mode: str,
        cost_mode: str,
    ) -> Optional[float]:
        """
        Returns the cost of an edge, or None when the edge is
        unusable.

        Args:
            edge_local_id (int): The edge local id.
            transport_mode (str): The transport mode filtering edges.
            cost_mode (str): "distance" (meters) or "time" (seconds).

        Returns:
            Optional[float]: The cost, or None when the mode is
            unavailable.
        """

        cache: Optional[dict[tuple[str, str], float]] = (
            self._edge_costs.get(edge_local_id)
        )

        key: tuple[str, str] = (cost_mode, transport_mode)

        if cache is not None and key in cache:
            return cache[key]

        edge: Any = self._edge_instance(local_id=edge_local_id)

        if not self._allows_mode(edge=edge, mode=transport_mode):
            cost: Optional[float] = None
        elif cost_mode == "time":
            cost = self._time_cost(edge=edge, transport_mode=transport_mode)
        else:
            cost = self._edge_length(edge=edge)

        if cache is None:
            cache = {}
            self._edge_costs[edge_local_id] = cache

        if cost is not None:
            cache[key] = cost

        return cost

    def _allows_mode(self, edge: Any, mode: str) -> bool:
        """
        Checks whether an edge carries a mode.

        Args:
            edge (Any): The edge StructInstance.
            mode (str): The routing mode.

        Returns:
            bool: True when the edge is usable in the mode (always
            true when the edge struct has no modes attribute).
        """

        if self.modes_attribute is None:
            return True

        value: Any = edge.attributes.get(self.modes_attribute)

        if value is None:
            return True

        raw: Any = value.get() if hasattr(value, "get") else value

        return mode in raw

    def _unannotated_speed(self) -> float:
        """
        Returns the speed assumed for edges without speed data.

        The network's fastest seen speed when known; a conservative
        urban default otherwise (the heuristic divides by the same
        value, so it stays admissible).

        Returns:
            float: Speed in m/s.
        """

        if self._max_speed_mps > 0.0:
            return self._max_speed_mps

        return _DEFAULT_SPEED_MPS

    def _time_cost(self, edge: Any, transport_mode: str) -> Optional[float]:
        """
        Returns the traversal time of an edge in seconds.

        The speed for the transport mode is preferred; an edge without
        that entry falls back to its fastest speed so partially
        annotated networks stay connected.

        Args:
            edge (Any): The edge StructInstance.
            transport_mode (str): The transport mode.

        Returns:
            Optional[float]: Seconds, or None when the edge has no
            usable speed.
        """

        if self.speed_attribute is None:
            return None

        value: Any = edge.attributes.get(self.speed_attribute)

        if value is None:
            # Missing (defaulted) speed attributes behave like empty
            # ones: optimistic estimate keeps the edge connected
            return self._edge_length(edge=edge) / self._unannotated_speed()

        raw: Any = value.get() if hasattr(value, "get") else value

        if not raw:
            # Unannotated edges stay connected with an optimistic
            # estimate (the network's fastest seen speed); this also
            # keeps the time heuristic admissible
            return self._edge_length(edge=edge) / self._unannotated_speed()

        length: float = self._edge_length(edge=edge)

        preferred: Optional[float] = raw.get(transport_mode)

        if (
            isinstance(preferred, (int, float))
            and not isinstance(preferred, bool)
            and preferred > 0.0
        ):
            speed_mps: float = preferred * _KMH_TO_MPS

            if speed_mps > self._max_speed_mps:
                self._max_speed_mps = speed_mps

            seconds: float = length / speed_mps

            return seconds

        best_speed_mps: Optional[float] = None

        for speed_kmh in raw.values():
            if isinstance(speed_kmh, bool) or not isinstance(speed_kmh, (int, float)):
                continue

            fallback_mps: float = speed_kmh * _KMH_TO_MPS

            if fallback_mps <= 0.0:
                continue

            if best_speed_mps is None or fallback_mps > best_speed_mps:
                best_speed_mps = fallback_mps

            if fallback_mps > self._max_speed_mps:
                self._max_speed_mps = fallback_mps

        if best_speed_mps is None:
            return None

        return length / best_speed_mps

    # --- Routing ---

    def snap(self, coord: Coordinate) -> Optional[int]:
        """
        Snaps a coordinate to the nearest node local id.

        Args:
            coord (Coordinate): The query coordinate.

        Returns:
            Optional[int]: The node local id, or None when the network
            has no nodes.
        """

        index: Any = self._ensure_node_index()

        result: list[int] = index.nearest(coord=coord, k=1)

        if not result:
            return None

        # The shared spatial index keys rows by global entity id;
        # network node ids are local ids in the nodes table
        return EntityId(value=result[0]).local_id

    def _absorb_speed(self, instance: Any) -> None:
        """
        Tracks an edge's fastest speed for the time heuristic.

        Args:
            instance (Any): The edge StructInstance.
        """

        if self.speed_attribute is None:
            return None

        value: Any = instance.attributes.get(self.speed_attribute)

        if value is None:
            return None

        raw: Any = value.get() if hasattr(value, "get") else value

        if not raw:
            return None

        for speed_kmh in raw.values():
            if isinstance(speed_kmh, bool) or not isinstance(
                speed_kmh, (int, float)
            ):
                continue

            if speed_kmh > 0.0:
                speed_mps: float = speed_kmh * _KMH_TO_MPS

                if speed_mps > self._max_speed_mps:
                    self._max_speed_mps = speed_mps

    def _mode_components(self, mode: str) -> dict[int, int]:
        """
        Maps every mode-usable node to its connected component id.

        Args:
            mode (str): The transport mode.

        Returns:
            dict[int, int]: Node local id to component id.
        """

        cached: Optional[dict[int, int]] = self._component_cache.get(mode)

        if cached is not None:
            return cached

        parent: dict[int, int] = {}
        rank: dict[int, int] = {}

        def find(node: int) -> int:
            """
            Returns the component root of one node (with path
            compression).

            Args:
                node (int): The node local id.

            Returns:
                int: The root local id.
            """

            root: int = node

            while parent.get(root, root) != root:
                root = parent[root]

            while parent.get(node, node) != node:
                parent[node], node = root, parent[node]

            return root

        def union(node_a: int, node_b: int) -> None:
            """
            Merges two components (by rank).

            Args:
                node_a (int): One side's node local id.
                node_b (int): The other side's node local id.
            """

            root_a: int = find(node_a)
            root_b: int = find(node_b)

            if root_a == root_b:
                return None

            rank_a: int = rank.get(root_a, 0)
            rank_b: int = rank.get(root_b, 0)

            if rank_a < rank_b:
                parent[root_a] = root_b

            elif rank_a > rank_b:
                parent[root_b] = root_a

            else:
                parent[root_b] = root_a
                rank[root_a] = rank_a + 1

        for node_id in self.node_ids:
            parent[node_id] = node_id

        for node_id, entries in self.adjacency.items():
            for edge_id, neighbor in entries:
                edge: Any = self._edge_instance(local_id=edge_id)

                if not self._allows_mode(edge=edge, mode=mode):
                    continue

                union(node_id, neighbor)

        components: dict[int, int] = {
            node_id: find(node_id) for node_id in parent
        }

        self._component_cache[mode] = components

        return components

    def _largest_component(self, components: dict[int, int]) -> int:
        """
        Returns the component id with the most nodes.

        Args:
            components (dict[int, int]): Node to component mapping.

        Returns:
            int: The dominant component id (0 when empty).
        """

        counts: dict[int, int] = {}

        for component in components.values():
            counts[component] = counts.get(component, 0) + 1

        return max(counts, key=lambda c: counts[c]) if counts else 0

    def _mode_snap(self, coord: Coordinate, mode: str) -> Optional[int]:
        """
        Snaps a coordinate to the nearest node usable in a mode.

        The plain snap ignores transport modes; a point in a
        pedestrian zone would otherwise anchor a car route to a
        walk-only node. Candidates are scanned nearest-first, preferring
        nodes of the mode's largest connected component so courtyards
        and parking-lot stubs do not strand the route.

        Args:
            coord (Coordinate): The query coordinate.
            mode (str): The transport mode.

        Returns:
            Optional[int]: The node local id, or None when no nearby
            node is usable in the mode.
        """

        index: Any = self._ensure_node_index()

        # Entity-id keys until converted; the scan widens until a
        # node of the mode's dominant component is found (dense
        # pedestrian zones can bury the car network deep in the
        # nearest-first order)
        fallback: Optional[int] = None

        for radius in (32, 128, 512, 2048):
            keys: list[int] = index.nearest(coord=coord, k=radius)

            components: Optional[dict[int, int]] = None

            for key in keys:
                node_id: int = EntityId(value=key).local_id

                usable: bool = False

                for edge_id, _neighbor in self.adjacency.get(node_id, []):
                    edge: Any = self._edge_instance(local_id=edge_id)

                    if self._allows_mode(edge=edge, mode=mode):
                        usable = True

                        break

                if not usable:
                    continue

                if fallback is None:
                    fallback = node_id

                if components is None:
                    components = self._mode_components(mode=mode)

                # The largest component wins: stubs and courtyards
                # are real nodes but dead ends for routing
                if components.get(
                    node_id, -1
                ) == self._largest_component(components=components):
                    return node_id

        return fallback

    def route(
        self,
        start: Coordinate,
        end: Coordinate,
        mode: str,
        cost: str = "distance",
    ) -> Path:
        """
        Routes between two coordinates over mode-filtered edges.

        Both endpoints snap to the nearest node; A* (Dijkstra when the
        heuristic degenerates) finds the cheapest path.

        Args:
            start (Coordinate): The start coordinate.
            end (Coordinate): The end coordinate.
            mode (str): The transport mode filtering edges (for
                example "car" or "pedestrian").
            cost (str, optional): Cost mode, "distance" (meters) or
                "time" (seconds via per-mode speeds). Defaults to
                "distance".

        Returns:
            Path: The routing result.

        Raises:
            ValueError: If no node is near an endpoint, or no path
                exists in the mode.
        """

        self._resolve_pending_edges()

        if cost not in ("distance", "time"):
            raise ValueError(
                f"Unknown cost mode '{cost}' (expected 'distance' or 'time')"
            )

        start_id: Optional[int] = self._mode_snap(coord=start, mode=mode)
        end_id: Optional[int] = self._mode_snap(coord=end, mode=mode)

        if start_id is None:
            raise ValueError("No network node near the start coordinate")

        if end_id is None:
            raise ValueError("No network node near the end coordinate")

        planet: Any = get_current_planet()

        def heuristic(node_id: int) -> float:
            """
            Estimates the remaining cost to the target.

            Args:
                node_id (int): The node local id.

            Returns:
                float: Meters for the distance mode, seconds for time
                (geodesic distance over the fastest known speed).
            """

            distance: float = self._node_position(local_id=node_id).distance(
                other=end, planet=planet
            )

            if cost == "time":
                if self._max_speed_mps <= 0.0:
                    return 0.0

                return distance / self._max_speed_mps

            return distance

        counter: int = 0

        # Heap entries: (f, tie-breaker, g, node) so stale pushes are
        # detected against the best known g score
        frontier: list[tuple[float, int, float, int]] = [
            (heuristic(node_id=start_id), counter, 0.0, start_id)
        ]
        g_score: dict[int, float] = {start_id: 0.0}
        came_from: dict[int, tuple[int, int]] = {}

        while frontier:
            _f, _tie, popped_g, current = heapq.heappop(frontier)

            if popped_g > g_score.get(current, math.inf):
                continue

            if current == end_id:
                return self._reconstruct(
                    came_from=came_from, end_id=end_id, cost=cost, mode=mode
                )

            for edge_id, neighbor in self.adjacency.get(current, []):
                edge_cost: Optional[float] = self._edge_cost(
                    edge_local_id=edge_id,
                    transport_mode=mode,
                    cost_mode=cost,
                )

                if edge_cost is None:
                    continue

                new_g: float = popped_g + edge_cost

                if new_g < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = new_g
                    came_from[neighbor] = (current, edge_id)

                    counter += 1

                    heapq.heappush(
                        frontier,
                        (
                            new_g + heuristic(node_id=neighbor),
                            counter,
                            new_g,
                            neighbor,
                        ),
                    )

        raise ValueError(
            f"No path exists for mode '{mode}' between the given "
            f"coordinates"
        )

    def _reconstruct(
        self,
        came_from: dict[int, tuple[int, int]],
        end_id: int,
        cost: str,
        mode: str,
    ) -> Path:
        """
        Rebuilds the Path from the A* came-from map.

        Args:
            came_from (dict[int, tuple[int, int]]): Node local id to
                (previous node, via edge) pairs.
            end_id (int): The goal node local id.
            mode (str): The routing mode.

        Returns:
            Path: The routing result.
        """

        node_ids: list[int] = [end_id]
        edge_ids: list[int] = []

        cursor: int = end_id

        while cursor in came_from:
            previous, via = came_from[cursor]

            edge_ids.append(via)
            node_ids.append(previous)

            cursor = previous

        node_ids.reverse()
        edge_ids.reverse()

        return Path(
            nodes=[
                self.nodes_table.get(local_id=node_id) for node_id in node_ids
            ],
            edges=[
                self._edge_instance(local_id=edge_id) for edge_id in edge_ids
            ],
            cost=0.0 if len(node_ids) == 1 else self._path_cost(
                edge_ids=edge_ids, cost=cost, mode=mode
            ),
            mode=mode,
        )

    def _path_cost(
        self,
        edge_ids: list[int],
        cost: str,
        mode: str,
    ) -> float:
        """
        Sums the cached edge costs along a path.

        Args:
            edge_ids (list[int]): The traversed edge local ids.
            mode (str): The routing mode.

        Returns:
            float: The total cost.
        """

        total: float = 0.0

        for edge_id in edge_ids:
            edge_cost: Optional[float] = self._edge_cost(
                edge_local_id=edge_id,
                transport_mode=mode,
                cost_mode=cost,
            )

            if edge_cost is not None:
                total += edge_cost

        return total
