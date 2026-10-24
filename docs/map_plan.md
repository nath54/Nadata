# Map & World Plan

A geospatial layer for Nadata: coordinates on real or fictional planets,
hierarchical regions, road networks with transport modes, buildings with
floor-by-floor interiors, field layers (elevation, terrain, sea level),
planet-scale spatial indexing, and streaming for open-world use.

Planning document — no implementation contract beyond the sequencing at
the end. Companion to the memory plan (streaming reuses the bounded
decode cache) and orthogonal to the autograd plan.

> **STATUS: PHASES A-E IMPLEMENTED.** Phase A (planet settings,
> `Coordinate`, geodesy builtins, binary cell encodings), Phase B
> (`Polyline` / `Polygon`, predicates, shape sugar), Phase C (the
> `spatial_index` builtin, `locate` / `nearest` / `within` queries,
> hierarchical descent, dump-time row clustering), Phase D
> (`GeoGrid[T]` field layers, bilinear `sample`, sea-level
> registration, `altitude` integration) and Phase E (`network_index`,
> `route` with distance/time costs, mode filtering, incremental
> maintenance) are merged with tests (`tests/test_geodesy.py`,
> `tests/test_geometry.py`, `tests/test_spatial_index.py`,
> `tests/test_geogrid.py`, `tests/test_network.py`). Phase F is not
> started.
>
> Phase C implementation notes (deviations from the original text
> below): queries take no attribute argument — a struct with exactly
> one spatial index is disambiguated automatically, several indexes
> raise; multi-cell geometries are deduplicated across cells in
> `nearest` / `within`; the "definitely outside" fast path is disabled
> (`circumradius = inf`) when the geometry spans more than a
> hemisphere of its anchor, and antimeridian-straddling rings (lon
> span > 180°) use a lat/lon-space anchor instead of the 3D direction
> mean; the cell lower bound applies the sine rule with the query
> point's latitude and a max-curvature radius on spheroids.
>
> Phase E implementation notes: `route(from, to, mode)` takes the
> transport mode ("car", "walk", ...) for edge filtering plus an
> optional `cost` keyword ("distance" meters, default, or "time"
> seconds via the edge's per-mode speeds; a missing speed entry
> falls back to the edge's fastest speed so partially annotated
> networks stay connected). Networks are session-scoped and
> registered on both the node and edge tables; node and edge local
> ids live in separate per-table spaces, so removal hooks carry the
> calling table. Edges inserted as shells (attributes populated in a
> later pass) park as pending and resolve before the next route.
> Endpoint snapping reuses a cell index over the node positions;
> `route` needs exactly one network per session (several raise).
>
> Phase D implementation notes: the cell size is declared in the type
> bracket (`GeoGrid[Float64, 3600]`, default 30 arc-seconds) matching
> the Tensor convention; grid attributes need an explicit `= {}`
> default like the other grids; the per-value sample default
> (`set_default`) is session state — the binary encoding persists
> cells only, so scripts re-establish defaults and
> `register_sea_level` after load (same pattern as index rebuilds);
> bilinear sampling weights the four surrounding cell centers,
> renormalizing over cells that have data, wrapping longitude and
> clamping latitude at the poles.

---

## 1. Goals

One data model serving two audiences:

1. **Game engines** — stream a planet-scale world around a player:
   districts, streets, building interiors loaded on demand, with
   adjacency graphs for indoor/outdoor navigation.
2. **Navigation applications** — point-in-region lookups, nearest-X
   queries, and routing across road/rail/pedestrian networks, without
   caring about region boundaries.

Both come from the same primitives: the only difference is which
queries a client runs. The engine ships *primitives and indexes*; the
domain semantics (countries, houses, doors) live in user schemas.

## 2. Current state (verified against the code)

What already exists and what we build on:

- **Opt-in hash indexes** — `index(Human, "email")` (`IndexFunction`,
  session-scoped) + `get_by()` O(1) lookups. The spatial index below
  deliberately copies this pattern.
- **Sparse spatial grids** — `GridSquare2D` / `GridHex2D` /
  `GridCube3D` over *integer* cells with the member-call pattern
  (`GridCallable`: `at` / `set_at` / `has` / ...). `GeoGrid` below
  generalizes this to continuous planet coordinates.
- **Immutable home value types** — `DateTime`, `Color` show the model
  for `Coordinate` (value class, `Type` subclass, binary cell encoding,
  method-style members).
- **Persistence machinery** — varint/zigzag primitives in
  `binary_cells.py`, lazy rows + bounded LRU decode cache,
  `pin`/`unpin`, dirty-only delta appends, tombstones.
- **Settings** — `get_setting` / `set_setting` freshness markers: the
  natural home for the planet definition.
- **Self/circular entity references** (ex2) — the hierarchy mechanism
  (`Region.parent: Region`) needs no new engine support.
- **I/O builtins** — `write_json` gives GeoJSON export for free;
  `graph.py` / `vec.py` in home_types are empty stubs.

What does not exist: any angular-coordinate math, geometry primitives,
spatial predicates, spatial indexes, or network routing.

## 3. Principles

1. **No god type.** There is no `Map` builtin that knows about regions
   or roads. The engine provides `Coordinate` / `Polyline` / `Polygon`
   / `GeoGrid` plus two opt-in indexes; everything else is user schema.
2. **Indexes follow the `index()` pattern.** Opt-in, session-scoped,
   rebuilt on open, declared with a builtin call. No hidden global
   state, consistent with hash indexes.
3. **Value types are immutable** (the DateTime model). Geometry never
   mutates in place; derived indexes stay valid by construction.
4. **Geodetic latitude everywhere.** Matches GPS / GeoJSON / OSM real-
   world data. Radius from planet center is stored per point, which is
   what makes tunnels, bridges and flying objects unambiguous.
5. **Wrapping, not splitting, in the math.** Longitude ±180 is handled
   inside `distance` / `contains` / index keys; the external converter
   may split polygons at the antimeridian only for rendering.
6. **The reference domain schema lives in an example**, not the
   engine. `AdminRegion`, `Building`, `Room`, `Door` are example
   structs users copy and adapt (including per-country subclasses).
7. **Ingestion is external.** A converter script in the repo turns
   GeoJSON into `.nastruct`/`.nadata`; the engine itself stays pure
   stdlib and knows nothing about GeoJSON.
8. **No new dependencies.** Sphere/spheroid math is closed-form;
   routing is A* over user rows.

## 4. Planet model and geodesy

### 4.1 Planet definition

Project-level settings (read via `get_setting`):

```text
set_setting("planet.kind", "ellipsoid");
set_setting("planet.equatorial_radius", 6378137.0);   // meters
set_setting("planet.polar_radius", 6356752.3);        // meters
```

- `"sphere"` accepts a single `planet.radius`; `"ellipsoid"` accepts
  equatorial + polar (oblate spheroid). Triaxial ellipsoids are out of
  scope (see open points).
- Geodesy builtins read the setting implicitly; every function accepts
  an optional explicit radius/planet override for multi-planet data in
  one session.

### 4.2 Coordinate semantics

A point is `(lat, lon, radius)` — angular latitude/longitude in
degrees, radius in meters from the planet center.

- **Surface radius** at latitude is *derived* (analytic for sphere and
  spheroid) and used as a fallback, never stored implicitly.
- Storing radius per point is what makes underground tunnels
  (`radius < surface`), bridges and flight levels expressible without
  a separate altitude concept.
- Latitude is geodetic. On an ellipsoid, geocentric conversion is an
  available builtin (`geocentric_lat`) for the rare client that needs
  it, not the storage format.

### 4.3 Geodesy builtins

Free functions and `Coordinate` members, one delegating to the other:

| Operation | Notes |
|---|---|
| `distance(a, b)` | Great-circle (sphere) / geodesic approximation (spheroid); chord fallback for exotic planets. Meters. |
| `bearing(a, b)`, `destination(a, bearing, dist)` | Standard closed forms. |
| `interpolate(a, b, t)` | Along the great circle; t in `[0, 1]`. |
| `surface_radius(lat)` | Analytic; the ellipsoid formula. |
| `altitude(c)` | `c.radius - surface_radius(c.lat)` — or minus the sampled sea-level layer when one is registered (§7). |

## 5. Builtin value types

All immutable; all get new binary cell encodings (§10).

### 5.1 `Coordinate`

- Constructor `Coordinate(lat: Float, lon: Float, radius: Float)`.
- Members (method-style, mirroring `GridCallable`): `distance`,
  `bearing`, `destination`, `interpolate`, `altitude`, plus `lat` /
  `lon` / `radius` accessors. Free-function forms delegate to these.
- Hashable via quantized key (index cells, §6) for dict/set usage.
- Normalizes lon into `[-180, 180)` on construction.

### 5.2 `Polyline`

- Constructor from an array of Coordinates. Internally stores
  delta-encoded vertices (§10).
- Members: `length` (geodesic, per segment), `point_at(t)` /
  `point_at_distance(d)`, `distance_to(Coordinate)`,
  `intersects(Polyline)` (later), `vertices`.

### 5.3 `Polygon`

- Constructor from a closed ring of Coordinates (last ≠ first, closed
  implicitly). Holes are a later extension (open points).
- Members: `area` (spherical excess, exact on sphere; spheroid
  approximation), `perimeter`, `centroid`, `bbox`, `contains(Coordinate)`
  (ray casting on the unwrapped sphere, antimeridian-safe),
  `distance_to(Coordinate)`.
- Shapes of any complexity are polylines/polygons with enough
  vertices; parametric sugar (`circle(center, radius_m, segments)`,
  `ellipse(...)`) are builtins that *produce* polygons, not new types.

### 5.4 `GeoGrid[T]`

Field layers on quantized cells (the generalization of
`ValueSpatialGrid` to planet coordinates):

```text
elevation = GeoGrid[Float](cell_arcsec: Float = 30.0);
elevation.set_at(Coordinate(...), 142.5);
h: Float = elevation.sample(Coordinate(...));   // bilinear, meters
```

- `set_at(coord, value)` writes the cell containing the coordinate;
  `sample(coord)` bilinearly interpolates the 2×2 neighborhood,
  falling back to a per-grid `default` (a callable or value — for a
  sea-level grid the default is `surface_radius`) when a cell is
  empty. Sparse: only set cells are stored.
- Cell size is a constructor parameter in arc-seconds; cells are keyed
  by quantized `(lat_cell, lon_cell)` with wrapping at ±180.

## 6. Spatial index and hierarchical locate

### 6.1 Declaration

```text
spatial_index(Region, "boundary", lod: Int = 12);
```

Copies the `index()` shape: struct binding + attribute name + optional
keyword. Session-scoped, rebuilt on open, maintained on insert /
update / removal like hash indexes. `lod` controls quantization (§6.3).

### 6.2 Queries

- `locate(Struct, Coordinate)` — the hierarchy descent: starting from
  candidate roots (structs whose instances have no parent in the
  *index*), use index cells to find candidate children containing the
  point, test `Polygon.contains` only on those, and descend. Cost is
  O(depth × candidates), not O(regions). Works for any self-
  referencing struct with a geometry attribute; no hardcoded "parent"
  attribute name beyond a declaration argument.
- `nearest(Struct, Coordinate, k)` / `within(Struct, Coordinate, radius)`
  — ring search over index cells outward from the query point,
  refining candidates with `distance_to`.

### 6.3 Center + radii fast paths (dual mode)

Polygon tests are O(vertices) and the sphere changes only constants —
the cost driver is *how many polygons get tested*. Each indexed region
therefore gets two cached scalars, computed once at index build from
the boundary (indexes already scan every row on open, matching the
hash-index pattern; nothing is stored on rows, so schema evolution and
re-merges recomputed them for free):

- **Inradius** — min distance from center to the boundary (exact: min
  segment-to-center distance over edges). A query point closer than
  this to the center is *definitely inside*.
- **Circumradius** — max distance from center to any vertex. A point
  farther than this is *definitely outside*.

`locate` / `nearest` / `within` then use a three-way test per
candidate: O(1) accept, O(1) reject, else O(v) ray-cast. Compact
regions (cities, départements — most admin geography) almost never pay
the polygon test; crescents and archipelagos degrade gracefully to
always-testing, never to wrong answers. The center defaults to the
polygon centroid, overridable per row for pathological shapes.

Comparisons use **chord distances** (Cartesian 3D, ~5 flops, no
arccos): chord distance is monotonic in great-circle distance, so with
radii cached as chords the fast paths stay exact. Geodesic math runs
only in the polygon fallback.

An explicit **approximate mode** (`exact: Bool = false`) answers by
nearest center with no polygon tests at all — a center-Voronoi
approximation for game streaming prefetch and LOD decisions where an
occasional wrong guess is harmless. Exact mode is the default.

### 6.4 Cell resolution: per index, not per planet

Resolution is a **per-index parameter**, because a whole-planet single
cell size cannot serve both continents and rooms:

- **Default, scaled to the planet:** with no explicit `lod`, the cell
  edge is derived from the planet's circumference — coarse enough that
  a planet-wide index stays small, fine enough to be useful.
- **Explicit `lod`:** cell edge ≈ `circumference / 2^lod`. A street
  network wants a fine lod; a continent layer a coarse one.
- **Hierarchies get resolution for free:** the recommended pattern is
  one struct (and one index) per hierarchy level — `Continent`,
  `Country`, `Departement`, ... each with its own `lod` forming a
  resolution pyramid. `locate()` descends through per-level indexes,
  so each level queries at its own resolution. A single flat struct
  with `parent` refs still works, at one resolution.

### 6.5 Streaming hooks

- **Row clustering:** when a struct has a spatial index, the dumper
  sorts that struct's rows by spatial key before writing, so "load
  district X" reads contiguous pages instead of scattered ones.
- **Region-bounded iteration:** `within` / `locate` results feed the
  existing lazy-row machinery — rows decode on demand under the
  bounded LRU; a game engine `pin`s the loaded neighborhood and lets
  the rest evict.
- Prefetching a radius around a point is `within` + pinning composed
  from scripts; no new primitive needed initially.

## 7. Field layers

Layer kinds and their carriers:

| Layer | Carrier |
|---|---|
| Elevation | `GeoGrid[Float]` of surface radii (or altitudes) |
| Sea level | `GeoGrid[Float]`, sparse, default = `surface_radius` — exactly the "few points, easy default" the vision calls for |
| Terrain type | `GeoGrid[Str]` (or an enum-like struct per cell for richer metadata) |
| Regions | User structs + spatial index (§6) |
| Transport networks | User structs + network index (§8) |

`altitude()` consults a registered sea-level grid when one exists
(registration is a builtin call, `register_sea_level(grid)`), otherwise
falls back to `surface_radius`. A layer is thus just data; "layers"
are a convention of which grids/structs a project keeps, not an engine
container.

## 8. Network layers and routing

### 8.1 Model: physical elements, per-mode overlays

Physical infrastructure is stored once; modes are overlays:

```text
class NetworkNode {
    position: Coordinate;
    kind: Str;                       // "junction", "entrance", "stop", ...
}

class NetworkEdge {
    from_node: NetworkNode;
    to_node: NetworkNode;
    shape: Poly = Poly(vertices: Array[Coordinate]);
    kind: Str;                       // "ground", "bridge", "tunnel", "underground"
    modes: Array[Str];               // ["car", "bike"], ["pedestrian"], ["train"], ...
    speed: Dict[Str, Float];         // per-mode speeds (km/h)
}
```

- A street walkable *and* drivable is one `NetworkEdge` row with two
  modes — geometry is never duplicated per mode.
- `Poly` above is a placeholder for how edges carry geometry: either a
  `Polyline` attribute directly, or a lightweight struct wrapping one;
  decided at implementation start (open points).
- Indoor navigation is the same machinery: rooms are nodes, doors and
  stairs are edges — one `route()` call spans street → door → room →
  stairs → room.

### 8.2 Declaration and queries

```text
network_index(NetworkNode, NetworkEdge);
route(from: Coordinate, to: Coordinate, mode: Str) -> Path;
```

- `route()` snaps both endpoints to the nearest network node
  (`nearest` on the node index), runs A* over edges filtered by mode,
  and returns a `Path` (sequence of edges + total cost). Cost modes:
  `distance`, or `time` via per-edge mode speeds. Weight penalties for
  `kind` (e.g. bridges) are per-mode settings, not code.
- User-defined cost callbacks per edge are explicitly *out of scope*
  for the first version — interpreter callbacks per edge would
  dominate runtime. Revisit only with profiling evidence.

## 9. The Earth example

`examples/ex13_earth` — the reference domain schema and the flagship
acceptance test. Hierarchy: `World → Continent → Country → Region →
Departement → City → District → Street → Building → Floor → Room →
Door`, plus road and rail networks and elevation/terrain/sea-level
grids.

- **Generic base, country-specific leaves.** `AdminRegion` carries
  `parent`, `boundary`, `name`; `Departement`, `County`, `Prefecture`
  are sibling subclasses — France's découpage and any other country's
  coexist in one schema. Fictional worlds reuse `AdminRegion` directly.
- **Interiors.** A `House` example with 2 floors, 5 rooms, doors as
  edges (including the front door edge joining a room to a street
  node) demonstrates the game-engine and navigation use cases on the
  same data. Stairs and elevators are elements, not attributes —
  games will render them.
- **Ingestion.** `tools/geojson_to_nadata.py` (stdlib only): simplified
  Natural-Earth-style polygons → `AdminRegion` rows, per-level files.
  The repo ships the generator, not megabytes of data; the example
  documents the exact regeneration command.
- **Scenario scripts:** `#manual(locate_and_route)` — locate a point,
  route car vs pedestrian between two addresses, walk a house's
  adjacency graph; `#manual(stream_district)` — touch every element of
  a district under a bounded cache to demonstrate streaming.

## 10. Persistence

- **New cell encodings** in `binary_cells.py`: `Coordinate` (3× Float64,
  or quantized delta for polylines), `Polyline` (varint count +
  zigzag-varint quantized deltas — meters-level quantization keeps
  planet-scale polylines small), `Polygon` (as polyline + closed flag),
  `GeoGrid` (cell key dict + typed values, like spatial grids today).
- **Index persistence** stays session-scoped (rebuilt on open),
  matching hash indexes; only row data is durable.
- **Row clustering** (§6.5) applies at dump/compact time for
  spatially indexed structs.
- Large polylines exceed one cell payload only in extreme cases; the
  existing chunked-record machinery handles spillover.

## 11. Integration checklist

| Touch point | Change |
|---|---|
| `src/core/values/values.py` | `ValueCoordinate`, `ValuePolyline`, `ValuePolygon`, `ValueGeoGrid` |
| `src/core/types/types.py` | `TypeCoordinate`, `TypePolyline`, `TypePolygon`, `TypeGeoGrid`; builtin-type registry |
| `src/core/expressions/builtins.py` | Geodesy functions, constructors, `spatial_index`, `network_index`, `route`, `register_sea_level`, `locate`/`nearest`/`within`; `CoordinateCallable` member dispatcher |
| `src/core/database/persistence/binary_cells.py` | Cell encodings + decode |
| `src/core/database/persistence/dumper.py` | Row clustering for spatially indexed structs |
| `src/core/database/database.py` | Index registries alongside hash indexes |
| `src/parser/nastruct_parser/resolver.py` | New builtin types resolvable in schemas |
| `tests/test_grammar.py` | New builtins, keywords, scopes |
| `editors/vscode-nadata` | Grammar additions |
| `docs/syntax.md`, `docs/api.md`, `docs/benchmarks.md` | Language/API reference; a "planet-scale streaming" benchmark section |

## 12. Sequencing

| Order | Phase | Contents | Depends on |
|---|---|---|---|
| 1 | **A — Foundation** | Planet settings, `Coordinate` value type, geodesy builtins (sphere + spheroid, wrapping), cell encoding, grammar/VS Code updates | — |
| 2 | **B — Geometry** | `Polyline` / `Polygon`, predicates (`contains`/`area`/`length`/`centroid`/`distance_to`), shape-constructor sugar | A |
| 3 | **C — Spatial index** | `spatial_index` + `locate`/`nearest`/`within`, center/radii fast paths (§6.3), approximate mode, hierarchy descent, dumper row clustering, streaming iteration | B |
| 4 | **D — Field layers** | `GeoGrid[T]`, `sample` interpolation, sea-level registration, `altitude` integration | A, B |
| 5 | **E — Networks & routing** | `network_index`, `route` with `distance`/`time` modes, per-mode overlays, indoor door-graph unification | C |
| 6 | **F — Earth example** | `ex13_earth`, `tools/geojson_to_nadata.py`, scenario scripts, docs, benchmarks | A–E |

Each phase ships independently usable: A+B alone give users
coordinates and shapes; C unlocks hierarchy queries; F is the
acceptance story for the whole plan.

## 13. Open points

1. **Edge geometry carrier** — `Polyline` attribute directly vs. a
   wrapping struct; decide at the start of Phase E by checking how the
   resolver handles geometry-typed attributes on entity structs.
2. **Spheroid `contains`/`area` accuracy** — planar ray-cast on an
   equirectangular unwrapping is fast and correct for region-sized
   polygons; geodesic-exact area can wait for benchmarks showing need.
3. **Polygon holes** (courtyards, enclaves) — deferred; the ring
   representation admits a `holes: Array[Polygon]` extension without
   breaking anything.
4. **Triaxial ellipsoids** — deferred; the setting schema leaves room
   for a third axis.
5. **`GeoGrid` multi-resolution** — start single-resolution per grid;
   pyramid/quadtree only if `sample` cost at planet scale demands it.
6. **Rendering/export** — GeoJSON out via `write_json` is free; a
   richer export (SVG tiles, KML) is a tools-level concern, not engine.
7. **User cost callbacks for `route`** — revisit after profiling; the
   builtin mode set covers the common cases.
8. **Polyline / polygon delta compression** — Phase B encodes
   geometry vertices as exact 3 x float64 per vertex (matching the
   engine's exact-round-trip float policy); delta + zigzag varint
   compression can layer on top once benchmarks justify it, but it
   cannot be bit-exact, so it would change equality semantics.
