"""
Road Network Extraction, Flood Severing, and Evacuation Routing Module.
Extracts drive network using OSMnx, identifies and severs flooded edges
intersecting inundation polygons, and computes Dijkstra shortest safe paths.
"""

import os
import math
from typing import Tuple, List, Dict, Any, Optional
import networkx as nx
import osmnx as ox
from shapely.geometry import box, LineString, Polygon, mapping, shape


# Maximum allowed distance (metres) from Mandi centre before a coordinate
# is considered outside the downloaded graph coverage area.
_GRAPH_COVERAGE_RADIUS_M = 7500


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return approximate great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Center coordinates for Mandi, Himachal Pradesh (Beas River valley)
DEFAULT_CENTER_LAT = 31.7088
DEFAULT_CENTER_LON = 76.9320
DEFAULT_RADIUS_METERS = 8000   # expanded from 4000 m — covers all drivable corridors in range


class RoadNetworkRouter:
    """
    Manages road network graphs, flood corridor severing, and dynamic rerouting.
    """

    def __init__(
        self,
        center_lat: float = DEFAULT_CENTER_LAT,
        center_lon: float = DEFAULT_CENTER_LON,
        radius_m: int = DEFAULT_RADIUS_METERS,
        cache_path: str = "data/mandi_drive_network.graphml",
        auto_load: bool = True,
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_m = radius_m
        self.cache_path = cache_path

        self.G: Optional[nx.MultiDiGraph] = None
        self.G_safe: Optional[nx.MultiDiGraph] = None
        self.flood_polygon: Optional[Polygon] = None
        self.severed_edges: List[Tuple[Any, Any, Any]] = []

        if auto_load:
            self.load_or_fetch_graph()

    def _generate_synthetic_fallback_graph(self) -> nx.MultiDiGraph:
        """
        Generates a synthetic street grid around Mandi coordinates as a deterministic
        offline fallback if OSMnx Overpass API encounters network timeouts or rate limits.
        """
        G = nx.MultiDiGraph()
        G.graph["crs"] = "epsg:4326"

        lat_step = 0.005
        lon_step = 0.005
        rows, cols = 8, 8

        node_id = 0
        grid = {}
        for r in range(rows):
            for c in range(cols):
                lat = self.center_lat + (r - rows / 2) * lat_step
                lon = self.center_lon + (c - cols / 2) * lon_step
                G.add_node(node_id, y=lat, x=lon)
                grid[(r, c)] = node_id
                node_id += 1

        # Add bidirectional edges
        for r in range(rows):
            for c in range(cols):
                u = grid[(r, c)]
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if nr < rows and nc < cols:
                        v = grid[(nr, nc)]
                        line = LineString([
                            (G.nodes[u]["x"], G.nodes[u]["y"]),
                            (G.nodes[v]["x"], G.nodes[v]["y"]),
                        ])
                        dist = 500.0  # approximate meters
                        G.add_edge(u, v, key=0, length=dist, geometry=line)
                        G.add_edge(v, u, key=0, length=dist, geometry=line)

        return G

    def load_or_fetch_graph(self, force_refresh: bool = False) -> nx.MultiDiGraph:
        """
        Load street graph from local cache or fetch from OpenStreetMap.
        """
        if not force_refresh and os.path.exists(self.cache_path):
            try:
                self.G = ox.load_graphml(self.cache_path)
                self.G_safe = self.G.copy()
                return self.G
            except Exception:
                pass

        try:
            center_point = (self.center_lat, self.center_lon)
            self.G = ox.graph_from_point(
                center_point, dist=self.radius_m, network_type="drive"
            )
            # Cache locally
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            ox.save_graphml(self.G, self.cache_path)
        except Exception as e:
            # Fallback to deterministic synthetic valley network
            self.G = self._generate_synthetic_fallback_graph()

        self.G_safe = self.G.copy()
        return self.G

    def get_default_flood_polygon(self) -> Polygon:
        """
        Generates the default Beas River corridor flood polygon based on network bounds.
        """
        if self.G is None:
            self.load_or_fetch_graph()

        nodes, edges = ox.graph_to_gdfs(self.G)
        minx, miny, maxx, maxy = edges.total_bounds

        # Submerge central river corridor (35% to 65% width, 35% to 60% height)
        poly = box(
            minx + 0.35 * (maxx - minx),
            miny + 0.35 * (maxy - miny),
            minx + 0.65 * (maxx - minx),
            miny + 0.60 * (maxy - miny),
        )
        return poly

    def sever_flooded_edges(
        self, flood_polygon: Optional[Polygon] = None
    ) -> Tuple[nx.MultiDiGraph, List[Tuple[Any, Any, Any]]]:
        """
        Identify road segments that intersect the flood polygon and sever them from G_safe.
        """
        if self.G is None:
            self.load_or_fetch_graph()

        if flood_polygon is None:
            flood_polygon = self.get_default_flood_polygon()

        self.flood_polygon = flood_polygon
        self.G_safe = self.G.copy()
        self.severed_edges = []

        for u, v, k, data in self.G.edges(keys=True, data=True):
            if "geometry" in data:
                geom = data["geometry"]
            else:
                geom = LineString([
                    (self.G.nodes[u]["x"], self.G.nodes[u]["y"]),
                    (self.G.nodes[v]["x"], self.G.nodes[v]["y"]),
                ])

            if geom.intersects(flood_polygon):
                self.severed_edges.append((u, v, k))
                if self.G_safe.has_edge(u, v, k):
                    self.G_safe.remove_edge(u, v, k)

        return self.G_safe, self.severed_edges

    def _snap_to_passable_node(
        self, lat: float, lon: float, candidate_nodes: List[Any]
    ) -> Any:
        """
        Find nearest node among candidate nodes using Euclidean distance.
        """
        best_node = None
        best_dist = float("inf")
        for n in candidate_nodes:
            ny = self.G.nodes[n]["y"]
            nx_coord = self.G.nodes[n]["x"]
            d = (ny - lat) ** 2 + (nx_coord - lon) ** 2
            if d < best_dist:
                best_dist = d
                best_node = n
        return best_node

    def find_evacuation_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        flood_polygon: Optional[Polygon] = None,
        weight: str = "length",
    ) -> Dict[str, Any]:
        """
        Calculates safe shortest evacuation route avoiding submerged road segments.
        Returns GeoJSON-compatible route dictionary.
        """
        if self.G_safe is None or flood_polygon is not None:
            self.sever_flooded_edges(flood_polygon)

        # --- Boundary guard: warn early if coordinates fall outside graph coverage ---
        for label, lat, lon in [
            ("Origin", origin_lat, origin_lon),
            ("Destination", dest_lat, dest_lon),
        ]:
            dist_from_centre = _haversine_m(
                lat, lon, DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON
            )
            if dist_from_centre > _GRAPH_COVERAGE_RADIUS_M:
                raise ValueError(
                    f"{label} coordinate ({lat:.4f}, {lon:.4f}) is {dist_from_centre/1000:.1f} km "
                    f"from Mandi centre, which exceeds the road network coverage radius of "
                    f"{_GRAPH_COVERAGE_RADIUS_M/1000:.1f} km. "
                    "Please move the pin closer to Mandi town."
                )

        # Determine safe connected components
        safe_components = list(nx.strongly_connected_components(self.G_safe))
        if not safe_components:
            # Fall back to weakly connected components if directed cycles are broken
            safe_components = list(nx.weakly_connected_components(self.G_safe))

        if not safe_components:
            raise nx.NetworkXNoPath("No navigable road components remaining in safe network.")

        largest_cc = max(safe_components, key=len)
        cc_nodes = list(largest_cc)

        # Snap origin and destination coordinates to nearest accessible nodes
        orig_node = self._snap_to_passable_node(origin_lat, origin_lon, cc_nodes)
        dest_node = self._snap_to_passable_node(dest_lat, dest_lon, cc_nodes)

        if orig_node is None or dest_node is None:
            raise nx.NetworkXNoPath("Unable to snap coordinates to safe network.")

        try:
            # Shortest path via Dijkstra algorithm
            route_nodes = nx.shortest_path(
                self.G_safe, source=orig_node, target=dest_node, weight=weight
            )
        except nx.NetworkXNoPath:
            # If directed path not found between specific snapped nodes, attempt undirected bridge
            u_safe = self.G_safe.to_undirected()
            route_nodes = nx.shortest_path(
                u_safe, source=orig_node, target=dest_node, weight=weight
            )

        # Extract coordinates and calculate metrics
        route_coords = []
        total_distance_m = 0.0

        for i, node in enumerate(route_nodes):
            lat = self.G.nodes[node]["y"]
            lon = self.G.nodes[node]["x"]
            route_coords.append((lat, lon))

            if i > 0:
                prev_node = route_nodes[i - 1]
                if self.G.has_edge(prev_node, node):
                    edge_data = list(self.G[prev_node][node].values())[0]
                    total_distance_m += edge_data.get("length", 50.0)
                else:
                    # Euclidean approximation
                    plat = self.G.nodes[prev_node]["y"]
                    plon = self.G.nodes[prev_node]["x"]
                    total_distance_m += math.hypot((lat - plat) * 111000, (lon - plon) * 111000)

        # Estimated travel time in minutes at 30 km/h emergency speed
        travel_time_min = round((total_distance_m / 1000.0) / 30.0 * 60.0, 1)

        # Convert to standard GeoJSON FeatureCollection
        geojson_data = self.to_geojson_feature_collection(
            route_coords=route_coords,
            origin_coords=(origin_lat, origin_lon),
            dest_coords=(dest_lat, dest_lon),
            distance_m=round(total_distance_m, 2),
            travel_time_min=travel_time_min,
            waypoints_count=len(route_nodes),
            severed_edges_count=len(self.severed_edges),
            flood_polygon=self.flood_polygon,
        )

        return {
            "status": "success",
            "waypoints": route_nodes,
            "route_coords": route_coords,
            "distance_meters": round(total_distance_m, 2),
            "travel_time_minutes": travel_time_min,
            "severed_edges_count": len(self.severed_edges),
            "geojson": geojson_data,
        }

    def to_geojson_feature_collection(
        self,
        route_coords: List[Tuple[float, float]],
        origin_coords: Tuple[float, float],
        dest_coords: Tuple[float, float],
        distance_m: float,
        travel_time_min: float,
        waypoints_count: int,
        severed_edges_count: int,
        flood_polygon: Optional[Polygon],
    ) -> Dict[str, Any]:
        """
        Builds standard RFC 7946 GeoJSON FeatureCollection representation.
        Note: GeoJSON coordinates are formatted as [longitude, latitude].
        """
        features = []

        # 1. Safe Evacuation Route Feature (LineString)
        geojson_route_coords = [[lon, lat] for lat, lon in route_coords]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": geojson_route_coords,
            },
            "properties": {
                "role": "safe_evacuation_corridor",
                "stroke": "#0066FF",
                "stroke_width": 5,
                "distance_meters": distance_m,
                "estimated_time_minutes": travel_time_min,
                "waypoints_count": waypoints_count,
                "routing_algorithm": "dijkstra_safe_graph",
            },
        })

        # 2. Origin Pickup Marker (Point)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [origin_coords[1], origin_coords[0]],
            },
            "properties": {
                "role": "evacuation_origin",
                "name": "Evacuation Pickup Location",
                "marker_color": "green",
            },
        })

        # 3. Destination Relief Shelter Marker (Point)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [dest_coords[1], dest_coords[0]],
            },
            "properties": {
                "role": "evacuation_destination",
                "name": "Emergency Relief Shelter",
                "marker_color": "red",
            },
        })

        # 4. Active Flood Inundation Hazard Polygon (Polygon)
        if flood_polygon is not None:
            features.append({
                "type": "Feature",
                "geometry": mapping(flood_polygon),
                "properties": {
                    "role": "flood_hazard_zone",
                    "hazard_level": "submerged_impassable",
                    "fill": "#ff3333",
                    "fill_opacity": 0.45,
                    "severed_edges_count": severed_edges_count,
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "region": "Mandi, Himachal Pradesh",
                "system": "FloodLens AI",
                "status": "active_rerouting",
            },
        }
