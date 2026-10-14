"""
Interactive Geospatial Visualization Module for FloodLens AI.
Generates interactive Leaflet/Folium disaster management maps with dual
Esri Street & Satellite layers, flood hazard polygons, and safe rescue paths.
"""

from typing import List, Tuple, Optional, Union
import folium
from shapely.geometry import Polygon, mapping


ESRI_STREET_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
)
ESRI_IMAGERY_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


class RescueMapVisualizer:
    """
    Renders high-definition disaster relief operations maps using Folium.
    """

    def __init__(
        self,
        center_lat: float = 31.7088,
        center_lon: float = 76.9320,
        zoom_start: int = 13,
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom_start = zoom_start

    def generate_map(
        self,
        route_coords: Optional[List[Tuple[float, float]]] = None,
        flood_polygon: Optional[Polygon] = None,
        start_label: str = "Evacuation Pickup",
        end_label: str = "Emergency Relief Shelter",
    ) -> folium.Map:
        """
        Builds and returns the configured Folium Map object.
        """
        # 1. Base Map: Esri World Street Map (reliable, no API key required)
        map_center = [self.center_lat, self.center_lon]
        if route_coords and len(route_coords) > 0:
            map_center = [route_coords[0][0], route_coords[0][1]]

        m = folium.Map(
            location=map_center,
            zoom_start=self.zoom_start,
            tiles=ESRI_STREET_TILES,
            attr="Esri",
            control_scale=True,
        )

        # 2. Layer: Esri World Imagery (High-res Satellite View)
        folium.TileLayer(
            tiles=ESRI_IMAGERY_TILES,
            attr="Esri World Imagery",
            name="Satellite View",
        ).add_to(m)

        # 3. Layer: Submerged Flood Inundation Zone
        if flood_polygon is not None:
            flood_geo = mapping(flood_polygon)
            folium.GeoJson(
                flood_geo,
                name="Submerged Flood Zone",
                style_function=lambda x: {
                    "fillColor": "#ff3333",
                    "color": "#ff0000",
                    "weight": 2,
                    "fillOpacity": 0.45,
                },
                tooltip="Active Flood Inundation Zone (Impassable)",
            ).add_to(m)

        # 4. Layer: Dynamic Safe Evacuation Path
        if route_coords and len(route_coords) > 1:
            folium.PolyLine(
                locations=route_coords,
                color="#0066FF",
                weight=5,
                opacity=0.9,
                tooltip="Dynamic Safe Rescue Path",
            ).add_to(m)

            # 5. Start and End Waypoint Markers
            folium.Marker(
                location=route_coords[0],
                popup=f"<b>Start:</b> {start_label}",
                tooltip=f"Start: {start_label}",
                icon=folium.Icon(color="green", icon="play", prefix="glyphicon"),
            ).add_to(m)

            folium.Marker(
                location=route_coords[-1],
                popup=f"<b>End:</b> {end_label}",
                tooltip=f"End: {end_label}",
                icon=folium.Icon(color="red", icon="home", prefix="glyphicon"),
            ).add_to(m)

        # 6. Interactive Layer Controls
        folium.LayerControl(position="topright").add_to(m)

        return m

    def save_map_html(
        self,
        output_path: str = "floodlens_rescue_map.html",
        route_coords: Optional[List[Tuple[float, float]]] = None,
        flood_polygon: Optional[Polygon] = None,
    ) -> str:
        """
        Saves the interactive disaster response map to a standalone HTML file.
        """
        m = self.generate_map(route_coords=route_coords, flood_polygon=flood_polygon)
        m.save(output_path)
        return output_path

    def get_html_string(
        self,
        route_coords: Optional[List[Tuple[float, float]]] = None,
        flood_polygon: Optional[Polygon] = None,
    ) -> str:
        """
        Returns raw HTML string representation of the map.
        """
        m = self.generate_map(route_coords=route_coords, flood_polygon=flood_polygon)
        return m.get_root().render()
