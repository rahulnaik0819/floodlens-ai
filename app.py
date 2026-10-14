"""
FloodLens AI - Production FastAPI Service.
Provides RESTful geospatial endpoints for multimodal flood hazard detection,
submerged road edge severing, and dynamic Dijkstra evacuation rerouting.
"""

import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from shapely.geometry import shape, Polygon

from src.router import RoadNetworkRouter, DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON
from src.visualizer import RescueMapVisualizer
from src.model import SARFloodSegmenter, get_default_device


# Global singletons
router_instance: Optional[RoadNetworkRouter] = None
segmenter_instance: Optional[SARFloodSegmenter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize model segmenter and road network graph on application startup.
    """
    global router_instance, segmenter_instance
    device = get_default_device()
    segmenter_instance = SARFloodSegmenter(device=device)

    # Initialize router (loads local cache or downloads/generates grid)
    router_instance = RoadNetworkRouter(
        center_lat=DEFAULT_CENTER_LAT,
        center_lon=DEFAULT_CENTER_LON,
        radius_m=8000,   # matches DEFAULT_RADIUS_METERS — expands coverage to all plottable corridors
        auto_load=True,
    )
    yield


app = FastAPI(
    title="FloodLens AI - Geospatial Emergency Rerouting API",
    description=(
        "Production API for real-time flood inundation mapping from synthetic "
        "dual-polarization SAR satellite radar and dynamic evacuation rerouting around submerged road networks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients / GIS dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------
class RerouteRequest(BaseModel):
    origin_lat: float = Field(
        ..., ge=-90.0, le=90.0, description="Latitude of the evacuation pickup origin."
    )
    origin_lon: float = Field(
        ..., ge=-180.0, le=180.0, description="Longitude of the evacuation pickup origin."
    )
    dest_lat: float = Field(
        ..., ge=-90.0, le=90.0, description="Latitude of the emergency shelter destination."
    )
    dest_lon: float = Field(
        ..., ge=-180.0, le=180.0, description="Longitude of the emergency shelter destination."
    )
    custom_flood_polygon: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional GeoJSON Polygon geometry representing flood inundation zone.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "origin_lat": 31.7180,
                "origin_lon": 76.9200,
                "dest_lat": 31.7000,
                "dest_lon": 76.9450,
            }
        }
    }


class RerouteResponse(BaseModel):
    status: str = Field(..., example="success")
    distance_meters: float = Field(..., example=4120.5)
    estimated_time_minutes: float = Field(..., example=8.2)
    waypoints_count: int = Field(..., example=14)
    severed_edges_count: int = Field(..., example=297)
    geojson: Dict[str, Any] = Field(
        ..., description="Standard GeoJSON FeatureCollection containing route and hazard features."
    )


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    target_region: str = Field(..., example="Mandi, Himachal Pradesh (Beas River Valley)")
    road_network_nodes: int = Field(..., example=413)
    road_network_edges: int = Field(..., example=980)
    device: str = Field(..., example="cpu")
    model_ready: bool = Field(..., example=True)


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """
    Returns system health, device architecture, and road graph connectivity stats.
    """
    global router_instance, segmenter_instance

    if router_instance is None or router_instance.G is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Road network graph is not yet initialized.",
        )

    nodes_count = len(router_instance.G.nodes)
    edges_count = len(router_instance.G.edges)
    device_str = str(segmenter_instance.device) if segmenter_instance else "unknown"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        target_region="Mandi, Himachal Pradesh (Beas River Valley)",
        road_network_nodes=nodes_count,
        road_network_edges=edges_count,
        device=device_str,
        model_ready=segmenter_instance is not None,
    )


@app.post("/api/v1/reroute", response_model=RerouteResponse, tags=["Routing"])
def reroute_evacuation(payload: RerouteRequest):
    """
    Dynamic Dijkstra Safe Evacuation Recalculation:
      - Snaps origin and destination GPS points to nearest passable road network nodes.
      - Intersects road edges against active flood inundation zones and severs submerged segments.
      - Computes the shortest surviving path using Dijkstra's algorithm.
      - Returns a valid GeoJSON FeatureCollection for direct GIS / frontend map rendering.
    """
    global router_instance

    if router_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router engine is unavailable.",
        )

    flood_poly = None
    if payload.custom_flood_polygon:
        try:
            flood_poly = shape(payload.custom_flood_polygon)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid GeoJSON geometry for flood polygon: {str(e)}",
            )

    try:
        route_result = router_instance.find_evacuation_route(
            origin_lat=payload.origin_lat,
            origin_lon=payload.origin_lon,
            dest_lat=payload.dest_lat,
            dest_lon=payload.dest_lon,
            flood_polygon=flood_poly,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Safe evacuation path could not be computed: {str(e)}",
        )

    return RerouteResponse(
        status="success",
        distance_meters=route_result["distance_meters"],
        estimated_time_minutes=route_result["travel_time_minutes"],
        waypoints_count=len(route_result["waypoints"]),
        severed_edges_count=route_result["severed_edges_count"],
        geojson=route_result["geojson"],
    )


@app.post("/api/v1/visualize", response_class=HTMLResponse, tags=["Visualization"])
def visualize_rescue_map(payload: RerouteRequest):
    """
    Generates an interactive Folium Leaflet map with dual Esri basemaps,
    submerged flood hazard polygon overlay, and highlighted evacuation route.
    """
    global router_instance

    if router_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router engine is unavailable.",
        )

    flood_poly = None
    if payload.custom_flood_polygon:
        flood_poly = shape(payload.custom_flood_polygon)

    route_result = router_instance.find_evacuation_route(
        origin_lat=payload.origin_lat,
        origin_lon=payload.origin_lon,
        dest_lat=payload.dest_lat,
        dest_lon=payload.dest_lon,
        flood_polygon=flood_poly,
    )

    visualizer = RescueMapVisualizer(
        center_lat=payload.origin_lat, center_lon=payload.origin_lon
    )
    html_content = visualizer.get_html_string(
        route_coords=route_result["route_coords"],
        flood_polygon=router_instance.flood_polygon,
    )

    return HTMLResponse(content=html_content)


# Mount static dashboard AFTER all API routes to avoid path conflicts
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
_os.makedirs(_static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
