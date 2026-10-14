"""
Integration tests for FloodLens AI FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test GET / now serves the static HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>FloodLens AI" in response.text


def test_static_dashboard_html(client):
    """Test that dashboard HTML contains required FloodLens title tag."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>FloodLens AI" in response.text


def test_health_check_endpoint(client):
    """Test GET /health returns operational status and graph nodes."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_ready"] is True
    assert data["road_network_nodes"] > 0
    assert data["road_network_edges"] > 0
    assert "Mandi" in data["target_region"]


def test_reroute_endpoint_valid_coordinates(client):
    """Test POST /api/v1/reroute calculates safe Dijkstra path and returns GeoJSON."""
    payload = {
        "origin_lat": 31.7180,
        "origin_lon": 76.9200,
        "dest_lat": 31.7000,
        "dest_lon": 76.9450,
    }
    response = client.post("/api/v1/reroute", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["distance_meters"] > 0
    assert data["estimated_time_minutes"] > 0
    assert data["waypoints_count"] >= 2
    assert data["severed_edges_count"] >= 0

    geojson = data["geojson"]
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 3

    # Check route feature properties
    route_feature = geojson["features"][0]
    assert route_feature["geometry"]["type"] == "LineString"
    assert route_feature["properties"]["stroke"] == "#0066FF"
    assert route_feature["properties"]["routing_algorithm"] == "dijkstra_safe_graph"


def test_reroute_validation_error(client):
    """Test POST /api/v1/reroute with invalid latitude values (>90.0)."""
    payload = {
        "origin_lat": 999.0,
        "origin_lon": 76.9200,
        "dest_lat": 31.7000,
        "dest_lon": 76.9450,
    }
    response = client.post("/api/v1/reroute", json=payload)
    assert response.status_code == 422


def test_visualize_endpoint_html(client):
    """Test POST /api/v1/visualize returns interactive Folium Leaflet HTML."""
    payload = {
        "origin_lat": 31.7180,
        "origin_lon": 76.9200,
        "dest_lat": 31.7000,
        "dest_lon": 76.9450,
    }
    response = client.post("/api/v1/visualize", json=payload)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html_text = response.text
    assert "leaflet" in html_text.lower() or "folium" in html_text.lower()
    assert "esri" in html_text.lower()
