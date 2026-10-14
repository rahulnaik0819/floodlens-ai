# 🛰️ FloodLens AI: Multimodal Geospatial Emergency Evacuation System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 11 Passed](https://img.shields.io/badge/pytest-11%20passed-brightgreen.svg)](https://docs.pytest.org)

An end-to-end multimodal disaster intelligence platform combining dual-polarization Synthetic Aperture Radar (SAR) remote sensing segmentation with dynamic Dijkstra graph-severing algorithms. Designed specifically for cloud-obscured mountainous river basins (centered on Mandi, Himachal Pradesh), FloodLens AI prunes submerged transport corridors in real time to coordinate safe evacuation paths.

---

## 📌 Executive Summary

During catastrophic flood events, humanitarian response operations face two fatal failure modes:
1. **Zero Optical Visibility:** Severe meteorological downpours, cloud cover, and smoke blind multispectral/RGB satellite sensors.
2. **Static Network Disconnect:** Standard navigation engines route over static road graphs, systematically directing emergency personnel into submerged or impassable water corridors.

FloodLens AI resolves this bottleneck through a coupled remote sensing and spatial graph pipeline:

```text
[ Sentinel-1 SAR Dual-Pol (VV/VH) ]
               │
               ▼
   [ Specular Backscatter Physics ]  -->  (~ -10 dB Attenuation across open water)
               │
               ▼
    [ U-Net + ResNet-34 Engine ]     -->  (Trained via Binary Dice Loss)
               │
               ▼
     [ Vector Hazard Polygon ]       -->  (Raster-to-Shapely Polygon Extraction)
               │
               ▼
     [ OSMnx Dynamic Pruning ]       -->  (G_safe = G_raw \ Intersecting_Edges)
               │
               ▼
    [ Dijkstra Safe Routing ]        -->  (Shortest dry-path execution)
               │
               ▼
  [ FastAPI Command Dashboard ]      -->  (RFC 7946 GeoJSON + Leaflet HUD)
```
---

## 📐 System Architecture & Design

### 1. Data Ingestion & Radar Physics (`src/dataset.py`)
Active radar microwave sensors emit pulses and measure returned backscatter ($\sigma^0$). Because calm surface water creates specular reflection, incoming microwave pulses bounce away from the satellite receiver, producing a steep attenuation drop:
$$\Delta \sigma^0 \approx -10 \text{ dB}$$

The synthetic SAR pipeline simulates dual-band imagery:
* **VV Channel:** High sensitivity to surface capillary wave deformation.
* **VH Channel:** Cross-polarization measuring volume scattering changes across vegetation-to-water transition boundaries.

### 2. Deep Inundation Segmentation (`src/model.py`)
To isolate narrow river networks with minimal false negatives, FloodLens AI implements a U-Net architecture built upon a ResNet-34 backbone:
* **Encoder:** ResNet-34 pre-trained features extracting multiscale structural spatial semantics.
* **Decoder:** Transpose convolution blocks paired with skip connections to preserve localized edge boundaries.
* **Loss Function (Soft Dice Loss):** Mitigates extreme spatial class imbalance where flooded corridors occupy $< 8\%$ of the satellite tile:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} p_i y_i + \epsilon}{\sum_{i} p_i + \sum_{i} y_i + \epsilon}$$

### 3. Spatial Graph Pruning & Safe Dijkstra Routing (`src/router.py`)
* Extracts the OpenStreetMap drivable multigraph $G = (V, E)$ centered on Mandi, HP via OSMnx.
* Performs 2D geometric intersection tests using Shapely between the raster-derived flood polygon $P_{\text{flood}}$ and each road segment edge $e \in E$:
$$E_{\text{submerged}} = \{ e \in E \mid \text{LineString}(e) \cap P_{\text{flood}} \neq \emptyset \}$$
* Constructs the safe subgraph $G_{\text{safe}} = (V, E \setminus E_{\text{submerged}})$.
* Executes Dijkstra's algorithm over $G_{\text{safe}}$ using road length weights to ensure zero traversal over submerged nodes.

### 4. High-Throughput REST Microservice (`app.py`)
* Exposes `/health` telemetry check (status, device, model state).
* Exposes `POST /api/v1/reroute` accepting origin/destination coordinates and returning RFC 7946 GeoJSON lines along with transit metrics.
* Mounts a responsive command dashboard directly at root `/` via FastAPI `StaticFiles`.

---

## 🛠️ Tech Stack & Dependencies

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Microservice Backend** | FastAPI, Uvicorn, Pydantic v2 | High-concurrency async REST API & static serving |
| **Deep Learning & Vision**| PyTorch, SMP, NumPy, Rasterio | SAR simulation, U-Net ResNet-34 inference |
| **Geospatial Graph Engine** | OSMnx, NetworkX, Shapely | Vector road graph extraction & dynamic Dijkstra pruning |
| **Operations Frontend** | Leaflet.js, Tailwind CSS, FontAwesome | Dark glassmorphism HUD, real-time map canvas |
| **Quality & Assurance** | PyTest, HTTPX | Unit assertions, shape verification, API contract tests |

---

```text
floodlens-ai/
├── app.py                      # FastAPI application entrypoint & static mounting
├── requirements.txt            # Production dependencies
├── .gitignore                  # Git tracking exclusion rules
├── README.md                   # System documentation & reproduction manual
├── src/
│   ├── __init__.py             # Core package initialization
│   ├── dataset.py              # Synthetic dual-pol SAR simulation pipeline
│   ├── model.py                # U-Net ResNet-34 segmentation network
│   ├── router.py               # Graph pruning engine & safe Dijkstra router
│   ├── evaluate.py             # Metric evaluation script (IoU, Dice, Precision)
│   └── visualizer.py           # Folium/Leaflet layer generator
├── static/
│   └── index.html              # Emergency command dashboard UI
└── tests/
    ├── __init__.py             # Test package initialization
    ├── test_api.py             # FastAPI health & reroute endpoint validation
    └── test_pipeline.py        # Tensor shape, Dice Loss, and graph-severing tests
```


---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or 3.11
* Git

### Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rahulnaik0819/floodlens-ai.git](https://github.com/rahulnaik0819/floodlens-ai.git)
   cd floodlens-ai
Create and activate a virtual environment:

Windows (PowerShell):

PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
Linux / macOS:

Bash
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:

Bash
pip install --upgrade pip
pip install -r requirements.txt
🧪 Testing & Verification
Run the full automated test suite verifying dataset tensor dimensions, U-Net ResNet-34 forward passes, Dijkstra edge-pruning logic, and API contracts:

## 🧪 Testing & Verification

Run the full automated test suite verifying dataset tensor dimensions, U-Net ResNet-34 forward passes, Dijkstra edge-pruning logic, and API contracts:

```bash
pytest tests/ -v
```

Expected output:

```text
tests/test_api.py::test_health_endpoint PASSED                           [  9%]
tests/test_api.py::test_root_dashboard_serves_html PASSED                [ 18%]
tests/test_api.py::test_reroute_endpoint_valid_path PASSED               [ 27%]
tests/test_pipeline.py::test_dataset_shapes PASSED                       [ 36%]
tests/test_pipeline.py::test_model_forward_pass PASSED                   [ 45%]
tests/test_pipeline.py::test_dice_loss_computation PASSED                [ 54%]
tests/test_pipeline.py::test_graph_severing_removes_edges PASSED         [ 63%]
tests/test_pipeline.py::test_dijkstra_finds_valid_detour PASSED          [ 72%]
tests/test_pipeline.py::test_out_of_bounds_coordinate_handling PASSED     [ 81%]
tests/test_pipeline.py::test_geojson_serialization_structure PASSED      [ 90%]
tests/test_pipeline.py::test_edge_case_impassable_zone PASSED             [100%]

============================== 11 passed in 4.12s ==============================
```

To run offline metric evaluation:

```bash
python -m src.evaluate
```

Bash
python -m src.evaluate
🖥️ Running the Application
Start the FastAPI application with auto-reloading:

Bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
Once the server initializes, access the services:

Interactive Command Dashboard: http://127.0.0.1:8000/

Swagger Interactive API Documentation: http://127.0.0.1:8000/docs

ReDoc Technical Schema: http://127.0.0.1:8000/redoc

Health Check Endpoint: http://127.0.0.1:8000/health

## 📡 API Reference

### `POST /api/v1/reroute`
Calculates an optimal safe evacuation route circumventing active flood zones.

#### Request Body
```json
{
  "origin_lat": 31.7180,
  "origin_lon": 76.9280,
  "dest_lat": 31.7000,
  "dest_lon": 76.9450
}
```

#### Response (`200 OK`)
```json
{
  "status": "success",
  "total_distance_meters": 5042.15,
  "waypoint_count": 52,
  "hazard_avoidance": true,
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [76.9280, 31.7180],
            [76.9312, 31.7142],
            [76.9450, 31.7000]
          ]
        },
        "properties": {
          "stroke": "#06b6d4",
          "stroke-width": 5
        }
      }
    ]
  }
}
```

---

## 🤖 AI Assistance & Transparency Disclosure

In adherence to competition guidelines regarding ethical AI usage, AI coding assistants were leveraged as architectural pair-programmers during the development of FloodLens AI.

### Tools Utilized
* **Google Antigravity / Gemini / ChatGPT:** Architectural outlining, synthetic dataset scaffolding, rapid frontend boilerplate assembly, and PyTest test generation.

### Module Breakdown
* **Assisted Workflows:** Synthetic Sentinel-1 radar physics boilerplate (`src/dataset.py`), Leaflet UI shell design (`static/index.html`), and initial PyTest assertion scaffolding (`tests/`).
* **Human Verification & System Ownership:** All core domain logic—including radar attenuation physics modeling, U-Net ResNet-34 hyperparameter selection, Shapely polygon intersection mathematics, OSMnx road graph pruning, and Mandi coordinate configurations—was audited, verified, and integrated by the author.
