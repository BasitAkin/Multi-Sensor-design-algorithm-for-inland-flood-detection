# Where to Watch the Water: Multi-Sensor Network Design Optimization for Inland Flood Detection

**Author:** Basit A. Akinade
**Affiliation:** Water Intelligence and Geospatial Sensing (WINGS) Lab
**Date:** January 2025

---

## Overview

This repository contains the complete reproducible codebase for the paper *"Where to Watch the Water: Multi-Sensor Network Design Optimization for Inland Flood Detection."* The framework implements a decision-focused, submodular optimization algorithm for optimal placement of heterogeneous flood detection sensors (water level, discharge, and camera) across HUC10 watersheds, followed by validation against existing USGS infrastructure and the NOAA National Water Model.

The analysis pipeline consists of three sequential stages:

1. **Multi-Sensor Network Optimization** - Greedy submodular optimization for sensor placement across four operational scenarios.
2. **USGS Expansion Analysis** - Integration analysis quantifying how the proposed network complements existing USGS gaging stations.
3. **NWM Validation** - Validation of selected sensor locations against National Water Model retrospective streamflow metrics.

---

## Repository Structure

```
.
|-- README.md
|-- requirements.txt
|-- data/
|   |-- master_dataset.csv                # Candidate sensor locations with attributes
|   |-- studyarea_huc10s.shp              # HUC10 watershed boundaries (+ .dbf, .shx, .prj, etc.)
|   |-- USGSgages_in_ROI_5states.csv      # USGS gage locations within the study area (required for Step 2)
|-- multisensor_optimization.py           # Step 1: Multi-sensor placement algorithm
|-- usgs_expansion_analysis.py            # Step 2: USGS integration analysis
|-- nwm_validation.py                     # Step 3: National Water Model validation
```

**Output directories** (generated automatically when each script runs):

- `Multisensor_Algorithms_outputs_v44S/` - Outputs from Step 1 (sensor locations, maps, tables)
- `usgs_IntAnaly_outputs_v31HYBRID/` - Outputs from Step 2 (classification, synergy analysis)
- `nwm_validation_v22S_outputs/` - Outputs from Step 3 (flow metrics, statistical comparisons)

---

## Data Sources

### Required Input Data

All input data should be placed in the `data/` directory before running the scripts.

#### 1. Master Dataset (`master_dataset.csv`)

The master candidate dataset contains potential sensor locations with attributes including geographic coordinates, flood risk scores, NHDPlus stream attributes (stream order, flow accumulation, upstream length), and HUC10 identifiers. This dataset is project-specific and was compiled from the sources listed below.

#### 2. HUC10 Watershed Boundaries (`studyarea_huc10s.shp`)

USGS Watershed Boundary Dataset (WBD) polygons for the study area at the HUC10 level.

- **Source:** USGS Watershed Boundary Dataset (WBD)
- **Download:** [https://www.usgs.gov/national-hydrography/watershed-boundary-dataset](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset)
- **Alternative:** [USGS National Map Data Download](https://apps.nationalmap.gov/downloader/)
- **Format:** Shapefile (.shp, .dbf, .shx, .prj)

#### 3. USGS Gage Locations (`USGSgages_in_ROI_5states.csv`)

Active and historical USGS streamflow gaging station locations within the region of interest.

- **Source:** USGS National Water Information System (NWIS)
- **Site Inventory:** [https://waterdata.usgs.gov/nwis/inventory](https://waterdata.usgs.gov/nwis/inventory)
- **NWIS Mapper (interactive):** [https://maps.waterdata.usgs.gov/mapper/](https://maps.waterdata.usgs.gov/mapper/)
- **GageLoc (indexed to NHDPlus v2.1):** [https://www.sciencebase.gov/catalog/item/577445bee4b07657d1a991b6](https://www.sciencebase.gov/catalog/item/577445bee4b07657d1a991b6)
- **Web Services API:** [https://waterservices.usgs.gov/](https://waterservices.usgs.gov/)

#### 4. National Water Model Retrospective Data (accessed remotely by Step 3)

The NWM validation script accesses NWM v3.0 retrospective streamflow data directly from AWS S3 - no manual download is required. The script reads the Zarr-formatted channel routing output (CHRTOUT) covering February 1979 through January 2023.

- **S3 Zarr Path:** `s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr`
- **Registry of Open Data on AWS:** [https://registry.opendata.aws/nwm-archive/](https://registry.opendata.aws/nwm-archive/)
- **S3 Browser (NetCDF format):** [https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/index.html](https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/index.html)
- **NOAA Big Data Program Docs:** [https://github.com/NOAA-Big-Data-Program/nodd-data-docs/blob/main/nwm/README.md](https://github.com/NOAA-Big-Data-Program/nodd-data-docs/blob/main/nwm/README.md)
- **CUAHSI HydroShare Notebooks:** [https://www.hydroshare.org/resource/6ca065138d764339baf3514ba2f2d72f/](https://www.hydroshare.org/resource/6ca065138d764339baf3514ba2f2d72f/)

> **Note:** NWM data access requires an internet connection but does *not* require an AWS account (the bucket is publicly accessible via `--no-sign-request`).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Multi-SensorPaper.git
cd Multi-SensorPaper

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.9
- NumPy, Pandas, Matplotlib, SciPy
- GeoPandas (for shapefile processing in Step 1)
- xarray, fsspec, s3fs, zarr (for NWM cloud data access in Step 3)

---

## Usage

The three scripts must be run **sequentially** from the repository root directory, as each step depends on outputs from the previous one.

### Step 1: Multi-Sensor Network Optimization

```bash
python multisensor_optimization.py
```

Runs the basin-by-basin submodular optimization across four operational scenarios (A-D). Outputs sensor locations, performance tables, and diagnostic figures to `Multisensor_Algorithms_outputs_v44S/`.

### Step 2: USGS Expansion Analysis

```bash
python usgs_expansion_analysis.py
```

Analyzes how the proposed sensor network integrates with existing USGS gaging stations. Classifies sensors as Sentinel, Cascade Sentinel, Gap-Filler, or Validator. Outputs to `usgs_IntAnaly_outputs_v31HYBRID/`.

### Step 3: NWM Validation

```bash
python nwm_validation.py
```

Validates selected sensor locations against NWM retrospective streamflow metrics (AMA, Q95, Q99, AMAX, Flashiness). Compares sensor locations to 10,000 random reference points. Outputs to `nwm_validation_v22S_outputs/`.

> **Note:** Step 3 downloads NWM data from AWS S3. Depending on network speed, this may take significant time on first execution.

---

## Operational Scenarios

| Scenario | Description | False Positive Penalty (lambda) | Cost Fraction (tau) |
|:--------:|:------------|:-:|:-:|
| **A** | Maximum Coverage | 0.10 | 0.02 |
| **B** | Balanced (Baseline) | 0.15 | 0.05 |
| **C** | Precision-Focused | 0.25 | 0.05 |
| **D** | Resource-Constrained | 0.15 | 0.10 |

---

## Objective Function

The optimization maximizes a penalized detection score:

$$J(S) = \text{TP}_{\text{rate}} - \lambda \cdot \text{FP}_{\text{rate}} - \tau \cdot |S|$$

Where TP_rate is the true positive (flood detection) rate, FP_rate is the false positive (false alarm) rate, lambda is the false positive penalty weight, tau is the sensor deployment cost fraction, and |S| is the network size.

---

## Citation

If you use this code in your research, please cite:

```
Akinade, B. A. (2025). Where to Watch the Water: Multi-Sensor Network Design
Optimization for Inland Flood Detection. Water Intelligence and Geospatial
Sensing (WINGS) Lab.
```

---

## Contact

For questions or collaboration inquiries, please contact the WINGS Lab.

---

## License

This project is provided for academic and research purposes.
