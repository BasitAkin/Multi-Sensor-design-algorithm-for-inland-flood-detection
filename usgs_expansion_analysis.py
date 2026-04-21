"""
USGS INTEGRATION ANALYSIS MODULE
===============================================
Post-algorithm INTERPRETIVE LAYER that analyzes the relationship between
the proposed flood sensor network and existing USGS infrastructure.

The topology columns (Hydroseq, Levelpathi, dnhydroseq) contain placeholder
values in this dataset. Instead, we use a MULTI-INDICATOR approach with
valid NHDPlus variables:

UPSTREAM INDICATORS (all must be valid and agree):
  1. Elevation: Sensor elevation > USGS elevation (primary)
  2. Stream Order: Sensor order <= USGS order (headwater -> mainstem)
  3. Flow Accumulation: Sensor flowacc < USGS flowacc (less drainage area)
  4. Upstream Length: Sensor upstream_km < USGS upstream_km (less network above)

CONFIDENCE LEVELS:
  - HIGH: 3-4 indicators agree + same HUC8
  - MEDIUM: 2 indicators agree + same HUC8
  - LOW: 1 indicator + spatial proximity

Framework: "Flood Warning Cascade Architecture"
- Layer 1: USGS Flood Coverage Audit
- Layer 2: Network Synergy Quantification
- Layer 3: Multi-Indicator Upstream Analysis (IMPROVED)
- Sensor Classification: Sentinel / Cascade Sentinel / Gap-Filler / Validator

Author: WINGS Lab
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from scipy.spatial import cKDTree
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
MULTISENSOR_OUTPUT_DIR = "Multisensor_Algorithms_outputs_v44S"
SELECTED_SITES_ALL_FILE = os.path.join(MULTISENSOR_OUTPUT_DIR, "selected_sites_all_scenarios.csv")
BASIN_SUMMARY_ALL_FILE = os.path.join(MULTISENSOR_OUTPUT_DIR, "basin_summary_all_scenarios.csv")

MASTER_DATA_FILE = os.path.join("data", "master_dataset.csv")
USGS_STATIONS_FILE = os.path.join("data", "USGSgages_in_ROI_5states.csv")
HUC10_SHAPEFILE = os.path.join("data", "studyarea_huc10s.shp")

# Output directory
OUT_DIR = os.path.join(os.getcwd(), "usgs_IntAnaly_outputs_v31HYBRID")
os.makedirs(OUT_DIR, exist_ok=True)

# Output files
OUT_USGS_SNAPPED = os.path.join(OUT_DIR, "usgs_stations_snapped_attributes.csv")
OUT_CLASSIFICATION_ALL = os.path.join(OUT_DIR, "sensor_classification_all_scenarios.csv")
OUT_SYNERGY_ALL = os.path.join(OUT_DIR, "network_synergy_all_scenarios.csv")
OUT_SCENARIO_COMPARISON = os.path.join(OUT_DIR, "scenario_usgs_comparison.csv")
OUT_LEAD_TIME_ALL = os.path.join(OUT_DIR, "cascade_sentinel_lead_times_all.csv")
OUT_UPSTREAM_DETAILS = os.path.join(OUT_DIR, "upstream_classification_details.csv")
OUT_INDICATOR_SUMMARY = os.path.join(OUT_DIR, "upstream_indicator_agreement.csv")

# Figures
OUT_FIG_COMPARISON = os.path.join(OUT_DIR, "usgs_scenario_comparison.png")
OUT_FIG_MAPS_ALL = os.path.join(OUT_DIR, "usgs_integration_maps_all_scenarios.png")
OUT_FIG_LEAD_TIMES = os.path.join(OUT_DIR, "cascade_sentinel_lead_times_comparison.png")
OUT_FIG_INDICATORS = os.path.join(OUT_DIR, "upstream_indicator_analysis.png")

# Analysis parameters
DEFAULT_DETECTION_RADIUS_KM = 20.0
SNAP_TOLERANCE_KM = 2.0

# Upstream classification thresholds
ELEVATION_THRESHOLD_M = 10.0      # Sensor must be at least 10m higher
MAX_SEARCH_DISTANCE_KM = 50.0     # Only consider USGS within 50 km
SAME_HUC8_REQUIRED = True         # Require same HUC8 for high confidence

# Scenario definitions
SCENARIOS = {
    'A: Maximum Coverage': {'short_name': 'MaxCoverage', 'color': '#2ecc71', 'marker': 'o'},
    'B: Balanced': {'short_name': 'Balanced', 'color': '#3498db', 'marker': 's'},
    'C: Precision-Focused': {'short_name': 'Precision', 'color': '#9b59b6', 'marker': '^'},
    'D: Resource-Constrained': {'short_name': 'Resource', 'color': '#e74c3c', 'marker': 'D'}
}

# Stream velocity for lead time
STREAM_VELOCITY_BY_ORDER = {1: 0.8, 2: 1.0, 3: 1.3, 4: 1.6, 5: 2.0, 6: 2.5, 7: 3.0, 8: 3.5}
DEFAULT_VELOCITY_MS = 1.5
FLOOD_WAVE_CELERITY_FACTOR = 1.5
SINUOSITY_FACTOR = 1.3

print("=" * 80)
print("USGS INTEGRATION ANALYSIS MODULE (v3.1-HYBRID)")
print("  -> Multi-Indicator Upstream Classification")
print("  -> Using: Elevation + Stream Order + Flow Accumulation + Upstream Length")
print("  -> Avoids corrupted topology columns (Hydroseq, Levelpathi)")
print("  -> Multi-Scenario Analysis (4 scenarios from v4.4-S)")
print("=" * 80)


# ================== UTILITY FUNCTIONS ==================

def haversine_km(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance between two points."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def minmax(x):
    """Min-max normalize array."""
    x = np.asarray(x, float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    rng = xmax - xmin
    if not np.isfinite(rng) or rng <= 0:
        return np.zeros_like(x)
    return np.clip((x - xmin) / (rng + 1e-12), 0, 1)


def compute_coverage(station_coords_km, all_points_coords_km, risk_weights, detection_radius_km):
    """Compute weighted flood risk coverage by a set of stations."""
    if all_points_coords_km is None or risk_weights is None:
        return 0, 0, 0, 0, None
    
    n_points = len(all_points_coords_km)
    n_stations = len(station_coords_km)
    
    if n_stations == 0:
        return 0, float(risk_weights.sum()), 0, 0, np.zeros(n_points, dtype=bool)
    
    tree_points = cKDTree(all_points_coords_km)
    covered_mask = np.zeros(n_points, dtype=bool)
    
    for i in range(n_stations):
        neighbors = tree_points.query_ball_point(station_coords_km[i], r=detection_radius_km)
        covered_mask[neighbors] = True
    
    covered_risk = float(risk_weights[covered_mask].sum())
    total_risk = float(risk_weights.sum())
    coverage_frac = covered_risk / max(total_risk, 1e-9)
    
    return covered_risk, total_risk, coverage_frac, int(covered_mask.sum()), covered_mask


def get_velocity_by_stream_order(stream_order):
    """Get stream velocity (m/s) based on stream order."""
    if pd.isna(stream_order) or not np.isfinite(stream_order):
        return DEFAULT_VELOCITY_MS
    order = int(min(max(stream_order, 1), 8))
    return STREAM_VELOCITY_BY_ORDER.get(order, DEFAULT_VELOCITY_MS)


def estimate_lead_time_hours(distance_km, stream_order=None):
    """Estimate flood wave travel time in HOURS using Euclidean distance with sinuosity."""
    if distance_km <= 0:
        return 0.0, 0.0
    
    v = get_velocity_by_stream_order(stream_order)
    celerity_kmh = v * FLOOD_WAVE_CELERITY_FACTOR * 3.6
    network_distance = distance_km * SINUOSITY_FACTOR
    lead_time_hours = network_distance / max(celerity_kmh, 0.1)
    
    return float(lead_time_hours), float(v)


def safe_float(val):
    """Safely convert value to float."""
    try:
        f = float(val)
        if np.isfinite(f):
            return f
        return None
    except:
        return None


def normalize_huc(huc_value, digits=8):
    """Normalize HUC code to specified digits."""
    if pd.isna(huc_value):
        return None
    huc_str = str(huc_value).strip()
    if '.' in huc_str:
        huc_str = huc_str.split('.')[0]
    return huc_str.zfill(10)[:digits]


# ================== MULTI-INDICATOR UPSTREAM CLASSIFICATION ==================

def classify_upstream_multi_indicator(sensor_attrs, usgs_attrs, same_huc8=True):
    """
    Classify if sensor is upstream of USGS station using multiple indicators.
    
    Indicators (sensor vs USGS):
      1. Elevation: sensor > USGS (higher = upstream)
      2. Stream Order: sensor <= USGS (smaller/equal = headwater tributary)
      3. Flow Accumulation: sensor < USGS (less area = upstream)
      4. Upstream Length: sensor < USGS (less network above = upstream)
    
    Returns:
      (is_upstream, confidence, n_indicators_agree, indicator_details)
    """
    indicators = {}
    n_agree = 0
    
    # 1. Elevation check (PRIMARY)
    sensor_elev = sensor_attrs.get('elevation_m')
    usgs_elev = usgs_attrs.get('elevation_m')
    if sensor_elev is not None and usgs_elev is not None:
        elev_diff = sensor_elev - usgs_elev
        indicators['elevation'] = {
            'sensor': sensor_elev,
            'usgs': usgs_elev,
            'diff': elev_diff,
            'upstream': elev_diff > ELEVATION_THRESHOLD_M
        }
        if indicators['elevation']['upstream']:
            n_agree += 1
    
    # 2. Stream Order check
    sensor_order = sensor_attrs.get('nhd_streamorder')
    usgs_order = usgs_attrs.get('nhd_streamorder')
    if sensor_order is not None and usgs_order is not None:
        indicators['stream_order'] = {
            'sensor': sensor_order,
            'usgs': usgs_order,
            'upstream': sensor_order <= usgs_order  # Smaller or equal = tributary/headwater
        }
        if indicators['stream_order']['upstream']:
            n_agree += 1
    
    # 3. Flow Accumulation check
    sensor_flowacc = sensor_attrs.get('flowacc')
    usgs_flowacc = usgs_attrs.get('flowacc')
    if sensor_flowacc is not None and usgs_flowacc is not None:
        # Log scale comparison (flow acc varies by orders of magnitude)
        ratio = sensor_flowacc / max(usgs_flowacc, 1)
        indicators['flow_accumulation'] = {
            'sensor': sensor_flowacc,
            'usgs': usgs_flowacc,
            'ratio': ratio,
            'upstream': ratio < 1.0  # Less drainage area = upstream
        }
        if indicators['flow_accumulation']['upstream']:
            n_agree += 1
    
    # 4. Upstream Length check
    sensor_upkm = sensor_attrs.get('nhd_upstream_km')
    usgs_upkm = usgs_attrs.get('nhd_upstream_km')
    if sensor_upkm is not None and usgs_upkm is not None:
        indicators['upstream_km'] = {
            'sensor': sensor_upkm,
            'usgs': usgs_upkm,
            'upstream': sensor_upkm < usgs_upkm  # Less network above = more upstream
        }
        if indicators['upstream_km']['upstream']:
            n_agree += 1
    
    # Determine confidence
    n_indicators = len(indicators)
    
    if n_indicators == 0:
        return False, 'invalid', 0, indicators
    
    # Must have elevation as primary indicator
    if 'elevation' not in indicators or not indicators['elevation']['upstream']:
        return False, 'low', n_agree, indicators
    
    # Confidence based on agreement
    if same_huc8:
        if n_agree >= 3:
            confidence = 'high'
        elif n_agree >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
    else:
        # Lower confidence without HUC8 match
        if n_agree >= 4:
            confidence = 'medium'
        elif n_agree >= 3:
            confidence = 'low'
        else:
            confidence = 'proxy'
    
    is_upstream = (confidence in ['high', 'medium']) or (confidence == 'low' and n_agree >= 2)
    
    return is_upstream, confidence, n_agree, indicators


# ================== LOAD DATA ==================

print("\n STEP 1: Loading data...")

HUC_COLUMNS = ['huc10', 'huc_cd', 'huc8', 'huc12']
dtype_dict = {col: str for col in HUC_COLUMNS}

# Load multi-scenario sensor locations
try:
    df_selected_all = pd.read_csv(SELECTED_SITES_ALL_FILE, dtype=dtype_dict)
    scenarios_in_data = df_selected_all['scenario'].unique()
    print(f"  [OK] Loaded {len(df_selected_all):,} sensor locations across {len(scenarios_in_data)} scenarios")
    for scen in scenarios_in_data:
        n = (df_selected_all['scenario'] == scen).sum()
        print(f"      {scen}: {n} sensors")
except FileNotFoundError:
    raise FileNotFoundError(f"Multi-scenario output not found: {SELECTED_SITES_ALL_FILE}")

# Verify required columns
required_cols = ['elevation_m', 'nhd_streamorder', 'flowacc', 'nhd_upstream_km']
available_cols = [c for c in required_cols if c in df_selected_all.columns]
print(f"\n  Multi-indicator columns:")
print(f"    [OK] Available: {available_cols}")

# Load basin summaries
try:
    df_basin_summary_all = pd.read_csv(BASIN_SUMMARY_ALL_FILE, dtype=dtype_dict)
    print(f"  [OK] Loaded basin summaries")
except FileNotFoundError:
    df_basin_summary_all = pd.DataFrame()

# Load master dataset
try:
    df_master = pd.read_csv(MASTER_DATA_FILE, dtype=dtype_dict)
    print(f"  [OK] Loaded {len(df_master):,} candidate points from master dataset")
except FileNotFoundError:
    df_master = None
    print("  Master dataset not found")

# Load USGS stations
try:
    df_usgs = pd.read_csv(USGS_STATIONS_FILE, dtype=dtype_dict)
    if 'dec_lat_va' in df_usgs.columns:
        df_usgs['lat'] = df_usgs['dec_lat_va'].astype(float)
        df_usgs['lon'] = df_usgs['dec_long_va'].astype(float)
    
    df_usgs = df_usgs.dropna(subset=['lat', 'lon'])
    df_usgs = df_usgs[(df_usgs['lat'] > 0) & (df_usgs['lon'] < 0)]
    n_usgs = len(df_usgs)
    print(f"  [OK] Loaded {n_usgs} USGS streamgages")
except FileNotFoundError:
    raise FileNotFoundError(f"USGS stations file not found: {USGS_STATIONS_FILE}")

# Load shapefile
gdf_basins = None
try:
    import geopandas as gpd
    if os.path.exists(HUC10_SHAPEFILE):
        gdf_basins = gpd.read_file(HUC10_SHAPEFILE)
        print(f" Loaded {len(gdf_basins)} HUC10 basin boundaries")
except ImportError:
    pass
except Exception as e:
    print(f"  Could not load shapefile: {e}")


# ================== SNAP USGS TO GET ATTRIBUTES ==================

print("\n" + "=" * 80)
print("STEP 2: SNAPPING USGS STATIONS TO GET NHDPlus ATTRIBUTES")
print("=" * 80)

if df_master is not None:
    lat_mean = np.radians(df_master['lat'].mean())
    cos_lat = np.cos(lat_mean)
    
    lons_master = df_master['lon'].to_numpy(float)
    lats_master = df_master['lat'].to_numpy(float)
    xy_master = np.column_stack([lons_master * cos_lat * 111.0, lats_master * 111.0])
    tree_master = cKDTree(xy_master)
    
    lons_usgs = df_usgs['lon'].to_numpy(float)
    lats_usgs = df_usgs['lat'].to_numpy(float)
    xy_usgs_orig = np.column_stack([lons_usgs * cos_lat * 111.0, lats_usgs * 111.0])
    
    usgs_snapped = []
    
    for i in range(n_usgs):
        dist, idx = tree_master.query(xy_usgs_orig[i])
        
        snap_info = {
            'usgs_idx': i,
            'site_no': df_usgs['site_no'].iloc[i] if 'site_no' in df_usgs.columns else str(i),
            'station_nm': df_usgs['station_nm'].iloc[i] if 'station_nm' in df_usgs.columns else f'USGS_{i}',
            'usgs_lon': lons_usgs[i],
            'usgs_lat': lats_usgs[i],
            'snap_distance_km': dist,
            'snap_valid': dist <= SNAP_TOLERANCE_KM
        }
        
        if dist <= SNAP_TOLERANCE_KM:
            snap_info['snapped_lon'] = lons_master[idx]
            snap_info['snapped_lat'] = lats_master[idx]
            snap_info['elevation_m'] = safe_float(df_master['elevation_m'].iloc[idx])
            snap_info['nhd_streamorder'] = safe_float(df_master['nhd_streamorder'].iloc[idx])
            snap_info['flowacc'] = safe_float(df_master['flowacc'].iloc[idx])
            snap_info['nhd_upstream_km'] = safe_float(df_master['nhd_upstream_km'].iloc[idx])
            snap_info['huc10'] = normalize_huc(df_master['huc10'].iloc[idx], 10)
            snap_info['huc8'] = normalize_huc(df_master['huc10'].iloc[idx], 8)
        else:
            for col in ['snapped_lon', 'snapped_lat', 'elevation_m', 'nhd_streamorder', 
                       'flowacc', 'nhd_upstream_km', 'huc10', 'huc8']:
                snap_info[col] = None
        
        usgs_snapped.append(snap_info)
    
    df_usgs_snapped = pd.DataFrame(usgs_snapped)
    
    n_valid = df_usgs_snapped['snap_valid'].sum()
    print(f"\n USGS Snapping Results:")
    print(f"    Total: {n_usgs}")
    print(f"    Successfully snapped: {n_valid} ({100*n_valid/n_usgs:.1f}%)")
    print(f"    Mean snap distance: {df_usgs_snapped['snap_distance_km'].mean():.3f} km")
    
    df_usgs_snapped.to_csv(OUT_USGS_SNAPPED, index=False)
    print(f"  Saved: {OUT_USGS_SNAPPED}")
else:
    df_usgs_snapped = None
    print("  Cannot snap USGS - master dataset not available")


# ================== PREPARE ARRAYS ==================

print("\n STEP 3: Preparing analysis arrays...")

# Use original USGS coordinates for spatial analysis
usgs_lons = df_usgs['lon'].to_numpy(float)
usgs_lats = df_usgs['lat'].to_numpy(float)
usgs_ids = df_usgs['site_no'].astype(str).to_numpy() if 'site_no' in df_usgs.columns else np.arange(n_usgs).astype(str)
usgs_names = df_usgs['station_nm'].to_numpy() if 'station_nm' in df_usgs.columns else usgs_ids

# Detection radius
if len(df_basin_summary_all) > 0 and 'detection_radius_discharge' in df_basin_summary_all.columns:
    DETECTION_RADIUS_KM = float(df_basin_summary_all['detection_radius_discharge'].mean())
else:
    DETECTION_RADIUS_KM = DEFAULT_DETECTION_RADIUS_KM

print(f"  Detection radius: {DETECTION_RADIUS_KM:.1f} km")
print(f"  USGS stations: {n_usgs}")

# Risk field
if df_master is not None:
    flood_freq = df_master['flood_events_norm'].to_numpy(float)
    flood_sev = df_master['flood_risk_value_norm'].to_numpy(float)
    risk_all = minmax(0.6 * flood_freq + 0.4 * flood_sev)
    lons_all = df_master['lon'].to_numpy(float)
    lats_all = df_master['lat'].to_numpy(float)
    total_weighted_risk = float(risk_all.sum())
    print(f"  Total weighted flood risk: {total_weighted_risk:.1f}")
else:
    risk_all = None

# Spatial indices
xy_usgs = np.column_stack([usgs_lons * cos_lat * 111.0, usgs_lats * 111.0])
tree_usgs = cKDTree(xy_usgs)

if df_master is not None:
    xy_all = np.column_stack([lons_all * cos_lat * 111.0, lats_all * 111.0])
else:
    xy_all = None

print("  Spatial indices built")


# ================== USGS BASELINE COVERAGE ==================

print("\n" + "=" * 80)
print("LAYER 1: USGS BASELINE COVERAGE")
print("=" * 80)

if xy_all is not None:
    usgs_covered, total_risk, usgs_coverage, usgs_points_covered, usgs_mask = compute_coverage(
        xy_usgs, xy_all, risk_all, DETECTION_RADIUS_KM
    )
    
    high_risk_threshold = np.percentile(risk_all, 75)
    desert_mask = (~usgs_mask) & (risk_all >= high_risk_threshold)
    desert_risk = risk_all[desert_mask].sum()
    
    print(f"\n USGS Network Flood Coverage:")
    print(f"    Stations: {n_usgs}")
    print(f"    Detection radius: {DETECTION_RADIUS_KM:.1f} km")
    print(f"    Coverage: {usgs_coverage:.1%}")
    print(f"    Desert risk (high-risk, uncovered): {desert_risk:.1f}")
else:
    usgs_coverage = 0.0
    usgs_mask = None
    desert_mask = None


# ================== METHOD EXPLANATION ==================

print("\n" + "=" * 80)
print(" MULTI-INDICATOR UPSTREAM CLASSIFICATION (v3.1-HYBRID)")
print("=" * 80)

print("""
This method uses MULTIPLE VALID NHDPlus variables to classify upstream 
relationships, avoiding corrupted topology columns.

INDICATORS USED:
  1. ELEVATION (Primary): Sensor elevation > USGS elevation + 10m
     -> Higher elevation = upstream in mountain terrain

  2. STREAM ORDER: Sensor order <= USGS order
     -> Smaller streams (lower order) are tributaries/headwaters

  3. FLOW ACCUMULATION: Sensor flowacc < USGS flowacc
     -> Less drainage area = more upstream position

  4. UPSTREAM LENGTH: Sensor upstream_km < USGS upstream_km
     -> Less stream network above = closer to headwaters

CONFIDENCE LEVELS:
  - HIGH: Elevation + 2-3 other indicators agree + same HUC8
  - MEDIUM: Elevation + 1 other indicator agrees + same HUC8
  - LOW: Only elevation indicates upstream

This approach is physically justified for Appalachian headwaters where:
  - Topographic gradients strongly control flow direction
  - Stream order reflects position in drainage hierarchy
  - Flow accumulation correlates with downstream position
""")


# ================== ANALYZE EACH SCENARIO ==================

print("\n" + "=" * 80)
print("STEP 4: ANALYZING EACH SCENARIO WITH MULTI-INDICATOR METHOD")
print("=" * 80)

scenario_results = {}
all_classifications = []
all_lead_times = []
all_synergy = []
all_upstream_details = []
indicator_summary = []

for scenario_name in scenarios_in_data:
    print(f"\n{'-'*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'-'*60}")
    
    scenario_config = SCENARIOS.get(scenario_name, {
        'short_name': scenario_name.split(':')[0].strip(),
        'color': 'gray',
        'marker': 'o'
    })
    
    df_scenario = df_selected_all[df_selected_all['scenario'] == scenario_name].copy()
    n_sensors = len(df_scenario)
    
    lons_sel = df_scenario['lon'].to_numpy(float)
    lats_sel = df_scenario['lat'].to_numpy(float)
    xy_sel = np.column_stack([lons_sel * cos_lat * 111.0, lats_sel * 111.0])
    
    print(f"  Sensors: {n_sensors}")
    
    # Get sensor attributes
    sensor_elevations = df_scenario['elevation_m'].to_numpy(float)
    sensor_stream_orders = df_scenario['nhd_streamorder'].to_numpy(float) if 'nhd_streamorder' in df_scenario.columns else np.full(n_sensors, np.nan)
    sensor_flowacc = df_scenario['flowacc'].to_numpy(float) if 'flowacc' in df_scenario.columns else np.full(n_sensors, np.nan)
    sensor_upstream_km = df_scenario['nhd_upstream_km'].to_numpy(float) if 'nhd_upstream_km' in df_scenario.columns else np.full(n_sensors, np.nan)
    sensor_huc10 = df_scenario['huc10'].apply(lambda x: normalize_huc(x, 10)).to_numpy() if 'huc10' in df_scenario.columns else np.full(n_sensors, None)
    sensor_huc8 = df_scenario['huc10'].apply(lambda x: normalize_huc(x, 8)).to_numpy() if 'huc10' in df_scenario.columns else np.full(n_sensors, None)
    
    # ========== LAYER 2: SYNERGY ==========
    if xy_all is not None:
        prop_covered, _, prop_coverage, _, prop_mask = compute_coverage(
            xy_sel, xy_all, risk_all, DETECTION_RADIUS_KM
        )
        
        combined_coords = np.vstack([xy_usgs, xy_sel])
        combined_covered, _, combined_coverage, _, _ = compute_coverage(
            combined_coords, xy_all, risk_all, DETECTION_RADIUS_KM
        )
        
        overlap = (usgs_covered + prop_covered) - combined_covered
        redundancy_ratio = overlap / max(prop_covered, 1e-9)
        incremental_coverage = combined_coverage - usgs_coverage
        
        desert_covered = risk_all[desert_mask & prop_mask].sum() if prop_mask is not None and desert_mask is not None else 0
        
        print(f"  Coverage: Proposed={prop_coverage:.1%}, Combined={combined_coverage:.1%}, Delta=+{incremental_coverage:.1%}")
        
        all_synergy.append({
            'scenario': scenario_name,
            'short_name': scenario_config['short_name'],
            'n_sensors': n_sensors,
            'usgs_coverage_pct': usgs_coverage * 100,
            'proposed_coverage_pct': prop_coverage * 100,
            'combined_coverage_pct': combined_coverage * 100,
            'incremental_pct': incremental_coverage * 100,
            'redundancy_ratio_pct': redundancy_ratio * 100
        })
    else:
        prop_coverage = combined_coverage = incremental_coverage = redundancy_ratio = 0
    
    # ========== LAYER 3: MULTI-INDICATOR UPSTREAM CLASSIFICATION ==========
    upstream_candidates = {}
    n_high = n_medium = n_low = 0
    indicator_counts = {'elevation': 0, 'stream_order': 0, 'flow_accumulation': 0, 'upstream_km': 0}
    
    for j in range(n_sensors):
        sensor_attrs = {
            'elevation_m': safe_float(sensor_elevations[j]),
            'nhd_streamorder': safe_float(sensor_stream_orders[j]),
            'flowacc': safe_float(sensor_flowacc[j]),
            'nhd_upstream_km': safe_float(sensor_upstream_km[j])
        }
        sensor_huc8_val = sensor_huc8[j]
        
        for i in range(n_usgs):
            if df_usgs_snapped is None or not df_usgs_snapped.iloc[i]['snap_valid']:
                continue
            
            # Spatial proximity check
            dist_km = haversine_km(lons_sel[j], lats_sel[j], usgs_lons[i], usgs_lats[i])
            if dist_km > MAX_SEARCH_DISTANCE_KM:
                continue
            
            usgs_row = df_usgs_snapped.iloc[i]
            usgs_attrs = {
                'elevation_m': safe_float(usgs_row['elevation_m']),
                'nhd_streamorder': safe_float(usgs_row['nhd_streamorder']),
                'flowacc': safe_float(usgs_row['flowacc']),
                'nhd_upstream_km': safe_float(usgs_row['nhd_upstream_km'])
            }
            usgs_huc8_val = usgs_row['huc8']
            
            # HUC8 check
            same_huc8 = (sensor_huc8_val is not None and usgs_huc8_val is not None and 
                        sensor_huc8_val == usgs_huc8_val)
            
            if SAME_HUC8_REQUIRED and not same_huc8:
                continue
            
            # Multi-indicator classification
            is_upstream, confidence, n_agree, indicators = classify_upstream_multi_indicator(
                sensor_attrs, usgs_attrs, same_huc8
            )
            
            if is_upstream:
                if j not in upstream_candidates:
                    upstream_candidates[j] = []
                
                upstream_candidates[j].append({
                    'usgs_idx': i,
                    'usgs_id': usgs_ids[i],
                    'confidence': confidence,
                    'n_indicators': n_agree,
                    'indicators': indicators,
                    'distance_km': dist_km,
                    'same_huc8': same_huc8
                })
                
                # Count by confidence
                if confidence == 'high':
                    n_high += 1
                elif confidence == 'medium':
                    n_medium += 1
                else:
                    n_low += 1
                
                # Count indicator agreement
                for ind_name, ind_data in indicators.items():
                    if ind_data.get('upstream', False):
                        indicator_counts[ind_name] += 1
                
                # Store details
                all_upstream_details.append({
                    'scenario': scenario_name,
                    'sensor_idx': j,
                    'sensor_lon': lons_sel[j],
                    'sensor_lat': lats_sel[j],
                    'usgs_idx': i,
                    'usgs_id': usgs_ids[i],
                    'distance_km': dist_km,
                    'same_huc8': same_huc8,
                    'confidence': confidence,
                    'n_indicators_agree': n_agree,
                    'elev_sensor': sensor_attrs['elevation_m'],
                    'elev_usgs': usgs_attrs['elevation_m'],
                    'elev_diff': (sensor_attrs['elevation_m'] or 0) - (usgs_attrs['elevation_m'] or 0),
                    'order_sensor': sensor_attrs['nhd_streamorder'],
                    'order_usgs': usgs_attrs['nhd_streamorder'],
                    'flowacc_sensor': sensor_attrs['flowacc'],
                    'flowacc_usgs': usgs_attrs['flowacc'],
                    'upkm_sensor': sensor_attrs['nhd_upstream_km'],
                    'upkm_usgs': usgs_attrs['nhd_upstream_km']
                })
    
    n_upstream = len(upstream_candidates)
    print(f"  Upstream candidates: {n_upstream}/{n_sensors} ({100*n_upstream/n_sensors:.0f}%)")
    print(f"    Confidence: {n_high} high, {n_medium} medium, {n_low} low")
    print(f"    Indicator agreement: elev={indicator_counts['elevation']}, order={indicator_counts['stream_order']}, "
          f"flowacc={indicator_counts['flow_accumulation']}, upkm={indicator_counts['upstream_km']}")
    
    indicator_summary.append({
        'scenario': scenario_name,
        'n_upstream': n_upstream,
        'n_high': n_high,
        'n_medium': n_medium,
        'n_low': n_low,
        **{f'ind_{k}': v for k, v in indicator_counts.items()}
    })
    
    # ========== CLASSIFICATION ==========
    sensor_is_upstream = {j: True for j in upstream_candidates.keys()}
    
    classifications = []
    cascade_lead_times = []
    
    for j in range(n_sensors):
        dist_to_usgs, nearest_idx = tree_usgs.query(xy_sel[j])
        is_upstream = j in sensor_is_upstream
        
        # Get best upstream match
        upstream_info = upstream_candidates.get(j, [])
        best_upstream = None
        if upstream_info:
            # Prefer high confidence, then closest distance
            high_conf = [u for u in upstream_info if u['confidence'] == 'high']
            if high_conf:
                best_upstream = min(high_conf, key=lambda x: x['distance_km'])
            else:
                med_conf = [u for u in upstream_info if u['confidence'] == 'medium']
                if med_conf:
                    best_upstream = min(med_conf, key=lambda x: x['distance_km'])
                else:
                    best_upstream = min(upstream_info, key=lambda x: x['distance_km'])
        
        # Classification
        if dist_to_usgs <= DETECTION_RADIUS_KM:
            if is_upstream:
                classification = "Cascade Sentinel"
            else:
                classification = "Validator"
        else:
            if is_upstream:
                classification = "Sentinel"
            else:
                classification = "Gap-Filler"
        
        classifications.append({
            'scenario': scenario_name,
            'short_name': scenario_config['short_name'],
            'sensor_idx': j,
            'lon': lons_sel[j],
            'lat': lats_sel[j],
            'classification': classification,
            'dist_to_nearest_usgs_km': dist_to_usgs,
            'nearest_usgs_id': usgs_ids[nearest_idx],
            'is_upstream': is_upstream,
            'upstream_confidence': best_upstream['confidence'] if best_upstream else None,
            'upstream_usgs_id': best_upstream['usgs_id'] if best_upstream else None,
            'n_indicators_agree': best_upstream['n_indicators'] if best_upstream else None
        })
        
        # Lead time for Cascade Sentinels
        if classification == "Cascade Sentinel" and best_upstream:
            stream_order = sensor_stream_orders[j] if np.isfinite(sensor_stream_orders[j]) else 3
            lead_time, velocity = estimate_lead_time_hours(best_upstream['distance_km'], stream_order)
            
            cascade_lead_times.append({
                'scenario': scenario_name,
                'short_name': scenario_config['short_name'],
                'sensor_idx': j,
                'lon': lons_sel[j],
                'lat': lats_sel[j],
                'upstream_usgs_id': best_upstream['usgs_id'],
                'distance_km': best_upstream['distance_km'],
                'stream_order': stream_order,
                'velocity_ms': velocity,
                'lead_time_hours': lead_time,
                'confidence': best_upstream['confidence'],
                'n_indicators': best_upstream['n_indicators']
            })
    
    class_df = pd.DataFrame(classifications)
    class_counts = class_df['classification'].value_counts()
    
    n_sentinel = class_counts.get('Sentinel', 0)
    n_cascade = class_counts.get('Cascade Sentinel', 0)
    n_validator = class_counts.get('Validator', 0)
    n_gapfiller = class_counts.get('Gap-Filler', 0)
    
    print(f"  Classification: Sentinel={n_sentinel}, Cascade={n_cascade}, Validator={n_validator}, Gap-Filler={n_gapfiller}")
    
    # Lead time stats
    if cascade_lead_times:
        lead_df = pd.DataFrame(cascade_lead_times)
        median_lead = lead_df['lead_time_hours'].median()
        mean_dist = lead_df['distance_km'].mean()
        print(f"  Cascade Sentinel: median lead time={median_lead:.1f} hrs, mean distance={mean_dist:.1f} km")
        all_lead_times.extend(cascade_lead_times)
    else:
        median_lead = None
    
    all_classifications.extend(classifications)
    
    scenario_results[scenario_name] = {
        'config': scenario_config,
        'n_sensors': n_sensors,
        'n_upstream': n_upstream,
        'n_high': n_high,
        'n_medium': n_medium,
        'n_low': n_low,
        'n_sentinel': n_sentinel,
        'n_cascade': n_cascade,
        'n_validator': n_validator,
        'n_gapfiller': n_gapfiller,
        'prop_coverage': prop_coverage,
        'combined_coverage': combined_coverage,
        'incremental': incremental_coverage,
        'redundancy': redundancy_ratio,
        'median_lead_time': median_lead,
        'lons': lons_sel,
        'lats': lats_sel,
        'classifications': class_df
    }


# ================== COMPARISON TABLE ==================

print("\n" + "=" * 80)
print("SCENARIO COMPARISON: USGS INTEGRATION (Multi-Indicator Method)")
print("=" * 80)

comparison_rows = []
for scenario_name, results in scenario_results.items():
    comparison_rows.append({
        'Scenario': results['config']['short_name'],
        'N_Sensors': results['n_sensors'],
        'USGS_Coverage': f"{usgs_coverage:.1%}",
        'Proposed_Coverage': f"{results['prop_coverage']:.1%}",
        'Combined_Coverage': f"{results['combined_coverage']:.1%}",
        'Incremental': f"+{results['incremental']:.1%}",
        'Upstream': f"{results['n_upstream']} ({100*results['n_upstream']/results['n_sensors']:.0f}%)",
        'High_Conf': results['n_high'],
        'Med_Conf': results['n_medium'],
        'Sentinel': results['n_sentinel'],
        'Cascade': results['n_cascade'],
        'Gap_Filler': results['n_gapfiller'],
        'Validator': results['n_validator'],
        'Lead_Time': f"{results['median_lead_time']:.1f} hrs" if results['median_lead_time'] else "N/A"
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n" + comparison_df.to_string(index=False))


# ================== SAVE RESULTS ==================

print("\n Saving results...")

pd.DataFrame(all_synergy).to_csv(OUT_SYNERGY_ALL, index=False)
print(f"  {OUT_SYNERGY_ALL}")

pd.DataFrame(all_classifications).to_csv(OUT_CLASSIFICATION_ALL, index=False)
print(f" {OUT_CLASSIFICATION_ALL}")

if all_upstream_details:
    pd.DataFrame(all_upstream_details).to_csv(OUT_UPSTREAM_DETAILS, index=False)
    print(f" {OUT_UPSTREAM_DETAILS}")

if all_lead_times:
    pd.DataFrame(all_lead_times).to_csv(OUT_LEAD_TIME_ALL, index=False)
    print(f" {OUT_LEAD_TIME_ALL}")

comparison_df.to_csv(OUT_SCENARIO_COMPARISON, index=False)
print(f" {OUT_SCENARIO_COMPARISON}")

pd.DataFrame(indicator_summary).to_csv(OUT_INDICATOR_SUMMARY, index=False)
print(f" {OUT_INDICATOR_SUMMARY}")


# ================== VISUALIZATIONS ==================

print("\n Creating visualizations...")

colors_class = {'Sentinel': 'green', 'Cascade Sentinel': 'blue', 'Gap-Filler': 'orange', 'Validator': 'purple'}

# Figure 1: Scenario Comparison
fig1, axes = plt.subplots(2, 3, figsize=(18, 12))

scenario_names_short = [r['config']['short_name'] for r in scenario_results.values()]
scenario_colors = [r['config']['color'] for r in scenario_results.values()]

# Panel 1: Network size
ax1 = axes[0, 0]
n_sensors_list = [r['n_sensors'] for r in scenario_results.values()]
bars1 = ax1.bar(scenario_names_short, n_sensors_list, color=scenario_colors, edgecolor='black')
ax1.set_ylabel('Number of Sensors')
ax1.set_title('Network Size by Scenario', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, n_sensors_list):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 5, str(val), ha='center', fontweight='bold')

# Panel 2: Combined coverage
ax2 = axes[0, 1]
combined_list = [r['combined_coverage']*100 for r in scenario_results.values()]
bars2 = ax2.bar(scenario_names_short, combined_list, color=scenario_colors, edgecolor='black')
ax2.axhline(usgs_coverage*100, color='navy', linestyle='--', linewidth=2, label=f'USGS: {usgs_coverage:.1%}')
ax2.set_ylabel('Combined Coverage (%)')
ax2.set_title('Flood Risk Coverage (USGS + Proposed)', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Upstream confidence
ax3 = axes[0, 2]
x = np.arange(len(scenario_names_short))
width = 0.6
high_list = [r['n_high'] for r in scenario_results.values()]
med_list = [r['n_medium'] for r in scenario_results.values()]
low_list = [r['n_low'] for r in scenario_results.values()]
ax3.bar(x, high_list, width, label='High', color='darkgreen', edgecolor='black')
ax3.bar(x, med_list, width, bottom=high_list, label='Medium', color='lightgreen', edgecolor='black')
ax3.bar(x, low_list, width, bottom=np.array(high_list)+np.array(med_list), label='Low', color='yellow', edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(scenario_names_short)
ax3.set_ylabel('Upstream Classifications')
ax3.set_title('Multi-Indicator Upstream Confidence', fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Classification
ax4 = axes[1, 0]
sentinel_list = [r['n_sentinel'] for r in scenario_results.values()]
cascade_list = [r['n_cascade'] for r in scenario_results.values()]
gapfiller_list = [r['n_gapfiller'] for r in scenario_results.values()]
validator_list = [r['n_validator'] for r in scenario_results.values()]
ax4.bar(x, sentinel_list, width, label='Sentinel', color='green', edgecolor='black')
ax4.bar(x, cascade_list, width, bottom=sentinel_list, label='Cascade Sentinel', color='blue', edgecolor='black')
ax4.bar(x, gapfiller_list, width, bottom=np.array(sentinel_list)+np.array(cascade_list), label='Gap-Filler', color='orange', edgecolor='black')
ax4.bar(x, validator_list, width, bottom=np.array(sentinel_list)+np.array(cascade_list)+np.array(gapfiller_list), label='Validator', color='purple', edgecolor='black')
ax4.set_xticks(x)
ax4.set_xticklabels(scenario_names_short)
ax4.set_ylabel('Number of Sensors')
ax4.set_title('Sensor Classification', fontweight='bold')
ax4.legend(loc='upper right')
ax4.grid(axis='y', alpha=0.3)

# Panel 5: Lead time
ax5 = axes[1, 1]
lead_times_list = [r['median_lead_time'] if r['median_lead_time'] else 0 for r in scenario_results.values()]
bars5 = ax5.bar(scenario_names_short, lead_times_list, color=scenario_colors, edgecolor='black')
ax5.set_ylabel('Median Lead Time (hours)')
ax5.set_title('Cascade Sentinel Early Warning Time', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
for bar, val in zip(bars5, lead_times_list):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontweight='bold')

# Panel 6: Incremental coverage
ax6 = axes[1, 2]
incr_list = [r['incremental']*100 for r in scenario_results.values()]
bars6 = ax6.bar(scenario_names_short, incr_list, color=scenario_colors, edgecolor='black')
ax6.set_ylabel('Incremental Beyond USGS (%)')
ax6.set_title('Added Value of Proposed Network', fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
for bar, val in zip(bars6, incr_list):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 1, f'+{val:.1f}%', ha='center', fontweight='bold')

fig1.suptitle('USGS INTEGRATION ANALYSIS: Multi-Indicator Method (v3.1-HYBRID)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_FIG_COMPARISON, dpi=300, bbox_inches='tight', facecolor='white')
print(f" {OUT_FIG_COMPARISON}")

# Figure 2: Maps
fig2, axes2 = plt.subplots(2, 2, figsize=(20, 16))

for idx, (scenario_name, results) in enumerate(scenario_results.items()):
    ax = axes2[idx // 2, idx % 2]
    config = results['config']
    
    if df_master is not None:
        ax.scatter(lons_all, lats_all, c=risk_all, s=3, alpha=0.2, cmap='YlOrRd', vmin=0, vmax=1)
    
    ax.scatter(usgs_lons, usgs_lats, c='navy', s=100, marker='^', 
               edgecolors='black', linewidths=1.5, label=f'USGS (n={n_usgs})', zorder=10)
    
    class_df = results['classifications']
    for cls in ['Sentinel', 'Cascade Sentinel', 'Gap-Filler', 'Validator']:
        mask = class_df['classification'] == cls
        if mask.sum() > 0:
            ax.scatter(class_df.loc[mask, 'lon'], class_df.loc[mask, 'lat'],
                       c=colors_class[cls], s=60, marker='*', edgecolors='black', linewidths=0.5,
                       label=f'{cls} (n={mask.sum()})', zorder=5, alpha=0.8)
    
    if gdf_basins is not None:
        try:
            gdf_basins.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
        except:
            pass
    
    ax.set_title(f"{config['short_name']}: N={results['n_sensors']} | Combined={results['combined_coverage']:.1%}",
                 fontsize=11, fontweight='bold', color=config['color'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

fig2.suptitle('USGS INTEGRATION: Sensor Classification Maps (Multi-Indicator Method)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_FIG_MAPS_ALL, dpi=300, bbox_inches='tight', facecolor='white')
print(f" {OUT_FIG_MAPS_ALL}")

# Figure 3: Lead Times
if all_lead_times:
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    
    lead_df = pd.DataFrame(all_lead_times)
    
    # Box plot
    ax3a = axes3[0, 0]
    scenario_shorts = [SCENARIOS[s]['short_name'] for s in scenarios_in_data if s in SCENARIOS]
    lead_by_scenario = [lead_df.loc[lead_df['short_name'] == s, 'lead_time_hours'].values for s in scenario_shorts]
    lead_by_scenario = [l for l in lead_by_scenario if len(l) > 0]
    if lead_by_scenario:
        bp = ax3a.boxplot(lead_by_scenario, labels=scenario_shorts[:len(lead_by_scenario)], patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(list(SCENARIOS.values())[i]['color'] if i < len(SCENARIOS) else 'gray')
    ax3a.set_ylabel('Lead Time (hours)')
    ax3a.set_title('Cascade Sentinel Lead Times', fontweight='bold')
    ax3a.grid(axis='y', alpha=0.3)
    
    # Distance vs Lead time
    ax3b = axes3[0, 1]
    for scenario_name, results in scenario_results.items():
        config = results['config']
        scenario_leads = lead_df[lead_df['short_name'] == config['short_name']]
        if len(scenario_leads) > 0:
            ax3b.scatter(scenario_leads['distance_km'], scenario_leads['lead_time_hours'],
                        c=config['color'], s=40, alpha=0.6, label=config['short_name'],
                        marker=config['marker'], edgecolors='black', linewidths=0.3)
    ax3b.set_xlabel('Distance to USGS (km)')
    ax3b.set_ylabel('Lead Time (hours)')
    ax3b.set_title('Distance vs Lead Time', fontweight='bold')
    ax3b.legend()
    ax3b.grid(alpha=0.3)
    
    # Histogram
    ax3c = axes3[1, 0]
    ax3c.hist(lead_df['lead_time_hours'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax3c.axvline(lead_df['lead_time_hours'].median(), color='red', linestyle='--', linewidth=2, 
                 label=f'Median: {lead_df["lead_time_hours"].median():.2f} hrs')
    ax3c.set_xlabel('Lead Time (hours)')
    ax3c.set_ylabel('Count')
    ax3c.set_title('Lead Time Distribution (All Cascade Sentinels)', fontweight='bold')
    ax3c.legend()
    ax3c.grid(alpha=0.3)
    
    # Confidence pie
    ax3d = axes3[1, 1]
    conf_counts = lead_df['confidence'].value_counts()
    ax3d.pie(conf_counts.values, labels=conf_counts.index, autopct='%1.1f%%',
             colors=['darkgreen', 'lightgreen', 'yellow'], startangle=90)
    ax3d.set_title('Upstream Classification Confidence\n(Cascade Sentinels)', fontweight='bold')
    
    fig3.suptitle('CASCADE SENTINEL EARLY WARNING ANALYSIS', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_FIG_LEAD_TIMES, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" {OUT_FIG_LEAD_TIMES}")

plt.close('all')


# ================== FINAL SUMMARY ==================

print("\n" + "=" * 80)
print("USGS INTEGRATION ANALYSIS COMPLETE (v3.1-HYBRID)")
print("=" * 80)

print(f"""
 METHODOLOGY: Multi-Indicator Upstream Classification

  Indicators used:
    Elevation (sensor > USGS + 10m)
    Stream Order (sensor <= USGS)
    Flow Accumulation (sensor < USGS)
    Upstream Length (sensor < USGS)

  Confidence levels:
    - HIGH: 3+ indicators agree + same HUC8
    - MEDIUM: 2 indicators agree + same HUC8
    - LOW: Elevation only

 KEY FINDINGS:

1. BASELINE USGS COVERAGE: {usgs_coverage:.1%}

2. SCENARIO COMPARISON:
""")

for scenario_name, results in scenario_results.items():
    short = results['config']['short_name']
    print(f"""
   {short}:
     Sensors: {results['n_sensors']}
     Combined coverage: {results['combined_coverage']:.1%} (+{results['incremental']:.1%})
     Upstream: {results['n_upstream']} ({100*results['n_upstream']/results['n_sensors']:.0f}%)
       High: {results['n_high']}, Medium: {results['n_medium']}, Low: {results['n_low']}
     Classification: Sentinel={results['n_sentinel']}, Cascade={results['n_cascade']}, 
                     Gap-Filler={results['n_gapfiller']}, Validator={results['n_validator']}
     Lead time: {f"{results['median_lead_time']:.2f} hrs" if results['median_lead_time'] else "N/A"}""")

print(f"""

 OUTPUT FILES:
   - {OUT_USGS_SNAPPED}
   - {OUT_SYNERGY_ALL}
   - {OUT_CLASSIFICATION_ALL}
   - {OUT_UPSTREAM_DETAILS}
   - {OUT_LEAD_TIME_ALL}
   - {OUT_INDICATOR_SUMMARY}
   - {OUT_FIG_COMPARISON}
   - {OUT_FIG_MAPS_ALL}
   - {OUT_FIG_LEAD_TIMES}

 Analysis complete using multi-indicator method
  Physically justified for Appalachian headwater systems
""")

print("=" * 80)
