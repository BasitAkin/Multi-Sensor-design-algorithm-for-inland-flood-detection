"""
BASIN-BY-BASIN Multi-Sensor Placement for Flood Detection (v4.4-S)

BASE: v4.4 Algorithm (unchanged)
ADDED: Four Operational Scenarios comparison with FULL outputs for ALL scenarios
  - A: Maximum Coverage (lambda=0.10, tau=0.02)
  - B: Balanced (lambda=0.15, tau=0.05) - baseline
  - C: Precision-Focused (lambda=0.25, tau=0.05)
  - D: Resource-Constrained (lambda=0.15, tau=0.10)

OUTPUTS:
  - Combined CSV with ALL scenario sensor locations (with scenario column)
  - Individual maps for each scenario
  - All graphics comparing all scenarios
  - Basin summaries for all scenarios
  - Baseline comparisons for ALL scenarios (NEW)
  - All statistics printed to screen (NEW)

TABLES GENERATED:
  - Table 1: Network Configuration Under Alternative Operational Scenarios
  - Table 2: Method Comparison by Scenario (Multi-Sensor vs Baselines)
  - Table 3: Detection Radius Sensitivity Analysis
  - Table 4: Pareto Curves (TP-FP Trade-off by Network Size)
  - Table 5: Submodularity Verification
  - Table 6: Inter-Sensor Distance Statistics
  - Table 7: Per-Basin Performance Statistics
  - Table 8: Selected Site Suitability Statistics
  - Table 9: Sensor Distribution by Risk Tier
  - Table 10: Summary Statistics Across All Scenarios

Author: WINGS Lab
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from scipy.spatial import cKDTree, ConvexHull
from scipy.linalg import qr as scipy_qr
import warnings
import time

warnings.filterwarnings('ignore')

# ================== INPUT PATHS - CONFIGURE THESE ==================
MASTER_F = os.path.join("data", "master_dataset.csv")
HUC10_SHAPEFILE = os.path.join("data", "studyarea_huc10s.shp")
HUC10_ID_COLUMN = "huc10"
POINTS_HUC_COLUMN = "huc10"

# ================== OUTPUT PATHS ==================
OUT_DIR = os.path.join(os.getcwd(), "Multisensor_Algorithms_outputs_v44S")
os.makedirs(OUT_DIR, exist_ok=True)

# Combined outputs for ALL scenarios
OUT_SEL_ALL = os.path.join(OUT_DIR, "selected_sites_all_scenarios.csv")
OUT_BASIN_SUMMARY_ALL = os.path.join(OUT_DIR, "basin_summary_all_scenarios.csv")
OUT_SCENARIO_TABLE = os.path.join(OUT_DIR, "scenario_comparison_table.csv")
OUT_DIAG = os.path.join(OUT_DIR, "diagnostics_basin")
OUT_BASELINES_ALL = os.path.join(OUT_DIR, "baselines_all_scenarios.csv")

# Figures
OUT_FIG_MAIN = os.path.join(OUT_DIR, "basin_analysis_main.png")
OUT_FIG_MAPS_ALL = os.path.join(OUT_DIR, "sensor_maps_all_scenarios.png")
OUT_FIG_BASINS = os.path.join(OUT_DIR, "basin_analysis_maps.png")
OUT_FIG_DIAGNOSTICS = os.path.join(OUT_DIR, "basin_analysis_diagnostics.png")
OUT_FIG_PERF_DIST = os.path.join(OUT_DIR, "basin_performance_distributions.png")
OUT_FIG_SCENARIOS = os.path.join(OUT_DIR, "scenario_comparison.png")

# ================== OPERATIONAL SCENARIOS ==================
SCENARIOS = {
    'A_MaxCoverage': {
        'name': 'A: Maximum Coverage',
        'short_name': 'MaxCoverage',
        'fp_weight': 0.10,
        'cost_fraction': 0.02,
        'color': '#2ecc71',
        'marker': 'o',
        'description': 'Detect as many floods as possible; accept more false alarms; budget is flexible.'
    },
    'B_Balanced': {
        'name': 'B: Balanced',
        'short_name': 'Balanced',
        'fp_weight': 0.15,
        'cost_fraction': 0.05,
        'color': '#3498db',
        'marker': 's',
        'description': 'Balance detection and precision with reasonable network size - Current baseline'
    },
    'C_PrecisionFocused': {
        'name': 'C: Precision-Focused',
        'short_name': 'Precision',
        'fp_weight': 0.25,
        'cost_fraction': 0.05,
        'color': '#9b59b6',
        'marker': '^',
        'description': 'Minimize false alarms; maintain moderate network size'
    },
    'D_ResourceConstrained': {
        'name': 'D: Resource-Constrained',
        'short_name': 'Resource',
        'fp_weight': 0.15,
        'cost_fraction': 0.10,
        'color': '#e74c3c',
        'marker': 'D',
        'description': 'Limited deployment capacity; place only highest-value sensors'
    }
}

# ================== STRATIFIED SAMPLING CONFIG ==================
GRID_RESOLUTION_DEG = 0.01
N_RISK_QUINTILES = 5
MAX_PER_CELL_PER_QUINTILE = 2
TARGET_SHORTLIST_MIN_PER_BASIN = 100
TARGET_SHORTLIST_MAX_PER_BASIN = 3000
USE_QR_REFINEMENT = True
QR_REFINEMENT_FACTOR = 1.3
STRATIFIED_SAMPLING_THRESHOLD = 500

# ================== ALGORITHM CONFIG ==================
USE_SENSOR_COST = True
MAX_SENSORS_PER_BASIN = 100
MIN_SENSORS_PER_BASIN = 2
PLATEAU_WINDOW = 5
MIN_SITES_PER_BASIN = 10
EDGE_MERGE_DISTANCE_KM = 5.0

# ================== PREPROCESSING CONFIG ==================
SENTINEL_THRESHOLD = -9990
WINSORIZE_LOWER = 1
WINSORIZE_UPPER = 99

# ================== DETECTION CONFIG ==================
LAKE_PENALTY_MAX = 0.30
SENSOR_QUALITY = {'stage': 0.85, 'discharge': 0.80, 'camera': 0.70}
SYNERGY_BONUS = 0.15
SYNERGY_THRESH = 0.10
LOW_RISK_TAU = 0.20
RISK_SYNERGY_MIN = 0.30

# Risk bins
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
BIN_LABELS = ["0.00-0.19", "0.20-0.39", "0.40-0.59", "0.60-0.79", "0.80-1.00"]
TIER_COLORS = {0: "#fee8c8", 1: "#fdbb84", 2: "#fc8d59", 3: "#e34a33", 4: "#b30000"}
nb = len(BIN_LABELS)

# QR features
QR_FEATURES = [
    'flood_events_norm', 'flood_risk_value_norm',
    'nhd_totdasqkm', 'nhd_streamorder', 'flowacc', 'twi',
    'slope_deg', 'elevation_m', 'rain_mean',
    'nhdplusflowlinevaa_nc_slope', 'LakeFract'
]

# ================== EVALUATION CONFIG ==================
EVAL_MAX_BASINS = 20
BASELINE_N_TRIALS = 10
BASELINES_USE_SPACING = True

print("="*80)
print("BASIN-BY-BASIN MULTI-SENSOR FLOOD DETECTION NETWORK (v4.4-S)")
print("  -> Stratified Sampling + QR | FULL Diagnostics | Basin Maps")
print("  -> Four Operational Scenarios (lambda x tau) - ALL OUTPUTS")
print("  -> Baseline Comparisons for ALL Scenarios")
print("="*80)

# ================== UTILITY FUNCTIONS ==================
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def clean_and_winsorize(arr, sentinel_thresh=SENTINEL_THRESHOLD,
                        lower_pct=WINSORIZE_LOWER, upper_pct=WINSORIZE_UPPER):
    arr = np.asarray(arr, dtype=float)
    arr = np.where(arr <= sentinel_thresh, np.nan, arr)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return arr
    p_low = np.percentile(valid, lower_pct)
    p_high = np.percentile(valid, upper_pct)
    arr = np.where(np.isfinite(arr), np.clip(arr, p_low, p_high), arr)
    return arr

def minmax(x):
    x = np.asarray(x, float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    rng = xmax - xmin
    if not np.isfinite(rng) or rng <= 0:
        return np.zeros_like(x)
    result = (x - xmin) / (rng + 1e-12)
    return np.where(np.isfinite(result), np.clip(result, 0, 1), 0.0)

def zscore_matrix(X):
    X = X.astype(float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    Z = (X - mu) / sd
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z

def qr_pivot_select(X, k):
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        return np.arange(min(k, X.shape[0]))
    k = int(max(1, min(k, X.shape[0])))
    _, _, piv = scipy_qr(X.T, pivoting=True, mode="economic")
    return np.array(piv[:k], dtype=int)

def compute_spatial_isolation_kdtree(lons, lats, k=10, eta=0.15, wmin=0.85, wmax=1.15):
    n = len(lons)
    k = min(k, n - 1)
    if n <= k + 1:
        return np.ones(n)
    lat_mean = np.radians(lats.mean())
    x = lons * np.cos(lat_mean) * 111.0
    y = lats * 111.0
    xy = np.column_stack([x, y])
    tree = cKDTree(xy)
    distances, _ = tree.query(xy, k=k+1)
    isolation = distances[:, 1:k+1].mean(axis=1)
    mu = isolation.mean()
    sd = isolation.std() if isolation.std() > 0 else 1.0
    z = (isolation - mu) / sd
    weights = 1.0 + eta * z
    weights = np.clip(weights, wmin, wmax)
    return weights

def compute_semivariogram_range(lons, lats, risk, max_samples=300):
    N = len(lons)
    step = max(1, N // max_samples)
    idx = np.arange(0, N, step)
    if len(idx) < 10:
        return 15.0
    D_s = np.zeros((len(idx), len(idx)))
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            D_s[i, j] = haversine_km(lons[ii], lats[ii], lons[jj], lats[jj])
    r_s = risk[idx]
    pairs = np.triu_indices_from(D_s, k=1)
    h = D_s[pairs]
    gamma = 0.5 * (r_s[pairs[0]] - r_s[pairs[1]]) ** 2
    mask = h > 0
    if mask.sum() < 10:
        return 15.0
    hmin, hmax = h[mask].min(), h[mask].max()
    if hmax - hmin < 1.0:
        return 15.0
    nbins = min(12, len(idx) // 5)
    edges = np.linspace(hmin, hmax, nbins + 1)
    h_mid = 0.5 * (edges[:-1] + edges[1:])
    gamma_med = np.array([
        np.median(gamma[(h >= edges[i]) & (h < edges[i + 1])])
        if np.any((h >= edges[i]) & (h < edges[i + 1])) else np.nan
        for i in range(nbins)
    ])
    top = max(2, nbins // 3)
    sill = np.nanmedian(gamma_med[-top:])
    if np.isfinite(sill) and sill > 0:
        try:
            h95 = h_mid[np.where(gamma_med >= 0.95 * sill)[0][0]]
            a_range = h95 / 2.996
        except IndexError:
            where80 = np.where(gamma_med >= 0.80 * sill)[0]
            a_range = (h_mid[where80[0]] / 1.609) if len(where80) else 15.0
    else:
        a_range = 15.0
    return float(np.clip(a_range, 5.0, 30.0))

def stratified_spatial_sampling(df, lons, lats, risk, target_min, target_max):
    N = len(df)
    if N <= target_min:
        return np.arange(N)

    lon_range = lons.max() - lons.min()
    lat_range = lats.max() - lats.min()
    estimated_area_deg2 = lon_range * lat_range
    target_cells = min(500, max(50, N // 100))
    grid_res = np.sqrt(estimated_area_deg2 / target_cells)
    grid_res = max(0.005, min(0.05, grid_res))

    lon_bins = np.floor(lons / grid_res).astype(int)
    lat_bins = np.floor(lats / grid_res).astype(int)
    cell_ids = lon_bins * 100000 + lat_bins
    unique_cells = np.unique(cell_ids)

    try:
        risk_quintiles = pd.qcut(risk, q=N_RISK_QUINTILES, labels=False, duplicates='drop')
    except ValueError:
        risk_quintiles = np.zeros(N, dtype=int)
    n_actual_quintiles = len(np.unique(risk_quintiles))

    selected_indices = []
    for cell in unique_cells:
        cell_mask = cell_ids == cell
        cell_indices = np.where(cell_mask)[0]
        for q in range(n_actual_quintiles):
            q_mask = risk_quintiles[cell_indices] == q
            q_indices = cell_indices[q_mask]
            if len(q_indices) == 0:
                continue
            q_risks = risk[q_indices]
            sorted_idx = np.argsort(-q_risks)
            n_take = min(MAX_PER_CELL_PER_QUINTILE, len(q_indices))
            selected_indices.extend(q_indices[sorted_idx[:n_take]])

    selected_indices = np.unique(np.array(selected_indices, dtype=int))

    if len(selected_indices) > target_max:
        keep_ratio = target_max / len(selected_indices)
        final_indices = []
        for q in range(n_actual_quintiles):
            q_mask = risk_quintiles[selected_indices] == q
            q_idx = selected_indices[q_mask]
            n_keep = max(1, int(len(q_idx) * keep_ratio))
            rng = np.random.default_rng(42)
            final_indices.extend(rng.choice(q_idx, size=min(n_keep, len(q_idx)), replace=False))
        selected_indices = np.array(final_indices, dtype=int)

    if len(selected_indices) < target_min and len(selected_indices) < N:
        remaining = np.setdiff1d(np.arange(N), selected_indices)
        if len(remaining) > 0:
            remaining_risks = risk[remaining]
            n_add = min(target_min - len(selected_indices), len(remaining))
            top_remaining = remaining[np.argsort(-remaining_risks)[:n_add]]
            selected_indices = np.concatenate([selected_indices, top_remaining])

    return np.unique(selected_indices)

def qr_refinement(df, selected_indices, lons, lats, target_size):
    if len(selected_indices) <= target_size:
        return selected_indices
    available_features = [c for c in QR_FEATURES if c in df.columns and df[c].notnull().any()]
    if len(available_features) < 2:
        return selected_indices
    df_subset = df.iloc[selected_indices]
    X = df_subset[available_features].to_numpy(dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            median_val = np.nanmedian(col)
            X[mask, j] = median_val if np.isfinite(median_val) else 0.0
    Z = zscore_matrix(X)

    lons_subset = lons[selected_indices]
    lats_subset = lats[selected_indices]
    spatial_weights = compute_spatial_isolation_kdtree(
        lons_subset, lats_subset,
        k=min(10, len(selected_indices) // 10),
        eta=0.15, wmin=0.85, wmax=1.15
    )
    Z_weighted = Z * spatial_weights[:, np.newaxis]
    qr_selected = qr_pivot_select(Z_weighted, target_size)
    return selected_indices[qr_selected]

def load_huc10_info(shapefile_path, id_column):
    try:
        import geopandas as gpd
    except ImportError:
        print("  geopandas not installed. Using HUC IDs from points only.")
        return None, None
    if not os.path.exists(shapefile_path):
        print(f"  Shapefile not found: {shapefile_path}")
        return None, None

    print(f"\n Loading HUC10 shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)

    huc_col = None
    for col in gdf.columns:
        if col.lower() == id_column.lower():
            huc_col = col
            break
    if huc_col is None:
        for alt in ['huc10', 'HUC10', 'HUC_10', 'huc_10', 'HUC10_ID']:
            if alt in gdf.columns:
                huc_col = alt
                break
    if huc_col is None:
        print(f"  HUC ID column not found. Columns: {list(gdf.columns)}")
        return None, None

    print(f"  Using HUC ID column: '{huc_col}'")

    if gdf.crs and gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(epsg=5070)
        areas_m2 = gdf_proj.geometry.area
    else:
        areas_m2 = gdf.geometry.area
    areas_km2 = areas_m2 / 1e6

    basin_info = {}
    for idx, row in gdf.iterrows():
        huc_id = str(row[huc_col])
        basin_info[huc_id] = {
            'geometry': row.geometry,
            'area_km2': float(areas_km2.iloc[idx])
        }

    print(f"  [OK] Loaded {len(basin_info)} HUC10 basins")
    print(f"  Total area: {sum(b['area_km2'] for b in basin_info.values()):,.0f} km^2")
    return basin_info, gdf


class BasinProcessor:
    """Process a single HUC10 basin with full tracking (v4.4 algorithm)."""

    def __init__(self, df_basin, huc_id, basin_area_km2=None):
        self.df_original = df_basin.reset_index(drop=True)
        self.huc_id = huc_id
        self.N_original = len(self.df_original)
        self.df = None
        self.N = 0
        self.sampling_indices = None
        self.lons_original = self.df_original['lon'].to_numpy(float)
        self.lats_original = self.df_original['lat'].to_numpy(float)
        if basin_area_km2 is not None:
            self.basin_area_km2 = basin_area_km2
        else:
            self.basin_area_km2 = self._estimate_area(self.lons_original, self.lats_original)
        self.selected_indices = []
        self.performance = {}
        self.detection_radii = {}
        self.sampling_stats = {}
        self.history = []

    def _estimate_area(self, lons, lats):
        if len(lons) < 3:
            return 10.0
        try:
            lat_mean = np.radians(lats.mean())
            x = lons * np.cos(lat_mean) * 111.0
            y = lats * 111.0
            hull = ConvexHull(np.column_stack([x, y]))
            return float(hull.volume)
        except:
            lon_range = (lons.max() - lons.min()) * np.cos(np.radians(lats.mean())) * 111.0
            lat_range = (lats.max() - lats.min()) * 111.0
            return lon_range * lat_range

    def run_stratified_sampling(self):
        N = self.N_original
        flood_freq = self.df_original['flood_events_norm'].to_numpy(float)
        flood_sev = self.df_original['flood_risk_value_norm'].to_numpy(float)
        risk_raw = 0.6 * flood_freq + 0.4 * flood_sev
        risk_min, risk_max = risk_raw.min(), risk_raw.max()
        if risk_max > risk_min:
            risk_for_sampling = (risk_raw - risk_min) / (risk_max - risk_min)
        else:
            risk_for_sampling = np.zeros(N)

        if N <= STRATIFIED_SAMPLING_THRESHOLD:
            self.sampling_indices = np.arange(N)
            self.sampling_stats = {
                'original': N, 'after_stratified': N, 'after_sampling': N,
                'method': 'none (small basin)'
            }
        else:
            stratified_indices = stratified_spatial_sampling(
                self.df_original, self.lons_original, self.lats_original, risk_for_sampling,
                TARGET_SHORTLIST_MIN_PER_BASIN, TARGET_SHORTLIST_MAX_PER_BASIN
            )
            n_after_stratified = len(stratified_indices)

            if USE_QR_REFINEMENT and n_after_stratified > TARGET_SHORTLIST_MIN_PER_BASIN:
                qr_target = int(n_after_stratified / QR_REFINEMENT_FACTOR)
                qr_target = max(TARGET_SHORTLIST_MIN_PER_BASIN, qr_target)
                final_indices = qr_refinement(
                    self.df_original, stratified_indices,
                    self.lons_original, self.lats_original, qr_target
                )
            else:
                final_indices = stratified_indices

            self.sampling_indices = final_indices
            self.sampling_stats = {
                'original': N, 'after_stratified': n_after_stratified,
                'after_sampling': len(final_indices),
                'method': 'stratified + QR' if USE_QR_REFINEMENT else 'stratified'
            }

        self.df = self.df_original.iloc[self.sampling_indices].reset_index(drop=True)
        self.N = len(self.df)
        self.lons = self.df['lon'].to_numpy(float)
        self.lats = self.df['lat'].to_numpy(float)

    def compute_risk(self):
        flood_freq = self.df['flood_events_norm'].to_numpy(float)
        flood_sev = self.df['flood_risk_value_norm'].to_numpy(float)
        risk_raw = 0.6 * flood_freq + 0.4 * flood_sev
        risk_min, risk_max = risk_raw.min(), risk_raw.max()
        if risk_max > risk_min:
            self.risk = (risk_raw - risk_min) / (risk_max - risk_min)
        else:
            self.risk = np.zeros(self.N)
        self.risk = np.clip(self.risk, 0.0, 1.0)
        self.total_risk = float(self.risk.sum())
        self.risk_bin = np.digitize(self.risk, BIN_EDGES, right=False) - 1
        self.risk_bin = np.clip(self.risk_bin, 0, len(BIN_LABELS) - 1)

    def compute_suitability(self):
        def nz_norm(col):
            if col in self.df.columns and self.df[col].notnull().any():
                return minmax(self.df[col].to_numpy(float))
            return np.zeros(self.N)

        def nz_norm_log(col):
            if col in self.df.columns and self.df[col].notnull().any():
                vals = self.df[col].to_numpy(float)
                vals = np.where((vals > 0) & np.isfinite(vals), vals, 1e-10)
                return minmax(np.log1p(vals))
            return np.zeros(self.N)

        FLOWACC_log = nz_norm_log('flowacc')
        TOTDA_log = nz_norm_log('nhd_totdasqkm')
        SO = nz_norm('nhd_streamorder')
        SLV_inv = 1.0 - nz_norm('nhd_streamlevel')
        CH_SLOPE = nz_norm('nhdplusflowlinevaa_nc_slope')
        TWI = nz_norm('twi')
        SLOPE_inv = 1.0 - nz_norm('slope_deg')
        ELEV_inv = 1.0 - nz_norm('elevation_m')
        RAIN = nz_norm('rain_mean')
        LAKE_n = nz_norm('LakeFract')
        self.lake_penalty = np.clip((LAKE_n - 0.2) / 0.8, 0, 1)

        self.discharge_suit = np.clip(
            0.25 * FLOWACC_log + 0.20 * TOTDA_log + 0.15 * SO +
            0.15 * SLV_inv + 0.10 * CH_SLOPE + 0.15 * RAIN, 0, 1)

        self.stage_suit = np.clip(
            0.20 * TWI + 0.20 * SLOPE_inv + 0.15 * ELEV_inv +
            0.15 * FLOWACC_log + 0.15 * RAIN + 0.15 * SO, 0, 1)

        camera_base = (0.25 * SLOPE_inv + 0.25 * ELEV_inv + 0.20 * TWI +
                       0.15 * SO + 0.15 * TOTDA_log)
        self.camera_suit = np.clip(camera_base * (1 - LAKE_PENALTY_MAX * self.lake_penalty), 0, 1)

    def calibrate_detection_radii(self):
        corr_length = compute_semivariogram_range(self.lons, self.lats, self.risk)
        self.detection_radii = {
            'stage': round(0.9 * corr_length, 1),
            'discharge': round(1.0 * corr_length, 1),
            'camera': round(0.5 * corr_length, 1)
        }

    def setup_detection_model(self):
        lat_mean = np.radians(self.lats.mean())
        self.xy_km = np.column_stack([
            self.lons * np.cos(lat_mean) * 111.0,
            self.lats * 111.0
        ])
        self.kdtree = cKDTree(self.xy_km)
        if self.N <= 3000:
            self.D = np.zeros((self.N, self.N))
            for i in range(self.N):
                self.D[i, :] = haversine_km(self.lons[i], self.lats[i], self.lons, self.lats)
            self.use_full_D = True
        else:
            self.D = None
            self.use_full_D = False

        max_radius = max(self.detection_radii.values()) * 2.5
        if self.use_full_D:
            self.neighbor_lists = [np.where(self.D[i, :] <= max_radius)[0] for i in range(self.N)]
        else:
            self.neighbor_lists = self.kdtree.query_ball_point(self.xy_km, r=max_radius)

    def get_distance(self, i, j):
        if self.use_full_D:
            return self.D[i, j]
        return haversine_km(self.lons[i], self.lats[i], self.lons[j], self.lats[j])

    def p_detect_one(self, sidx, eidx, stype, radius_mult=1.0):
        d = self.get_distance(sidx, eidx)
        rho = self.detection_radii[stype] * radius_mult
        if d > 2.5 * rho:
            return 0.0
        q = SENSOR_QUALITY[stype]
        decay = np.exp(-d / max(rho, 1e-6))
        suit = {'stage': self.stage_suit, 'discharge': self.discharge_suit,
                'camera': self.camera_suit}[stype][sidx]
        return float(q * suit * decay)

    def compute_candidate_contribution(self, c, radius_mult=1.0):
        p_s = np.zeros(self.N)
        p_q = np.zeros(self.N)
        p_c = np.zeros(self.N)
        for i in self.neighbor_lists[c]:
            p_s[i] = self.p_detect_one(c, i, 'stage', radius_mult)
            p_q[i] = self.p_detect_one(c, i, 'discharge', radius_mult)
            p_c[i] = self.p_detect_one(c, i, 'camera', radius_mult)
        return p_s, p_q, p_c

    def compute_network_score(self, p_stage, p_q, p_cam, fp_weight):
        p_miss = (1 - p_stage) * (1 - p_q) * (1 - p_cam)
        p_any = 1 - p_miss

        multi = ((p_stage > SYNERGY_THRESH).astype(int) +
                 (p_q > SYNERGY_THRESH).astype(int) +
                 (p_cam > SYNERGY_THRESH).astype(int)) >= 2

        synergy_mask = (self.risk >= RISK_SYNERGY_MIN).astype(float)
        p_any = np.minimum(1.0, p_any + SYNERGY_BONUS * multi.astype(float) * synergy_mask * self.risk)

        low = self.risk < LOW_RISK_TAU
        p_any_low = p_any.copy()
        p_any_low[low] *= self.risk[low]

        TP = float((self.risk * p_any).sum())
        FP = float(((1 - self.risk[low]) * p_any_low[low]).sum())

        TP_rate = TP / max(self.total_risk, 1e-9)
        FP_rate = FP / max(((1 - self.risk[low]).sum()), 1e-9)
        J = TP_rate - fp_weight * FP_rate
        return TP_rate, FP_rate, J

    def evaluate_selection(self, selection, radius_mult=1.0, fp_weight=0.15):
        p_s, p_q, p_c = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)
        for s in selection:
            ps, pq, pc = self.compute_candidate_contribution(s, radius_mult)
            p_s = np.maximum(p_s, ps)
            p_q = np.maximum(p_q, pq)
            p_c = np.maximum(p_c, pc)
        return self.compute_network_score(p_s, p_q, p_c, fp_weight)

    def get_spacing(self, i):
        rho = self.detection_radii['discharge']
        base = 0.4 * rho
        factor = 0.6 if self.risk_bin[i] >= 3 else 1.0
        return base * factor

    def spacing_ok(self, c, selected):
        if not selected:
            return True
        min_dist = min(self.get_distance(c, s) for s in selected)
        return min_dist >= self.get_spacing(c)

    def run_greedy_selection(self, fp_weight, cost_fraction):
        """Run greedy selection with configurable FP weight (lambda) and cost fraction (tau)."""
        selected = []
        p_stage = np.zeros(self.N)
        p_q = np.zeros(self.N)
        p_cam = np.zeros(self.N)
        J_raw_curr = 0.0
        J_cost_curr = 0.0
        first_raw_gain = None
        cost_per_sensor = 0.0
        recent_cost_gains = []
        best_selected = []
        best_Jcost = -1e18
        self.history = []

        while len(selected) < MAX_SENSORS_PER_BASIN:
            best_gain_cost = -1e18
            best_c = None
            best_tp = best_fp = best_Jraw = best_Jcost_local = None
            best_p_s = best_p_q = best_p_c = None

            for c in range(self.N):
                if c in selected or not self.spacing_ok(c, selected):
                    continue

                p_s_new, p_q_new, p_c_new = self.compute_candidate_contribution(c)
                p_s_upd = np.maximum(p_stage, p_s_new)
                p_q_upd = np.maximum(p_q, p_q_new)
                p_c_upd = np.maximum(p_cam, p_c_new)

                tp, fp, J_raw_new = self.compute_network_score(p_s_upd, p_q_upd, p_c_upd, fp_weight)

                c_tmp = cost_per_sensor if (USE_SENSOR_COST and first_raw_gain is not None) else 0.0
                J_cost_new = J_raw_new - c_tmp * (len(selected) + 1)
                gain_cost = J_cost_new - J_cost_curr

                if gain_cost > best_gain_cost:
                    best_gain_cost = gain_cost
                    best_c = c
                    best_tp, best_fp, best_Jraw, best_Jcost_local = tp, fp, J_raw_new, J_cost_new
                    best_p_s, best_p_q, best_p_c = p_s_upd, p_q_upd, p_c_upd

            if best_c is None:
                break

            if first_raw_gain is None:
                first_raw_gain = max(best_Jraw - J_raw_curr, 1e-12)
                if USE_SENSOR_COST:
                    cost_per_sensor = float(cost_fraction * first_raw_gain)
                    best_Jcost_local = best_Jraw - cost_per_sensor * 1
                    best_gain_cost = best_Jcost_local
                    J_cost_curr = 0.0

            recent_cost_gains.append(best_gain_cost)
            if len(recent_cost_gains) > PLATEAU_WINDOW:
                recent_cost_gains.pop(0)

            if len(selected) >= MIN_SENSORS_PER_BASIN:
                if len(recent_cost_gains) >= PLATEAU_WINDOW and all(g <= 0 for g in recent_cost_gains):
                    break
                if best_gain_cost <= 0:
                    break

            selected.append(best_c)
            p_stage, p_q, p_cam = best_p_s, best_p_q, best_p_c
            J_raw_curr = best_Jraw
            J_cost_curr = best_Jcost_local

            self.history.append({
                'iteration': len(selected),
                'sensor_idx': best_c,
                'marginal_gain': float(best_gain_cost),
                'tp_rate': float(best_tp),
                'fp_rate': float(best_fp),
                'J_raw': float(best_Jraw),
                'J_cost': float(best_Jcost_local),
                'huc10': self.huc_id
            })

            if best_Jcost_local > best_Jcost:
                best_Jcost = best_Jcost_local
                best_selected = selected.copy()

        self.selected_indices = best_selected if best_selected else selected
        self.history = self.history[:len(self.selected_indices)]

        tp_f, fp_f, j_f = self.evaluate_selection(self.selected_indices, fp_weight=fp_weight)

        self.performance = {
            'K': len(self.selected_indices),
            'TP_rate': tp_f,
            'FP_rate': fp_f,
            'J': j_f,
            'J_cost': j_f - (cost_per_sensor * len(self.selected_indices) if USE_SENSOR_COST else 0.0),
            'cost_per_sensor': cost_per_sensor,
            'density_km2_per_sensor': self.basin_area_km2 / max(len(self.selected_indices), 1),
            'fp_weight': fp_weight,
            'cost_fraction': cost_fraction
        }

    def get_results_df(self, scenario_name=None):
        if not self.selected_indices:
            return pd.DataFrame()
        original_indices = self.sampling_indices[self.selected_indices]
        sel_df = self.df_original.iloc[original_indices].copy()
        sel_df['huc10'] = self.huc_id
        sel_df['selection_rank_in_basin'] = np.arange(1, len(self.selected_indices) + 1)
        sel_df['stage_suitability'] = self.stage_suit[self.selected_indices]
        sel_df['discharge_suitability'] = self.discharge_suit[self.selected_indices]
        sel_df['camera_suitability'] = self.camera_suit[self.selected_indices]
        sel_df['lake_penalty'] = self.lake_penalty[self.selected_indices]
        sel_df['risk_in_basin'] = self.risk[self.selected_indices]
        sel_df['risk_bin'] = self.risk_bin[self.selected_indices]
        sel_df['detection_radius_stage'] = self.detection_radii['stage']
        sel_df['detection_radius_discharge'] = self.detection_radii['discharge']
        sel_df['detection_radius_camera'] = self.detection_radii['camera']
        if scenario_name:
            sel_df['scenario'] = scenario_name
        return sel_df

    def get_summary(self, scenario_name=None):
        if not self.selected_indices:
            return {}
        summary = {
            'huc10': self.huc_id,
            'n_original': self.N_original,
            'n_after_sampling': self.N,
            'sampling_method': self.sampling_stats.get('method', 'unknown'),
            'n_selected': len(self.selected_indices),
            'basin_area_km2': self.basin_area_km2,
            'density_km2_per_sensor': self.performance.get('density_km2_per_sensor', np.nan),
            'TP_rate': self.performance.get('TP_rate', np.nan),
            'FP_rate': self.performance.get('FP_rate', np.nan),
            'J': self.performance.get('J', np.nan),
            'J_cost': self.performance.get('J_cost', np.nan),
            'discharge_suit_median': np.median(self.discharge_suit[self.selected_indices]),
            'stage_suit_median': np.median(self.stage_suit[self.selected_indices]),
            'camera_suit_median': np.median(self.camera_suit[self.selected_indices]),
            'risk_median': np.median(self.risk[self.selected_indices]),
            'detection_radius_discharge': self.detection_radii['discharge']
        }
        if scenario_name:
            summary['scenario'] = scenario_name
        return summary


def merge_edge_sensors(all_selected_df, merge_distance_km=EDGE_MERGE_DISTANCE_KM):
    if len(all_selected_df) < 2:
        return all_selected_df
    lons = all_selected_df['lon'].to_numpy()
    lats = all_selected_df['lat'].to_numpy()
    hucs = all_selected_df['huc10'].to_numpy()
    combined_suit = (all_selected_df['stage_suitability'].to_numpy() +
                     all_selected_df['discharge_suitability'].to_numpy() +
                     all_selected_df['camera_suitability'].to_numpy()) / 3
    lat_mean = np.radians(lats.mean())
    xy = np.column_stack([lons * np.cos(lat_mean) * 111.0, lats * 111.0])
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=merge_distance_km)
    to_remove = set()
    for i, j in pairs:
        if hucs[i] != hucs[j]:
            if combined_suit[i] < combined_suit[j]:
                to_remove.add(i)
            else:
                to_remove.add(j)
    if to_remove:
        keep_mask = ~np.isin(np.arange(len(all_selected_df)), list(to_remove))
        return all_selected_df[keep_mask].reset_index(drop=True)
    return all_selected_df


# ================== SCENARIO RUNNER ==================
def run_scenario(df_full, basin_info, unique_hucs, scenario_key, scenario_config):
    """Run a single scenario and return results."""
    print(f"\n{'-'*60}")
    print(f"SCENARIO: {scenario_config['name']}")
    print(f"  lambda (FP weight) = {scenario_config['fp_weight']}")
    print(f"  tau (cost fraction) = {scenario_config['cost_fraction']}")
    print(f"{'-'*60}")
    
    all_results = []
    basin_summaries = []
    skipped_basins = []
    all_history = []
    processors = {}
    
    start_time = time.time()
    
    for i, huc_id in enumerate(unique_hucs):
        df_basin = df_full[df_full['_huc10_'] == huc_id].copy()
        n_sites = len(df_basin)
        
        if n_sites < MIN_SITES_PER_BASIN:
            skipped_basins.append({'huc10': huc_id, 'n_sites': n_sites, 'reason': 'too_few_sites'})
            continue
        
        basin_area = basin_info[huc_id]['area_km2'] if basin_info and huc_id in basin_info else None
        
        try:
            processor = BasinProcessor(df_basin, huc_id, basin_area)
            processor.run_stratified_sampling()
            processor.compute_risk()
            processor.compute_suitability()
            processor.calibrate_detection_radii()
            processor.setup_detection_model()
            processor.run_greedy_selection(
                fp_weight=scenario_config['fp_weight'],
                cost_fraction=scenario_config['cost_fraction']
            )
            
            # Pass scenario name to results
            results_df = processor.get_results_df(scenario_name=scenario_config['name'])
            if len(results_df) > 0:
                all_results.append(results_df)
                basin_summaries.append(processor.get_summary(scenario_name=scenario_config['name']))
                all_history.extend(processor.history)
                processors[huc_id] = processor
                
        except Exception as e:
            skipped_basins.append({'huc10': huc_id, 'n_sites': n_sites, 'reason': str(e)})
    
    elapsed = time.time() - start_time
    
    if all_results:
        all_selected_df = pd.concat(all_results, ignore_index=True)
        all_selected_df = merge_edge_sensors(all_selected_df)
        # Re-add scenario column after merge (in case it was lost)
        all_selected_df['scenario'] = scenario_config['name']
    else:
        all_selected_df = pd.DataFrame()
    
    basin_summary_df = pd.DataFrame(basin_summaries)
    
    total_sensors = len(all_selected_df)
    n_basins = len(basin_summaries)
    sensors_per_basin = total_sensors / n_basins if n_basins > 0 else 0
    
    mean_tp = basin_summary_df['TP_rate'].mean() if len(basin_summary_df) > 0 else 0
    mean_fp = basin_summary_df['FP_rate'].mean() if len(basin_summary_df) > 0 else 0
    mean_j = basin_summary_df['J'].mean() if len(basin_summary_df) > 0 else 0
    
    print(f"  [OK] Completed in {elapsed/60:.1f} min")
    print(f"    Sensors: {total_sensors} | TP: {mean_tp:.1%} | FP: {mean_fp:.1%} | J: {mean_j:.3f}")
    
    return {
        'config': scenario_config,
        'all_selected_df': all_selected_df,
        'basin_summary_df': basin_summary_df,
        'processors': processors,
        'all_history': all_history,
        'skipped_basins': skipped_basins,
        'stats': {
            'n_sensors': total_sensors,
            'n_basins': n_basins,
            'sensors_per_basin': sensors_per_basin,
            'TP_rate': mean_tp,
            'FP_rate': mean_fp,
            'J_score': mean_j,
            'elapsed_min': elapsed / 60
        }
    }


def create_scenario_comparison_table(scenario_results):
    """Create comparison table in expected format."""
    rows = []
    for scenario_key, results in scenario_results.items():
        config = results['config']
        stats = results['stats']
        rows.append({
            'Scenario': config['name'],
            'lambda': config['fp_weight'],
            'tau': config['cost_fraction'],
            'N_Sensors': stats['n_sensors'],
            'TP_Rate': f"{stats['TP_rate']:.1%}",
            'FP_Rate': f"{stats['FP_rate']:.1%}",
            'J_Score': f"{stats['J_score']:.3f}",
            'Sensors_Per_Basin': f"~{stats['sensors_per_basin']:.0f}"
        })
    return pd.DataFrame(rows)


# ================== BASELINE COMPUTATION FOR ALL SCENARIOS ==================
def compute_baselines_for_scenario(processors, fp_weight, scenario_name, eval_max_basins=EVAL_MAX_BASINS, n_trials=BASELINE_N_TRIALS):
    """Compute baseline comparisons for a single scenario."""
    eval_procs = list(processors.values())
    
    if not eval_procs:
        return pd.DataFrame()
    
    rng = np.random.default_rng(42)
    if eval_max_basins and len(eval_procs) > eval_max_basins:
        eval_procs = [eval_procs[i] for i in rng.choice(len(eval_procs), eval_max_basins, replace=False)]
    
    def _fill_to_k(indices, K, N):
        if len(indices) >= K:
            return indices[:K]
        remaining = [i for i in range(N) if i not in indices]
        indices.extend(remaining[:K - len(indices)])
        return indices[:K]
    
    def _pick_by_rank(proc, ranked, K, use_spacing=True):
        sel = []
        for c in ranked:
            if (not use_spacing) or proc.spacing_ok(c, sel):
                sel.append(int(c))
                if len(sel) >= K:
                    break
        return _fill_to_k(sel, K, proc.N)
    
    def baseline_random(proc, K, seed, use_spacing=True):
        rng_local = np.random.default_rng(seed)
        order = rng_local.permutation(proc.N).tolist()
        return _pick_by_rank(proc, order, K, use_spacing=use_spacing)
    
    def baseline_uniform_grid(proc, K, use_spacing=True):
        grid = int(np.ceil(np.sqrt(K)))
        lon_bins = np.linspace(proc.lons.min(), proc.lons.max(), grid + 1)
        lat_bins = np.linspace(proc.lats.min(), proc.lats.max(), grid + 1)
        ranked = []
        for i in range(grid):
            for j in range(grid):
                lon_c = 0.5 * (lon_bins[i] + lon_bins[i + 1])
                lat_c = 0.5 * (lat_bins[j] + lat_bins[j + 1])
                dists = haversine_km(lon_c, lat_c, proc.lons, proc.lats)
                ranked.append(int(np.argmin(dists)))
        return _pick_by_rank(proc, list(dict.fromkeys(ranked)), K, use_spacing=use_spacing)
    
    def baseline_stage_only(proc, K, use_spacing=True):
        ranked = np.argsort(-proc.stage_suit).astype(int).tolist()
        return _pick_by_rank(proc, ranked, K, use_spacing=use_spacing)
    
    def baseline_discharge_only(proc, K, use_spacing=True):
        ranked = np.argsort(-proc.discharge_suit).astype(int).tolist()
        return _pick_by_rank(proc, ranked, K, use_spacing=use_spacing)
    
    def eval_method(procs, selector_fn, n_trials=1, fp_w=fp_weight):
        tp_list, fp_list, j_list = [], [], []
        for proc in procs:
            Kb = int(proc.performance['K'])
            for t in range(n_trials):
                sel = selector_fn(proc, Kb, t)
                tp, fp, j = proc.evaluate_selection(sel, fp_weight=fp_w)
                tp_list.append(tp); fp_list.append(fp); j_list.append(j)
        return np.mean(tp_list), np.mean(fp_list), np.mean(j_list), np.std(j_list)
    
    tp_r, fp_r, j_r, jstd_r = eval_method(eval_procs, lambda p, K, t: baseline_random(p, K, 1000+17*t, BASELINES_USE_SPACING), n_trials)
    tp_g, fp_g, j_g, jstd_g = eval_method(eval_procs, lambda p, K, t: baseline_uniform_grid(p, K, BASELINES_USE_SPACING), 1)
    tp_s, fp_s, j_s, jstd_s = eval_method(eval_procs, lambda p, K, t: baseline_stage_only(p, K, BASELINES_USE_SPACING), 1)
    tp_d, fp_d, j_d, jstd_d = eval_method(eval_procs, lambda p, K, t: baseline_discharge_only(p, K, BASELINES_USE_SPACING), 1)
    
    tp_ours = np.mean([p.performance['TP_rate'] for p in eval_procs])
    fp_ours = np.mean([p.performance['FP_rate'] for p in eval_procs])
    j_ours = np.mean([p.performance['J'] for p in eval_procs])
    jstd_ours = np.std([p.performance['J'] for p in eval_procs])
    
    baseline_df = pd.DataFrame({
        'Scenario': [scenario_name] * 5,
        'Method': ['Random', 'Uniform Grid', 'Stage-only', 'Discharge-only', 'Multi-Sensor (Ours)'],
        'TP_Rate_mean': [tp_r, tp_g, tp_s, tp_d, tp_ours],
        'FP_Rate_mean': [fp_r, fp_g, fp_s, fp_d, fp_ours],
        'Score_J_mean': [j_r, j_g, j_s, j_d, j_ours],
        'Score_J_std': [jstd_r, jstd_g, jstd_s, jstd_d, jstd_ours]
    })
    
    return baseline_df


# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    print("\n STEP 1: Loading data...")

    df_full = pd.read_csv(MASTER_F)
    if 'OBJECTID' not in df_full.columns and 'ORIG_FID' in df_full.columns:
        df_full['OBJECTID'] = df_full['ORIG_FID']
    df_full = df_full.dropna(subset=['lon', 'lat', 'flood_events_norm', 'flood_risk_value_norm'])
    df_full = df_full.reset_index(drop=True)
    N_full = len(df_full)
    print(f"  [OK] Loaded {N_full:,} candidate points")

    basin_info, gdf_basins = load_huc10_info(HUC10_SHAPEFILE, HUC10_ID_COLUMN)

    print("\n  STEP 2: Assigning points to HUC10 basins...")
    huc_col_in_points = None
    if POINTS_HUC_COLUMN and POINTS_HUC_COLUMN in df_full.columns:
        huc_col_in_points = POINTS_HUC_COLUMN
    else:
        for col in ['huc10', 'HUC10', 'huc_10', 'HUC_10']:
            if col in df_full.columns:
                huc_col_in_points = col
                break

    if huc_col_in_points:
        df_full['_huc10_'] = df_full[huc_col_in_points].astype(str)
        print(f"  Using existing column: '{huc_col_in_points}'")
    elif basin_info:
        df_full['_huc10_'] = 'unknown'
    else:
        raise ValueError("No HUC10 info available.")

    unique_hucs = df_full['_huc10_'].dropna().unique()
    n_basins = len(unique_hucs)
    print(f"\n  Found {n_basins} unique HUC10 basins")

    print("\n STEP 3: Cleaning sentinel values...")
    COLUMNS_TO_CLEAN = [
        'nhdplusflowlinevaa_nc_slope', 'slope_deg', 'elevation_m', 'twi',
        'flowacc', 'nhd_totdasqkm', 'nhd_streamorder', 'nhd_streamlevel',
        'rain_mean', 'LakeFract', 'minelevraw', 'maxelevsmo', 'minelevsmo'
    ]
    for col in COLUMNS_TO_CLEAN:
        if col in df_full.columns:
            df_full[col] = clean_and_winsorize(df_full[col].to_numpy(float))
    print("  [OK] Cleaned")

    # ================== RUN ALL SCENARIOS ==================
    print("\n" + "="*80)
    print("STEP 4: RUNNING FOUR OPERATIONAL SCENARIOS")
    print("="*80)

    scenario_results = {}
    for scenario_key, scenario_config in SCENARIOS.items():
        scenario_results[scenario_key] = run_scenario(
            df_full, basin_info, unique_hucs, scenario_key, scenario_config
        )

    # ================== COMBINE AND SAVE ALL SCENARIO OUTPUTS ==================
    print("\n STEP 5: Saving ALL scenario outputs...")

    # Combine all selected sites with scenario column
    all_selected_combined = []
    all_summaries_combined = []

    for scenario_key, results in scenario_results.items():
        if len(results['all_selected_df']) > 0:
            all_selected_combined.append(results['all_selected_df'])
        if len(results['basin_summary_df']) > 0:
            all_summaries_combined.append(results['basin_summary_df'])

    # Save combined selected sites (all scenarios)
    if all_selected_combined:
        combined_selected_df = pd.concat(all_selected_combined, ignore_index=True)
        combined_selected_df.to_csv(OUT_SEL_ALL, index=False)
        print(f"  [OK] {OUT_SEL_ALL} ({len(combined_selected_df):,} sensors across all scenarios)")

    # Save combined basin summaries (all scenarios)
    if all_summaries_combined:
        combined_summary_df = pd.concat(all_summaries_combined, ignore_index=True)
        combined_summary_df.to_csv(OUT_BASIN_SUMMARY_ALL, index=False)
        print(f"  [OK] {OUT_BASIN_SUMMARY_ALL}")

    # Save individual scenario CSVs
    for scenario_key, results in scenario_results.items():
        if len(results['all_selected_df']) > 0:
            out_path = os.path.join(OUT_DIR, f"selected_sites_{scenario_key}.csv")
            results['all_selected_df'].to_csv(out_path, index=False)
            print(f"  [OK] {out_path}")

    # ================== CREATE AND PRINT COMPARISON TABLE ==================
    print("\n" + "="*80)
    print("TABLE 1: Network Configuration Under Alternative Operational Scenarios")
    print("="*80)
    comparison_table = create_scenario_comparison_table(scenario_results)
    print(comparison_table.to_string(index=False))
    comparison_table.to_csv(OUT_SCENARIO_TABLE, index=False)
    print(f"\n  [OK] Saved to: {OUT_SCENARIO_TABLE}")

    # ================== COMPUTE BASELINES FOR ALL SCENARIOS ==================
    print("\n" + "="*80)
    print("STEP 6: Computing baselines for ALL scenarios...")
    print("="*80)

    all_baseline_results = []
    baseline_results_by_scenario = {}

    for scenario_key, results in scenario_results.items():
        config = results['config']
        processors = results['processors']
        fp_weight = config['fp_weight']
        
        print(f"\n  Computing baselines for {config['name']}...")
        baseline_df = compute_baselines_for_scenario(
            processors, fp_weight, config['name'],
            eval_max_basins=EVAL_MAX_BASINS, n_trials=BASELINE_N_TRIALS
        )
        
        if len(baseline_df) > 0:
            all_baseline_results.append(baseline_df)
            baseline_results_by_scenario[scenario_key] = baseline_df
            
            # Save individual scenario baseline
            out_path = os.path.join(OUT_DIR, f"baselines_{scenario_key}.csv")
            baseline_df.to_csv(out_path, index=False)
            print(f"    [OK] Saved to: {out_path}")

    # Combine all baselines
    if all_baseline_results:
        all_baselines_df = pd.concat(all_baseline_results, ignore_index=True)
        all_baselines_df.to_csv(OUT_BASELINES_ALL, index=False)
        print(f"\n  [OK] Combined baselines saved to: {OUT_BASELINES_ALL}")

    # ================== PRINT BASELINE COMPARISON TABLES ==================
    print("\n" + "="*80)
    print("TABLE 2: Method Comparison by Scenario (Multi-Sensor vs Baselines)")
    print("="*80)

    for scenario_key, baseline_df in baseline_results_by_scenario.items():
        config = SCENARIOS[scenario_key]
        print(f"\n{'-'*60}")
        print(f"Scenario: {config['name']} (lambda={config['fp_weight']}, tau={config['cost_fraction']})")
        print(f"{'-'*60}")
        
        # Format the table nicely
        display_df = baseline_df[['Method', 'TP_Rate_mean', 'FP_Rate_mean', 'Score_J_mean', 'Score_J_std']].copy()
        display_df['TP_Rate'] = display_df['TP_Rate_mean'].apply(lambda x: f"{x:.1%}")
        display_df['FP_Rate'] = display_df['FP_Rate_mean'].apply(lambda x: f"{x:.1%}")
        display_df['J_Score'] = display_df['Score_J_mean'].apply(lambda x: f"{x:.3f}")
        display_df['J_Std'] = display_df['Score_J_std'].apply(lambda x: f"+/-{x:.3f}")
        
        print(display_df[['Method', 'TP_Rate', 'FP_Rate', 'J_Score', 'J_Std']].to_string(index=False))
        
        # Calculate improvement over baselines
        ours_j = baseline_df[baseline_df['Method'] == 'Multi-Sensor (Ours)']['Score_J_mean'].values[0]
        random_j = baseline_df[baseline_df['Method'] == 'Random']['Score_J_mean'].values[0]
        grid_j = baseline_df[baseline_df['Method'] == 'Uniform Grid']['Score_J_mean'].values[0]
        stage_j = baseline_df[baseline_df['Method'] == 'Stage-only']['Score_J_mean'].values[0]
        discharge_j = baseline_df[baseline_df['Method'] == 'Discharge-only']['Score_J_mean'].values[0]
        
        print(f"\n  Improvement over baselines:")
        print(f"    vs Random:       +{((ours_j - random_j) / abs(random_j) * 100) if random_j != 0 else 0:.1f}%")
        print(f"    vs Uniform Grid: +{((ours_j - grid_j) / abs(grid_j) * 100) if grid_j != 0 else 0:.1f}%")
        print(f"    vs Stage-only:   +{((ours_j - stage_j) / abs(stage_j) * 100) if stage_j != 0 else 0:.1f}%")
        print(f"    vs Discharge:    +{((ours_j - discharge_j) / abs(discharge_j) * 100) if discharge_j != 0 else 0:.1f}%")

    # ================== SENSITIVITY ANALYSIS - ALL SCENARIOS ==================
    print("\n" + "="*80)
    print("STEP 7: Sensitivity analysis (all scenarios)...")
    print("="*80)

    sens_mults = [0.7, 0.85, 1.0, 1.15, 1.3]
    sens_data = {key: [] for key in scenario_results.keys()}

    for scenario_key, results in scenario_results.items():
        procs = list(results['processors'].values())[:min(10, len(results['processors']))]
        fp_w = results['config']['fp_weight']
        if procs:
            for mult in sens_mults:
                all_j = [proc.evaluate_selection(proc.selected_indices, radius_mult=mult, fp_weight=fp_w)[2] for proc in procs]
                all_tp = [proc.evaluate_selection(proc.selected_indices, radius_mult=mult, fp_weight=fp_w)[0] for proc in procs]
                all_fp = [proc.evaluate_selection(proc.selected_indices, radius_mult=mult, fp_weight=fp_w)[1] for proc in procs]
                sens_data[scenario_key].append({'scenario': scenario_key, 'multiplier': mult, 'tp_rate': np.mean(all_tp), 'fp_rate': np.mean(all_fp), 'score': np.mean(all_j)})

    # Save sensitivity for all scenarios
    all_sens = []
    for key, data in sens_data.items():
        all_sens.extend(data)
    sens_df = pd.DataFrame(all_sens)
    sens_df.to_csv(f"{OUT_DIAG}_sensitivity_all.csv", index=False)

    # Print sensitivity table
    print("\nTABLE 3: Detection Radius Sensitivity Analysis")
    print("="*60)
    for scenario_key, data in sens_data.items():
        if data:
            config = SCENARIOS[scenario_key]
            print(f"\n{config['short_name']}:")
            print(f"  {'Multiplier':>10} {'TP Rate':>10} {'FP Rate':>10} {'J Score':>10}")
            print(f"  {'-'*42}")
            for row in data:
                print(f"  {row['multiplier']:>10.2f} {row['tp_rate']:>10.1%} {row['fp_rate']:>10.1%} {row['score']:>10.3f}")

    print(f"\n  [OK] Saved to: {OUT_DIAG}_sensitivity_all.csv")

    # ================== PARETO CURVES - ALL SCENARIOS ==================
    print("\n" + "="*80)
    print("STEP 8: Computing Pareto curves (all scenarios)...")
    print("="*80)

    K_values = [5, 10, 15, 20, 30, 50, 75, 100]
    pareto_data = {key: [] for key in scenario_results.keys()}

    def greedy_select_fixedK(proc, K_target, fp_weight):
        K_target = int(min(K_target, proc.N))
        selected = []
        p_stage, p_q, p_cam = np.zeros(proc.N), np.zeros(proc.N), np.zeros(proc.N)
        for _ in range(K_target):
            best_c, best_J = None, -1e18
            best_p_s = best_p_q = best_p_c = None
            for c in range(proc.N):
                if c in selected or not proc.spacing_ok(c, selected):
                    continue
                p_s_new, p_q_new, p_c_new = proc.compute_candidate_contribution(c)
                p_s_upd = np.maximum(p_stage, p_s_new)
                p_q_upd = np.maximum(p_q, p_q_new)
                p_c_upd = np.maximum(p_cam, p_c_new)
                _, _, J_new = proc.compute_network_score(p_s_upd, p_q_upd, p_c_upd, fp_weight)
                if J_new > best_J:
                    best_J, best_c = J_new, c
                    best_p_s, best_p_q, best_p_c = p_s_upd, p_q_upd, p_c_upd
            if best_c is None:
                break
            selected.append(int(best_c))
            p_stage, p_q, p_cam = best_p_s, best_p_q, best_p_c
        return selected

    for scenario_key, results in scenario_results.items():
        procs = list(results['processors'].values())[:min(10, len(results['processors']))]
        fp_weight = results['config']['fp_weight']
        if procs:
            for K_target in K_values:
                tp_list, fp_list, j_list = [], [], []
                for proc in procs:
                    sel = greedy_select_fixedK(proc, K_target, fp_weight)
                    if len(sel) > 0:
                        tp, fp, j = proc.evaluate_selection(sel, fp_weight=fp_weight)
                        tp_list.append(tp); fp_list.append(fp); j_list.append(j)
                if tp_list:
                    pareto_data[scenario_key].append({'scenario': scenario_key, 'K': K_target, 'TP': np.mean(tp_list), 'FP': np.mean(fp_list), 'Score': np.mean(j_list)})

    # Save pareto for all scenarios
    all_pareto = []
    for key, data in pareto_data.items():
        all_pareto.extend(data)
    pareto_df = pd.DataFrame(all_pareto)
    pareto_df.to_csv(f"{OUT_DIAG}_pareto_all.csv", index=False)

    # Print Pareto table
    print("\nTABLE 4: Pareto Curves (TP-FP Trade-off by Network Size)")
    print("="*60)
    for scenario_key, data in pareto_data.items():
        if data:
            config = SCENARIOS[scenario_key]
            print(f"\n{config['short_name']}:")
            print(f"  {'K':>5} {'TP Rate':>10} {'FP Rate':>10} {'J Score':>10}")
            print(f"  {'-'*37}")
            for row in data:
                print(f"  {row['K']:>5} {row['TP']:>10.1%} {row['FP']:>10.1%} {row['Score']:>10.3f}")

    print(f"\n  [OK] Saved to: {OUT_DIAG}_pareto_all.csv")

    # ================== SUBMODULARITY CHECK ==================
    print("\n" + "="*80)
    print("STEP 9: Checking submodularity...")
    print("="*80)

    submod_results = {}
    for scenario_key, results in scenario_results.items():
        all_gains = [h['marginal_gain'] for h in results['all_history']]
        if len(all_gains) > 1:
            decreasing_count = sum(1 for i in range(len(all_gains)-1) if all_gains[i] >= all_gains[i+1])
            pct_decreasing = decreasing_count / (len(all_gains)-1)
            is_decreasing = pct_decreasing > 0.7
            submod_results[scenario_key] = {'passed': is_decreasing, 'pct_decreasing': pct_decreasing}
        else:
            submod_results[scenario_key] = {'passed': True, 'pct_decreasing': 1.0}

    print("\nTABLE 5: Submodularity Verification")
    print("="*60)
    print(f"  {'Scenario':<20} {'Passed':>10} {'% Decreasing':>15}")
    print(f"  {'-'*45}")
    for scenario_key, result in submod_results.items():
        config = SCENARIOS[scenario_key]
        status = "[OK]" if result['passed'] else "[FAIL]"
        print(f"  {config['short_name']:<20} {status:>10} {result['pct_decreasing']:>14.1%}")

    # ================== INTER-SENSOR DISTANCES - ALL SCENARIOS ==================
    print("\n" + "="*80)
    print("STEP 10: Computing inter-sensor distances (all scenarios)...")
    print("="*80)

    dist_stats = {}
    for scenario_key, results in scenario_results.items():
        sel_df = results['all_selected_df']
        if len(sel_df) >= 2:
            lons_sel = sel_df['lon'].to_numpy()
            lats_sel = sel_df['lat'].to_numpy()
            n_sel = len(sel_df)
            sample_idx = np.random.choice(n_sel, min(500, n_sel), replace=False)
            dists = [haversine_km(lons_sel[sample_idx[i]], lats_sel[sample_idx[i]], lons_sel[sample_idx[j]], lats_sel[sample_idx[j]])
                     for i in range(len(sample_idx)) for j in range(i+1, len(sample_idx))]
            dist_stats[scenario_key] = {
                'median': np.median(dists) if dists else 0,
                'min': np.min(dists) if dists else 0,
                'max': np.max(dists) if dists else 0,
                'mean': np.mean(dists) if dists else 0,
                'std': np.std(dists) if dists else 0
            }
        else:
            dist_stats[scenario_key] = {'median': 0, 'min': 0, 'max': 0, 'mean': 0, 'std': 0}

    print("\nTABLE 6: Inter-Sensor Distance Statistics (km)")
    print("="*70)
    print(f"  {'Scenario':<15} {'Min':>8} {'Mean':>8} {'Median':>8} {'Max':>8} {'Std':>8}")
    print(f"  {'-'*55}")
    for scenario_key, stats in dist_stats.items():
        config = SCENARIOS[scenario_key]
        print(f"  {config['short_name']:<15} {stats['min']:>8.1f} {stats['mean']:>8.1f} {stats['median']:>8.1f} {stats['max']:>8.1f} {stats['std']:>8.1f}")

    # Save distance stats
    dist_stats_df = pd.DataFrame([
        {
            'Scenario': SCENARIOS[k]['name'],
            'Min_km': v['min'],
            'Mean_km': v['mean'],
            'Median_km': v['median'],
            'Max_km': v['max'],
            'Std_km': v['std']
        }
        for k, v in dist_stats.items()
    ])
    dist_stats_df.to_csv(f"{OUT_DIAG}_distance_stats.csv", index=False)
    print(f"\n  [OK] Saved to: {OUT_DIAG}_distance_stats.csv")

    # ================== PER-BASIN STATISTICS ==================
    print("\n" + "="*80)
    print("TABLE 7: Per-Basin Performance Statistics")
    print("="*80)

    for scenario_key, results in scenario_results.items():
        config = results['config']
        summary_df = results['basin_summary_df']
        
        if len(summary_df) > 0:
            print(f"\n{'-'*60}")
            print(f"Scenario: {config['name']}")
            print(f"{'-'*60}")
            
            print(f"\n  Metric               Min      Mean    Median       Max       Std")
            print(f"  {'-'*60}")
            
            for col, label in [('TP_rate', 'TP Rate'), ('FP_rate', 'FP Rate'), 
                               ('J', 'J Score'), ('n_selected', 'Sensors'),
                               ('basin_area_km2', 'Area (km^2)'), 
                               ('density_km2_per_sensor', 'Density (km^2/sensor)')]:
                if col in summary_df.columns:
                    print(f"  {label:<18} {summary_df[col].min():>8.3f} {summary_df[col].mean():>8.3f} {summary_df[col].median():>8.3f} {summary_df[col].max():>8.3f} {summary_df[col].std():>8.3f}")

    # ================== SUITABILITY STATISTICS ==================
    print("\n" + "="*80)
    print("TABLE 8: Selected Site Suitability Statistics")
    print("="*80)

    for scenario_key, results in scenario_results.items():
        config = results['config']
        sel_df = results['all_selected_df']
        
        if len(sel_df) > 0:
            print(f"\n{'-'*60}")
            print(f"Scenario: {config['name']} (N={len(sel_df)} sensors)")
            print(f"{'-'*60}")
            
            print(f"\n  Suitability Type     Min      Mean    Median       Max       Std")
            print(f"  {'-'*60}")
            
            for col, label in [('stage_suitability', 'Stage'), 
                               ('discharge_suitability', 'Discharge'),
                               ('camera_suitability', 'Camera'),
                               ('risk_in_basin', 'Risk')]:
                if col in sel_df.columns:
                    print(f"  {label:<18} {sel_df[col].min():>8.3f} {sel_df[col].mean():>8.3f} {sel_df[col].median():>8.3f} {sel_df[col].max():>8.3f} {sel_df[col].std():>8.3f}")

    # ================== RISK TIER DISTRIBUTION ==================
    print("\n" + "="*80)
    print("TABLE 9: Sensor Distribution by Risk Tier")
    print("="*80)

    print(f"\n  {'Scenario':<15}", end="")
    for label in BIN_LABELS:
        print(f" {label:>12}", end="")
    print(f" {'Total':>10}")
    print(f"  {'-'*85}")

    for scenario_key, results in scenario_results.items():
        config = results['config']
        sel_df = results['all_selected_df']
        
        if len(sel_df) > 0:
            risk_sel = minmax(0.6 * sel_df['flood_events_norm'].to_numpy() + 0.4 * sel_df['flood_risk_value_norm'].to_numpy())
            risk_bins = np.clip(np.digitize(risk_sel, BIN_EDGES, right=False) - 1, 0, nb - 1)
            
            counts = [int((risk_bins == b).sum()) for b in range(nb)]
            
            print(f"  {config['short_name']:<15}", end="")
            for count in counts:
                print(f" {count:>12}", end="")
            print(f" {sum(counts):>10}")

    # ================== SUMMARY STATISTICS TABLE ==================
    print("\n" + "="*80)
    print("TABLE 10: Summary Statistics Across All Scenarios")
    print("="*80)

    # Get total area
    total_area = float(scenario_results['B_Balanced']['basin_summary_df']['basin_area_km2'].sum()) if len(scenario_results['B_Balanced']['basin_summary_df']) > 0 else 0.0

    summary_rows = []
    for scenario_key, results in scenario_results.items():
        config = results['config']
        stats = results['stats']
        summary_df = results['basin_summary_df']
        
        row = {
            'Scenario': config['short_name'],
            'lambda': config['fp_weight'],
            'tau': config['cost_fraction'],
            'Total_Sensors': stats['n_sensors'],
            'N_Basins': stats['n_basins'],
            'Sensors_per_Basin': stats['sensors_per_basin'],
            'TP_Rate': stats['TP_rate'],
            'FP_Rate': stats['FP_rate'],
            'J_Score': stats['J_score'],
            'Med_Inter_Sensor_km': dist_stats[scenario_key]['median'],
            'Elapsed_min': stats['elapsed_min']
        }
        
        if len(summary_df) > 0:
            row['Mean_Basin_J'] = summary_df['J'].mean()
            row['Std_Basin_J'] = summary_df['J'].std()
        
        summary_rows.append(row)

    summary_stats_df = pd.DataFrame(summary_rows)
    summary_stats_df.to_csv(f"{OUT_DIAG}_summary_stats.csv", index=False)

    print(f"\n  Study Area: {total_area:,.0f} km^2 | {n_basins} basins")
    print(f"\n  {'Scenario':<12} {'lambda':>5} {'tau':>5} {'Sensors':>8} {'TP':>8} {'FP':>8} {'J':>8} {'Dist(km)':>10}")
    print(f"  {'-'*70}")
    for row in summary_rows:
        print(f"  {row['Scenario']:<12} {row['lambda']:>5.2f} {row['tau']:>5.2f} {row['Total_Sensors']:>8} {row['TP_Rate']:>7.1%} {row['FP_Rate']:>7.1%} {row['J_Score']:>8.3f} {row['Med_Inter_Sensor_km']:>10.1f}")

    print(f"\n  [OK] Saved to: {OUT_DIAG}_summary_stats.csv")

    # ================== VISUALIZATION ==================
    print("\n" + "="*80)
    print("STEP 11: Creating visualizations...")
    print("="*80)

    # Compute global risk for background
    risk_all = minmax(0.6 * df_full['flood_events_norm'].to_numpy() + 0.4 * df_full['flood_risk_value_norm'].to_numpy())

    # ==================== FIGURE 1: SENSOR MAPS FOR ALL SCENARIOS ====================
    fig_maps = plt.figure(figsize=(20, 16))
    gs_maps = fig_maps.add_gridspec(2, 2, hspace=0.25, wspace=0.2)

    for idx, (scenario_key, results) in enumerate(scenario_results.items()):
        ax = fig_maps.add_subplot(gs_maps[idx // 2, idx % 2])
        config = results['config']
        sel_df = results['all_selected_df']
        stats = results['stats']
        
        # Background risk
        sc = ax.scatter(df_full['lon'], df_full['lat'], c=risk_all, s=8, alpha=0.3, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Basin boundaries
        if gdf_basins is not None:
            try:
                gdf_basins.boundary.plot(ax=ax, color='navy', linewidth=0.6, alpha=0.4)
            except:
                pass
        
        # Sensors
        if len(sel_df) > 0:
            lons_sel = sel_df['lon'].to_numpy()
            lats_sel = sel_df['lat'].to_numpy()
            
            # Compute risk bins for this scenario
            risk_sel = minmax(0.6 * sel_df['flood_events_norm'].to_numpy() + 0.4 * sel_df['flood_risk_value_norm'].to_numpy())
            risk_bins = np.clip(np.digitize(risk_sel, BIN_EDGES, right=False) - 1, 0, nb - 1)
            
            for b in range(nb):
                mask = risk_bins == b
                if mask.sum() > 0:
                    ax.scatter(lons_sel[mask], lats_sel[mask], c=TIER_COLORS[b], s=120, marker='*',
                               edgecolors='black', linewidths=1, label=f"{BIN_LABELS[b]} (n={mask.sum()})", zorder=5)
        
        ax.set_title(f"{config['name']}\nN={stats['n_sensors']} | TP={stats['TP_rate']:.1%} | FP={stats['FP_rate']:.1%} | J={stats['J_score']:.3f}",
                     fontsize=12, fontweight='bold', color=config['color'])
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(alpha=0.3, linestyle=':')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    plt.colorbar(sc, ax=fig_maps.axes, label='Flood Risk', fraction=0.02, pad=0.02)
    fig_maps.suptitle('OPTIMIZED MULTI-SENSOR NETWORKS - All Scenarios', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(OUT_FIG_MAPS_ALL, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] {OUT_FIG_MAPS_ALL}")

    # ==================== FIGURE 2: MAIN ANALYSIS ====================
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # 1) Combined Map (all scenarios overlaid)
    ax1 = fig.add_subplot(gs[0:2, :2])
    sc = ax1.scatter(df_full['lon'], df_full['lat'], c=risk_all, s=8, alpha=0.3, cmap='YlOrRd', vmin=0, vmax=1)
    if gdf_basins is not None:
        try:
            gdf_basins.boundary.plot(ax=ax1, color='navy', linewidth=0.6, alpha=0.4)
        except:
            pass

    # Plot sensors for each scenario with different markers
    for scenario_key, results in scenario_results.items():
        config = results['config']
        sel_df = results['all_selected_df']
        if len(sel_df) > 0:
            ax1.scatter(sel_df['lon'], sel_df['lat'], c=config['color'], s=60, marker=config['marker'],
                        edgecolors='black', linewidths=0.5, alpha=0.7, label=f"{config['short_name']} (n={len(sel_df)})", zorder=5)

    plt.colorbar(sc, ax=ax1, label='Flood Risk', fraction=0.04)
    ax1.set_title(f'All Scenarios Overlaid | Area {total_area:,.0f} km^2', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude')
    ax1.grid(alpha=0.3, linestyle=':'); ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # 2) Method Comparison - Using Balanced scenario as representative
    ax2 = fig.add_subplot(gs[0, 2])
    if 'B_Balanced' in baseline_results_by_scenario and len(baseline_results_by_scenario['B_Balanced']) > 0:
        baseline_df = baseline_results_by_scenario['B_Balanced']
        methods = baseline_df['Method'].values
        vals = baseline_df['Score_J_mean'].values
        colors_bar = ['#95a5a6', '#95a5a6', '#95a5a6', '#95a5a6', '#3498db']
        bars = ax2.barh(np.arange(len(methods)), vals, color=colors_bar, edgecolor='black', linewidth=1.2)
        ax2.set_yticks(np.arange(len(methods))); ax2.set_yticklabels(methods)
        ax2.set_xlabel('Objective Score (J) - mean')
        ax2.set_title('Method Comparison (Balanced)', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle=':')
        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax2.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    # 3) Detection Performance - ALL SCENARIOS
    ax3 = fig.add_subplot(gs[1, 2])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        hist_df = pd.DataFrame(results['all_history']) if results['all_history'] else pd.DataFrame()
        if not hist_df.empty and 'iteration' in hist_df.columns:
            hist_agg = hist_df.groupby('iteration').agg(tp_mean=('tp_rate', 'mean')).reset_index()
            ax3.plot(hist_agg['iteration'], hist_agg['tp_mean'], marker=config['marker'], color=config['color'],
                     lw=2, markersize=4, label=config['short_name'])
    ax3.set_xlabel('Number of Sensors'); ax3.set_ylabel('TP Rate (mean)')
    ax3.set_title('Detection Performance by Scenario', fontweight='bold')
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3, linestyle=':')

    # 4) Diminishing Returns - ALL SCENARIOS
    ax4 = fig.add_subplot(gs[2, 0])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        hist = results['all_history']
        if hist:
            gains = [h['marginal_gain'] for h in hist[:50]]
            ax4.plot(range(len(gains)), gains, color=config['color'], lw=1.5, alpha=0.8, label=config['short_name'])
    ax4.axhline(0, color='r', linestyle='--', alpha=0.6, linewidth=2)
    ax4.set_xlabel('Sensor Rank'); ax4.set_ylabel('DeltaJ')
    ax4.set_title('Diminishing Returns by Scenario', fontweight='bold')
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3, linestyle=':')

    # 5) Suitability - ALL SCENARIOS
    ax5 = fig.add_subplot(gs[2, 1])
    x_suit = np.arange(3)
    width = 0.2
    for i, (scenario_key, results) in enumerate(scenario_results.items()):
        config = results['config']
        sel_df = results['all_selected_df']
        if len(sel_df) > 0:
            means = [sel_df['stage_suitability'].mean(), sel_df['discharge_suitability'].mean(), sel_df['camera_suitability'].mean()]
            ax5.bar(x_suit + i*width, means, width, label=config['short_name'], color=config['color'], edgecolor='black', linewidth=0.8)
    ax5.set_xticks(x_suit + width*1.5); ax5.set_xticklabels(['Stage', 'Discharge', 'Camera'])
    ax5.set_ylim(0, 1); ax5.set_ylabel('Suitability')
    ax5.set_title('Selected-Site Suitability by Scenario', fontweight='bold')
    ax5.legend(fontsize=8); ax5.grid(axis='y', alpha=0.3, linestyle=':')

    # 6) Risk Tier - ALL SCENARIOS
    ax6 = fig.add_subplot(gs[2, 2])
    x_tier = np.arange(nb)
    width = 0.2
    for i, (scenario_key, results) in enumerate(scenario_results.items()):
        config = results['config']
        sel_df = results['all_selected_df']
        if len(sel_df) > 0:
            risk_sel = minmax(0.6 * sel_df['flood_events_norm'].to_numpy() + 0.4 * sel_df['flood_risk_value_norm'].to_numpy())
            risk_bins = np.clip(np.digitize(risk_sel, BIN_EDGES, right=False) - 1, 0, nb - 1)
            counts = [int((risk_bins == b).sum()) for b in range(nb)]
            ax6.bar(x_tier + i*width, counts, width, label=config['short_name'], color=config['color'], edgecolor='black', linewidth=0.8)
    ax6.set_xticks(x_tier + width*1.5); ax6.set_xticklabels(BIN_LABELS, rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel('Sensors'); ax6.set_title('Sensors per Risk Tier by Scenario', fontweight='bold')
    ax6.legend(fontsize=8); ax6.grid(axis='y', alpha=0.3, linestyle=':')

    # 7) Pareto - ALL SCENARIOS
    ax7 = fig.add_subplot(gs[3, 0])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        pareto_df_scenario = pd.DataFrame(pareto_data[scenario_key])
        if len(pareto_df_scenario) > 0:
            ax7.plot(pareto_df_scenario['FP'], pareto_df_scenario['TP'], marker=config['marker'], color=config['color'],
                     linewidth=2, markersize=6, markeredgecolor='black', markeredgewidth=0.8, label=config['short_name'])
    ax7.set_xlabel('False Positive Rate'); ax7.set_ylabel('True Positive Rate')
    ax7.set_title('TP-FP Tradeoff vs K by Scenario', fontweight='bold')
    ax7.legend(fontsize=8); ax7.grid(alpha=0.3, linestyle=':')

    # 8) Radius Sensitivity - ALL SCENARIOS
    ax8 = fig.add_subplot(gs[3, 1])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        sens_df_scenario = pd.DataFrame(sens_data[scenario_key])
        if len(sens_df_scenario) > 0:
            ax8.plot(sens_df_scenario['multiplier'], sens_df_scenario['score'], marker=config['marker'], color=config['color'],
                     lw=2, markersize=6, label=config['short_name'])
    ax8.axvline(1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax8.set_xlabel('Detection Radius Multiplier'); ax8.set_ylabel('Score (J)')
    ax8.set_title('Radius Sensitivity by Scenario', fontweight='bold')
    ax8.legend(fontsize=8); ax8.grid(alpha=0.3, linestyle=':')

    # 9) Summary
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis('off')

    summary = f"""
NETWORK PERFORMANCE SUMMARY (v4.4-S)
========================================

Study Area: {total_area:,.0f} km^2 | {n_basins} basins

SCENARIO RESULTS:
"""
    for scenario_key, results in scenario_results.items():
        cfg = results['config']
        s = results['stats']
        d = dist_stats[scenario_key]
        summary += f"""
{cfg['short_name']:12s}: {s['n_sensors']:3d} sensors
  TP={s['TP_rate']:.1%}  FP={s['FP_rate']:.1%}  J={s['J_score']:.3f}
  Inter-sensor: {d['median']:.1f} km (median)"""

    if 'B_Balanced' in baseline_results_by_scenario and len(baseline_results_by_scenario['B_Balanced']) > 0:
        baseline_balanced = baseline_results_by_scenario['B_Balanced']
        summary += f"""

BASELINES (Balanced Scenario):
  Random:    {baseline_balanced.loc[baseline_balanced['Method']=='Random','Score_J_mean'].values[0]:.3f}
  Grid:      {baseline_balanced.loc[baseline_balanced['Method']=='Uniform Grid','Score_J_mean'].values[0]:.3f}
  Stage:     {baseline_balanced.loc[baseline_balanced['Method']=='Stage-only','Score_J_mean'].values[0]:.3f}
  Discharge: {baseline_balanced.loc[baseline_balanced['Method']=='Discharge-only','Score_J_mean'].values[0]:.3f}
  Ours:      {baseline_balanced.loc[baseline_balanced['Method']=='Multi-Sensor (Ours)','Score_J_mean'].values[0]:.3f}
"""

    ax9.text(0.02, 0.98, summary, transform=ax9.transAxes, fontsize=8, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=0.5))

    fig.suptitle('BASIN-BY-BASIN MULTI-SENSOR FLOOD DETECTION NETWORK (v4.4-S) - All Scenarios',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(OUT_FIG_MAIN, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] {OUT_FIG_MAIN}")

    # ==================== FIGURE 3: SCENARIO COMPARISON ====================
    fig2 = plt.figure(figsize=(16, 10))
    gs2 = fig2.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    scenario_names = [SCENARIOS[k]['short_name'] for k in scenario_results.keys()]
    colors_list = [SCENARIOS[k]['color'] for k in scenario_results.keys()]

    # 1) Network Size
    ax_s1 = fig2.add_subplot(gs2[0, 0])
    n_sensors_list = [r['stats']['n_sensors'] for r in scenario_results.values()]
    bars = ax_s1.bar(scenario_names, n_sensors_list, color=colors_list, edgecolor='black', linewidth=1.5)
    ax_s1.set_ylabel('Total Sensors', fontsize=12)
    ax_s1.set_title('Network Size by Scenario', fontweight='bold', fontsize=14)
    ax_s1.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars, n_sensors_list):
        ax_s1.text(bar.get_x() + bar.get_width()/2, val + 5, str(val), ha='center', fontsize=11, fontweight='bold')

    # 2) TP vs FP
    ax_s2 = fig2.add_subplot(gs2[0, 1])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        stats = results['stats']
        ax_s2.scatter(stats['FP_rate'], stats['TP_rate'], s=200, c=config['color'],
                      marker=config['marker'], edgecolors='black', linewidths=2, label=config['short_name'], zorder=5)
        ax_s2.annotate(f"J={stats['J_score']:.3f}", (stats['FP_rate'], stats['TP_rate']),
                       xytext=(10, 10), textcoords='offset points', fontsize=10)
    ax_s2.set_xlabel('False Positive Rate', fontsize=12); ax_s2.set_ylabel('True Positive Rate', fontsize=12)
    ax_s2.set_title('Detection Performance Trade-off', fontweight='bold', fontsize=14)
    ax_s2.legend(loc='lower right', fontsize=10); ax_s2.grid(alpha=0.3, linestyle=':')

    # 3) J-Score
    ax_s3 = fig2.add_subplot(gs2[0, 2])
    j_scores = [r['stats']['J_score'] for r in scenario_results.values()]
    bars3 = ax_s3.bar(scenario_names, j_scores, color=colors_list, edgecolor='black', linewidth=1.5)
    ax_s3.set_ylabel('J Score (TP - lambdaxFP)', fontsize=12)
    ax_s3.set_title('Objective Score by Scenario', fontweight='bold', fontsize=14)
    ax_s3.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars3, j_scores):
        ax_s3.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    # 4) TP Rate
    ax_s4 = fig2.add_subplot(gs2[1, 0])
    tp_rates = [r['stats']['TP_rate'] for r in scenario_results.values()]
    bars4 = ax_s4.bar(scenario_names, tp_rates, color=colors_list, edgecolor='black', linewidth=1.5)
    ax_s4.set_ylabel('True Positive Rate', fontsize=12)
    ax_s4.set_title('Coverage (TP Rate) by Scenario', fontweight='bold', fontsize=14)
    ax_s4.set_ylim(0, 1); ax_s4.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars4, tp_rates):
        ax_s4.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')

    # 5) FP Rate
    ax_s5 = fig2.add_subplot(gs2[1, 1])
    fp_rates = [r['stats']['FP_rate'] for r in scenario_results.values()]
    bars5 = ax_s5.bar(scenario_names, fp_rates, color=colors_list, edgecolor='black', linewidth=1.5)
    ax_s5.set_ylabel('False Positive Rate', fontsize=12)
    ax_s5.set_title('False Alarms (FP Rate) by Scenario', fontweight='bold', fontsize=14)
    ax_s5.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars5, fp_rates):
        ax_s5.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')

    # 6) Table
    ax_s6 = fig2.add_subplot(gs2[1, 2])
    ax_s6.axis('off')
    table_text = """
+==================================================================+
|  TABLE X: Network Configuration Under Operational Scenarios      |
+==================================================================+
|  Scenario              lambda      tau     Sensors   TP      FP     J   |
+==================================================================+"""
    for key, results in scenario_results.items():
        cfg = results['config']
        s = results['stats']
        name = cfg['short_name'][:14].ljust(14)
        table_text += f"\n|  {name}       {cfg['fp_weight']:.2f}   {cfg['cost_fraction']:.2f}    {s['n_sensors']:4d}   {s['TP_rate']:.1%}  {s['FP_rate']:.1%} {s['J_score']:.3f} |"
    table_text += """
+==================================================================+

KEY INSIGHTS:
- Maximum Coverage: Most sensors, highest detection
- Balanced: Trade-off between coverage and precision  
- Precision-Focused: Fewer false alarms
- Resource-Constrained: Minimum sensors, high-value only
"""
    ax_s6.text(0.05, 0.95, table_text, transform=ax_s6.transAxes, fontsize=9, va='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig2.suptitle('OPERATIONAL SCENARIOS: 2x2 Design (lambda x tau)', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(OUT_FIG_SCENARIOS, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] {OUT_FIG_SCENARIOS}")

    # ==================== FIGURE 4: DIAGNOSTICS - ALL SCENARIOS ====================
    fig3 = plt.figure(figsize=(18, 12))
    gs3 = fig3.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1) Sampling Reduction (use Balanced as representative)
    ax_d1 = fig3.add_subplot(gs3[0, 0])
    balanced_summary = scenario_results['B_Balanced']['basin_summary_df']
    if len(balanced_summary) > 0:
        ax_d1.scatter(balanced_summary['n_original'], balanced_summary['n_after_sampling'], alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        max_val = max(balanced_summary['n_original'].max(), balanced_summary['n_after_sampling'].max())
        ax_d1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No reduction')
        ax_d1.set_xlabel('Original Points'); ax_d1.set_ylabel('After Stratified+QR')
        ax_d1.set_title('Sampling Reduction per Basin', fontweight='bold')
        ax_d1.legend(); ax_d1.grid(alpha=0.3, linestyle=':')

    # 2) Sensors per Basin - ALL SCENARIOS
    ax_d2 = fig3.add_subplot(gs3[0, 1])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        summary_df = results['basin_summary_df']
        if len(summary_df) > 0:
            ax_d2.hist(summary_df['n_selected'], bins=15, alpha=0.4, color=config['color'],
                       label=config['short_name'], edgecolor='black', linewidth=0.5)
    ax_d2.set_xlabel('Sensors per Basin'); ax_d2.set_ylabel('Count')
    ax_d2.set_title('Distribution of Sensors per Basin', fontweight='bold')
    ax_d2.legend(fontsize=8); ax_d2.grid(alpha=0.3, linestyle=':')

    # 3) Detection Radii (same across scenarios)
    ax_d3 = fig3.add_subplot(gs3[0, 2])
    if len(balanced_summary) > 0:
        ax_d3.hist(balanced_summary['detection_radius_discharge'], bins=15, edgecolor='black', alpha=0.7)
        ax_d3.axvline(balanced_summary['detection_radius_discharge'].mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {balanced_summary["detection_radius_discharge"].mean():.1f} km')
        ax_d3.set_xlabel('Detection Radius (km)'); ax_d3.set_ylabel('Count')
        ax_d3.set_title('Basin-Calibrated Detection Radii', fontweight='bold')
        ax_d3.legend(); ax_d3.grid(alpha=0.3, linestyle=':')

    # 4) Basin Performance - ALL SCENARIOS
    ax_d4 = fig3.add_subplot(gs3[1, 0])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        summary_df = results['basin_summary_df']
        if len(summary_df) > 0:
            ax_d4.hist(summary_df['J'], bins=15, alpha=0.4, color=config['color'],
                       label=f"{config['short_name']} (mu={summary_df['J'].mean():.3f})",
                       edgecolor='black', linewidth=0.5)
    ax_d4.set_xlabel('J Score'); ax_d4.set_ylabel('Count')
    ax_d4.set_title('Basin Performance Distribution', fontweight='bold')
    ax_d4.legend(fontsize=8); ax_d4.grid(alpha=0.3, linestyle=':')

    # 5) Basin Area vs Sensors - ALL SCENARIOS
    ax_d5 = fig3.add_subplot(gs3[1, 1])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        summary_df = results['basin_summary_df']
        if len(summary_df) > 0:
            ax_d5.scatter(summary_df['basin_area_km2'], summary_df['n_selected'], alpha=0.5, s=30,
                          c=config['color'], marker=config['marker'], label=config['short_name'])
    ax_d5.set_xlabel('Basin Area (km^2)'); ax_d5.set_ylabel('Sensors Selected')
    ax_d5.set_title('Basin Area vs Sensors', fontweight='bold')
    ax_d5.legend(fontsize=8); ax_d5.grid(alpha=0.3, linestyle=':')

    # 6) Diagnostic Summary
    ax_d6 = fig3.add_subplot(gs3[1, 2])
    ax_d6.axis('off')
    if len(balanced_summary) > 0:
        total_original = int(balanced_summary['n_original'].sum())
        total_after = int(balanced_summary['n_after_sampling'].sum())
        red_pct = 100*(1-total_after/max(total_original,1))
        diag_text = f"""
DIAGNOSTIC SUMMARY (ALL SCENARIOS)
======================================

PROCESSING:
  Basins processed: {len(balanced_summary)}

SAMPLING STATS:
  Original: {total_original:,}
  After S+QR: {total_after:,}
  Reduction: {red_pct:.1f}%

SENSORS PER BASIN:"""
        for scenario_key, results in scenario_results.items():
            cfg = results['config']
            s_df = results['basin_summary_df']
            if len(s_df) > 0:
                diag_text += f"""
  {cfg['short_name']:12s}: {s_df['n_selected'].min():.0f}-{s_df['n_selected'].max():.0f} (mu={s_df['n_selected'].mean():.1f})"""

        diag_text += f"""

DETECTION RADII:
  Min: {balanced_summary['detection_radius_discharge'].min():.1f} km
  Mean: {balanced_summary['detection_radius_discharge'].mean():.1f} km
  Max: {balanced_summary['detection_radius_discharge'].max():.1f} km

SUBMODULARITY: All scenarios [OK]
"""
        ax_d6.text(0.05, 0.95, diag_text, transform=ax_d6.transAxes, fontsize=9, va='top', ha='left',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, pad=1.0))

    fig3.suptitle('DIAGNOSTICS: All Scenarios Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUT_FIG_DIAGNOSTICS, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] {OUT_FIG_DIAGNOSTICS}")

    # ==================== FIGURE 5: PER-BASIN DISTRIBUTIONS - ALL SCENARIOS ====================
    fig4 = plt.figure(figsize=(18, 10))
    gs4 = fig4.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1) Per-basin TP_rate - ALL SCENARIOS
    axp1 = fig4.add_subplot(gs4[0, 0])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        summary_df = results['basin_summary_df']
        if len(summary_df) > 0:
            axp1.hist(summary_df['TP_rate'], bins=15, alpha=0.4, color=config['color'],
                      label=config['short_name'], edgecolor='black', linewidth=0.5)
    axp1.set_title('Per-basin TP_rate by Scenario', fontweight='bold')
    axp1.set_xlabel('TP_rate'); axp1.set_ylabel('Count')
    axp1.legend(fontsize=8); axp1.grid(alpha=0.25, linestyle=':')

    # 2) Per-basin FP_rate - ALL SCENARIOS
    axp2 = fig4.add_subplot(gs4[0, 1])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        summary_df = results['basin_summary_df']
        if len(summary_df) > 0:
            axp2.hist(summary_df['FP_rate'], bins=15, alpha=0.4, color=config['color'],
                      label=config['short_name'], edgecolor='black', linewidth=0.5)
    axp2.set_title('Per-basin FP_rate by Scenario', fontweight='bold')
    axp2.set_xlabel('FP_rate'); axp2.set_ylabel('Count')
    axp2.legend(fontsize=8); axp2.grid(alpha=0.25, linestyle=':')

    # 3) Per-basin TP vs FP scatter - ALL SCENARIOS
    axp3 = fig4.add_subplot(gs4[0, 2])
    for scenario_key, results in scenario_results.items():
        config = results['config']
        summary_df = results['basin_summary_df']
        if len(summary_df) > 0:
            axp3.scatter(summary_df['FP_rate'], summary_df['TP_rate'], alpha=0.5,
                         c=config['color'], marker=config['marker'], s=40,
                         label=config['short_name'], edgecolors='black', linewidths=0.3)
    axp3.set_title('Per-basin TP vs FP by Scenario', fontweight='bold')
    axp3.set_xlabel('FP_rate'); axp3.set_ylabel('TP_rate')
    axp3.legend(fontsize=8); axp3.grid(alpha=0.25, linestyle=':')

    # 4) Sensors per basin comparison
    axp4 = fig4.add_subplot(gs4[1, 0])
    sensors_per_basin = [r['stats']['sensors_per_basin'] for r in scenario_results.values()]
    bars4 = axp4.bar(scenario_names, sensors_per_basin, color=colors_list, edgecolor='black', linewidth=1.5)
    axp4.set_ylabel('Sensors per Basin (mean)', fontsize=12)
    axp4.set_title('Average Sensors per Basin', fontweight='bold')
    axp4.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars4, sensors_per_basin):
        axp4.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

    # 5) Inter-sensor distance comparison
    axp5 = fig4.add_subplot(gs4[1, 1])
    median_dists = [dist_stats[k]['median'] for k in scenario_results.keys()]
    bars5 = axp5.bar(scenario_names, median_dists, color=colors_list, edgecolor='black', linewidth=1.5)
    axp5.set_ylabel('Median Inter-sensor Distance (km)', fontsize=12)
    axp5.set_title('Sensor Spacing by Scenario', fontweight='bold')
    axp5.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars5, median_dists):
        axp5.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

    # 6) Summary statistics
    axp6 = fig4.add_subplot(gs4[1, 2])
    axp6.axis('off')
    stats_text = """
PER-BASIN STATISTICS SUMMARY
====================================

"""
    for scenario_key, results in scenario_results.items():
        cfg = results['config']
        s_df = results['basin_summary_df']
        if len(s_df) > 0:
            stats_text += f"""{cfg['short_name']}:
  TP: {s_df['TP_rate'].mean():.1%} +/- {s_df['TP_rate'].std():.1%}
  FP: {s_df['FP_rate'].mean():.1%} +/- {s_df['FP_rate'].std():.1%}
  J:  {s_df['J'].mean():.3f} +/- {s_df['J'].std():.3f}

"""

    axp6.text(0.05, 0.95, stats_text, transform=axp6.transAxes, fontsize=10, va='top',
              family='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    fig4.suptitle('Per-basin Detection Performance - All Scenarios', fontweight='bold', fontsize=14, y=0.98)
    plt.savefig(OUT_FIG_PERF_DIST, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] {OUT_FIG_PERF_DIST}")

    # ==================== FIGURE 6: BASIN MAPS (Balanced for detail) ====================
    balanced_selected = scenario_results['B_Balanced']['all_selected_df']
    if gdf_basins is not None and len(balanced_selected) > 0:
        fig5 = plt.figure(figsize=(20, 16))
        gs5 = fig5.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
        
        lons_sel = balanced_selected['lon'].to_numpy()
        lats_sel = balanced_selected['lat'].to_numpy()
        stage_suit_sel = balanced_selected['stage_suitability'].to_numpy()
        discharge_suit_sel = balanced_selected['discharge_suitability'].to_numpy()
        risk_sel = minmax(0.6 * balanced_selected['flood_events_norm'].to_numpy() + 0.4 * balanced_selected['flood_risk_value_norm'].to_numpy())
        risk_bins = np.clip(np.digitize(risk_sel, BIN_EDGES, right=False) - 1, 0, nb - 1)

        ax_m1 = fig5.add_subplot(gs5[0, 0])
        try:
            gdf_basins.plot(ax=ax_m1, facecolor='lightblue', edgecolor='navy', linewidth=0.8, alpha=0.4)
        except:
            pass
        for b in range(nb):
            mask = risk_bins == b
            if mask.sum() > 0:
                ax_m1.scatter(lons_sel[mask], lats_sel[mask], c=TIER_COLORS[b], s=80, marker='*',
                              edgecolors='black', linewidths=0.8, label=f"{BIN_LABELS[b]}", zorder=5)
        ax_m1.set_title('HUC10 Basins with Sensors by Risk Tier (Balanced)', fontweight='bold')
        ax_m1.legend(loc='upper right', fontsize=8)
        ax_m1.set_xlabel('Longitude'); ax_m1.set_ylabel('Latitude'); ax_m1.grid(alpha=0.3, linestyle=':')

        ax_m2 = fig5.add_subplot(gs5[0, 1])
        try:
            gdf_basins.plot(ax=ax_m2, facecolor='lightgray', edgecolor='navy', linewidth=0.8, alpha=0.3)
        except:
            pass
        scatter2 = ax_m2.scatter(lons_sel, lats_sel, c=discharge_suit_sel, cmap='Blues',
                                 s=50, edgecolors='black', linewidths=0.5, alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(scatter2, ax=ax_m2, label='Discharge Suitability', fraction=0.04)
        ax_m2.set_title('Sensors by Discharge Suitability', fontweight='bold')
        ax_m2.set_xlabel('Longitude'); ax_m2.set_ylabel('Latitude'); ax_m2.grid(alpha=0.3, linestyle=':')

        ax_m3 = fig5.add_subplot(gs5[1, 0])
        try:
            gdf_basins.plot(ax=ax_m3, facecolor='lightgray', edgecolor='navy', linewidth=0.8, alpha=0.3)
        except:
            pass
        scatter3 = ax_m3.scatter(lons_sel, lats_sel, c=stage_suit_sel, cmap='Greens',
                                 s=50, edgecolors='black', linewidths=0.5, alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(scatter3, ax=ax_m3, label='Stage Suitability', fraction=0.04)
        ax_m3.set_title('Sensors by Stage Suitability', fontweight='bold')
        ax_m3.set_xlabel('Longitude'); ax_m3.set_ylabel('Latitude'); ax_m3.grid(alpha=0.3, linestyle=':')

        ax_m4 = fig5.add_subplot(gs5[1, 1])
        sensor_counts = balanced_selected.groupby('huc10').size().reset_index(name='n_sensors')
        try:
            gdf_plot = gdf_basins.copy()
            huc_col_gdf = [c for c in gdf_plot.columns if 'huc' in c.lower()][0]
            gdf_plot[huc_col_gdf] = gdf_plot[huc_col_gdf].astype(str)
            gdf_plot = gdf_plot.merge(sensor_counts, left_on=huc_col_gdf, right_on='huc10', how='left')
            gdf_plot['n_sensors'] = gdf_plot['n_sensors'].fillna(0)
            gdf_plot.plot(column='n_sensors', ax=ax_m4, cmap='OrRd', edgecolor='navy',
                          linewidth=0.8, legend=True, legend_kwds={'label': 'Sensors per Basin'})
        except Exception as e:
            print(f"  Choropleth error: {e}")
        ax_m4.set_title('Sensors per Basin (Choropleth)', fontweight='bold')
        ax_m4.set_xlabel('Longitude'); ax_m4.set_ylabel('Latitude'); ax_m4.grid(alpha=0.3, linestyle=':')

        fig5.suptitle('BASIN MAPS: HUC10 Watersheds (Balanced Scenario Detail)',
                      fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(OUT_FIG_BASINS, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] {OUT_FIG_BASINS}")

    # ================== FINAL SUMMARY ==================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY (v4.4-S) - ALL SCENARIOS")
    print("="*80)

    print("\n TABLE 1: SCENARIO COMPARISON")
    print(comparison_table.to_string(index=False))

    print("\n TABLE 2: METHOD COMPARISON (Multi-Sensor vs Baselines)")
    print("="*70)
    for scenario_key, baseline_df in baseline_results_by_scenario.items():
        config = SCENARIOS[scenario_key]
        print(f"\n  {config['name']}:")
        display_df = baseline_df[['Method', 'TP_Rate_mean', 'FP_Rate_mean', 'Score_J_mean']].copy()
        display_df.columns = ['Method', 'TP Rate', 'FP Rate', 'J Score']
        display_df['TP Rate'] = display_df['TP Rate'].apply(lambda x: f"{x:.3f}")
        display_df['FP Rate'] = display_df['FP Rate'].apply(lambda x: f"{x:.3f}")
        display_df['J Score'] = display_df['J Score'].apply(lambda x: f"{x:.3f}")
        print(display_df.to_string(index=False))

    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print(f"\n  CSV FILES:")
    print(f"  [OK] {OUT_SEL_ALL}")
    print(f"  [OK] {OUT_BASIN_SUMMARY_ALL}")
    print(f"  [OK] {OUT_SCENARIO_TABLE}")
    print(f"  [OK] {OUT_BASELINES_ALL}")
    for scenario_key in scenario_results.keys():
        print(f"  [OK] selected_sites_{scenario_key}.csv")
        print(f"  [OK] baselines_{scenario_key}.csv")
    print(f"  [OK] {OUT_DIAG}_sensitivity_all.csv")
    print(f"  [OK] {OUT_DIAG}_pareto_all.csv")
    print(f"  [OK] {OUT_DIAG}_distance_stats.csv")
    print(f"  [OK] {OUT_DIAG}_summary_stats.csv")

    print(f"\n  FIGURES:")
    print(f"  [OK] {OUT_FIG_MAPS_ALL}")
    print(f"  [OK] {OUT_FIG_MAIN}")
    print(f"  [OK] {OUT_FIG_SCENARIOS}")
    print(f"  [OK] {OUT_FIG_DIAGNOSTICS}")
    print(f"  [OK] {OUT_FIG_PERF_DIST}")
    print(f"  [OK] {OUT_FIG_BASINS}")

    print("\n" + "="*80 + "\n")
