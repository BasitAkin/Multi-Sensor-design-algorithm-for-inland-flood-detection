"""
NWM VALIDATION - MULTI-SCENARIO COMPREHENSIVE ANALYSIS
===================================================================
METRICS EXTRACTED:
  - AMA (Annual Mean Average)
  - Q95 (95th percentile flow)
  - Q99 (99th percentile flow)
  - AMAX (Annual Maximum - mean of yearly peaks)
  - Flashiness (Q99/Q95 ratio)

SCENARIOS:
  - A: Maximum Coverage
  - B: Balanced
  - C: Precision-Focused
  - D: Resource-Constrained

REFERENCE:
  - 10,000 random candidate points
  - EXCLUDES ALL sensor locations from ALL scenarios
  - Provides true null model for comparison

ANALYSES (per scenario):
  - Distribution comparison (Selected vs Reference)
  - Mann-Whitney U test
  - Cliff's delta effect size
  - Kolmogorov-Smirnov test
  - Percentile analysis (50th, 75th, 90th)
  - Absolute threshold analysis (>=1, >=5, >=10, >=50 m^3/s)
  - Risk-Q99 correlation
  - Risk-Flashiness correlation
  - Classification analysis (Sentinel, Cascade Sentinel, Gap-Filler)

OUTPUTS (COMPREHENSIVE):
  - reference_10000_nwm_metrics.csv (all 10K reference points with metrics)
  - scenario_{name}_nwm_metrics.csv (each scenario's sensors with metrics)
  - scenario_{name}_statistics.csv (detailed statistics for each scenario)
  - all_scenarios_comparison.csv (comparison table)
  - all_scenarios_statistics_full.csv (all stats in one file)
  - threshold_analysis_all.csv (threshold analysis)
  - percentile_analysis_all.csv (percentile analysis)
  - classification_flow_analysis.csv (flow by classification)
  - correlation_analysis_all.csv (risk correlations)
  - Visualization figures

Author: WINGS Lab
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from scipy.spatial import cKDTree
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
# Input files
SELECTED_SITES_ALL_FILE = "Multisensor_Algorithms_outputs_v44S/selected_sites_all_scenarios.csv"
MASTER_DATASET_FILE = os.path.join("data", "master_dataset.csv")
SENSOR_CLASSIFICATION_FILE = "usgs_IntAnaly_outputs_v44S/sensor_classification_all_scenarios.csv"

# NWM Zarr configuration
NWM_ZARR_URL = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
START_DATE = "1979-02-01"
END_DATE = "2023-01-31"
MAX_SNAP_DISTANCE_DEG = 0.05  # ~5.5 km

# Reference sample configuration
REFERENCE_SAMPLE_SIZE = 10000  # 10,000 random points
RANDOM_SEED = 42
CHUNK_SIZE = 100  # Process reaches in chunks

# Output directory
OUT_DIR = os.path.join(os.getcwd(), "nwm_validation_v22S_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Scenario definitions
SCENARIOS = {
    'A: Maximum Coverage': {'short_name': 'MaxCoverage', 'color': '#2ecc71'},
    'B: Balanced': {'short_name': 'Balanced', 'color': '#3498db'},
    'C: Precision-Focused': {'short_name': 'Precision', 'color': '#9b59b6'},
    'D: Resource-Constrained': {'short_name': 'Resource', 'color': '#e74c3c'}
}

# Thresholds for analysis
Q99_THRESHOLDS = [1.0, 5.0, 10.0, 50.0]  # m^3/s

print("=" * 80)
print("NWM VALIDATION v2.2-S FULL - MULTI-SCENARIO COMPREHENSIVE ANALYSIS")
print("  -> Extracts AMA, Q95, Q99, AMAX, Flashiness for ALL scenarios")
print("  -> 10,000 random reference points (excluding all sensor locations)")
print("  -> Complete statistical analysis per scenario")
print("  -> All outputs exported to CSV")
print("=" * 80)

# ================== HELPER FUNCTIONS ==================

def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return (more - less) / (n1 * n2)

def interpret_cliffs_delta(d):
    """Interpret Cliff's delta magnitude."""
    if pd.isna(d):
        return "N/A"
    d_abs = abs(d)
    if d_abs < 0.147:
        return "negligible"
    elif d_abs < 0.33:
        return "small"
    elif d_abs < 0.474:
        return "medium"
    else:
        return "large"

def snap_points_to_nwm_reaches(point_lons, point_lats, reach_lons, reach_lats, 
                                reach_ids, max_distance_deg=0.05):
    """Snap points to nearest NWM reaches using KD-tree."""
    reach_coords = np.column_stack([reach_lons, reach_lats])
    tree = cKDTree(reach_coords)
    
    point_coords = np.column_stack([point_lons, point_lats])
    distances, indices = tree.query(point_coords, k=1)
    
    mean_lat = np.mean(point_lats)
    km_per_deg = 111 * np.cos(np.radians(mean_lat))
    distances_km = distances * km_per_deg
    
    results = pd.DataFrame({
        'point_idx': range(len(point_coords)),
        'feature_id': reach_ids[indices],
        'reach_lon': reach_lons[indices],
        'reach_lat': reach_lats[indices],
        'snap_distance_deg': distances,
        'snap_distance_km': distances_km
    })
    
    too_far = distances > max_distance_deg
    if too_far.any():
        results.loc[too_far, 'feature_id'] = np.nan
    
    return results

def compute_flow_statistics_full(ds, feature_ids, start_date, end_date, chunk_size=100):
    """
    Compute comprehensive flow statistics for given feature IDs.
    Returns: DataFrame with AMA, Q95, Q99, AMAX, Flashiness
    """
    all_feature_ids = ds['feature_id'].values
    
    valid_ids = [fid for fid in feature_ids if fid in all_feature_ids]
    missing_ids = [fid for fid in feature_ids if fid not in all_feature_ids]
    
    print(f"    Feature IDs: {len(valid_ids)} valid, {len(missing_ids)} not in dataset")
    
    if not valid_ids:
        return pd.DataFrame()
    
    ds_time = ds.sel(time=slice(start_date, end_date))
    
    results = []
    n_chunks = (len(valid_ids) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(valid_ids), chunk_size):
        chunk_ids = valid_ids[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        
        if chunk_num % 10 == 1 or chunk_num == n_chunks:
            print(f"    Processing chunk {chunk_num}/{n_chunks} ({len(chunk_ids)} reaches)...")
        
        da_chunk = ds_time['streamflow'].sel(feature_id=chunk_ids)
        data = da_chunk.compute()
        
        for fid in chunk_ids:
            flows = data.sel(feature_id=fid).values
            flows = flows[~np.isnan(flows)]
            
            if len(flows) > 0:
                # All metrics
                ama = np.mean(flows)
                q95 = np.percentile(flows, 95)
                q99 = np.percentile(flows, 99)
                
                # Annual Maximum (AMAX)
                hours_per_year = 365.25 * 24
                n_complete_years = int(len(flows) / hours_per_year)
                
                if n_complete_years >= 1:
                    n_hours = int(n_complete_years * hours_per_year)
                    annual_data = flows[:n_hours].reshape(n_complete_years, -1)
                    annual_max = annual_data.max(axis=1)
                    amax_mean = annual_max.mean()
                    amax_std = annual_max.std()
                else:
                    amax_mean = flows.max()
                    amax_std = 0
                
                # Flashiness (Q99/Q95 ratio)
                flashiness = q99 / q95 if q95 > 0 else np.nan
                
                results.append({
                    'feature_id': fid,
                    'AMA_cms': round(ama, 4),
                    'Q95_cms': round(q95, 4),
                    'Q99_cms': round(q99, 4),
                    'AMAX_mean_cms': round(amax_mean, 4),
                    'AMAX_std_cms': round(amax_std, 4),
                    'Flashiness': round(flashiness, 4),
                    'n_hours': len(flows),
                    'n_years': n_complete_years
                })
            else:
                results.append({
                    'feature_id': fid,
                    'AMA_cms': np.nan,
                    'Q95_cms': np.nan,
                    'Q99_cms': np.nan,
                    'AMAX_mean_cms': np.nan,
                    'AMAX_std_cms': np.nan,
                    'Flashiness': np.nan,
                    'n_hours': 0,
                    'n_years': 0
                })
    
    return pd.DataFrame(results)

def compute_full_statistics(sel_df, ref_df, scenario_name, short_name):
    """
    Compute FULL statistics comparing selected vs reference.
    Replicates ALL analyses from v2.1.
    """
    metrics = ['AMA_cms', 'Q95_cms', 'Q99_cms', 'AMAX_mean_cms', 'Flashiness']
    results = {
        'scenario': scenario_name,
        'short_name': short_name,
        'n_selected': len(sel_df),
        'n_reference': len(ref_df)
    }
    
    for metric in metrics:
        sel_vals = sel_df[metric].dropna().values
        ref_vals = ref_df[metric].dropna().values
        
        if len(sel_vals) > 0 and len(ref_vals) > 0:
            # Basic statistics
            results[f'{metric}_sel_n'] = len(sel_vals)
            results[f'{metric}_sel_median'] = np.median(sel_vals)
            results[f'{metric}_sel_mean'] = np.mean(sel_vals)
            results[f'{metric}_sel_std'] = np.std(sel_vals)
            results[f'{metric}_sel_min'] = np.min(sel_vals)
            results[f'{metric}_sel_max'] = np.max(sel_vals)
            
            results[f'{metric}_ref_n'] = len(ref_vals)
            results[f'{metric}_ref_median'] = np.median(ref_vals)
            results[f'{metric}_ref_mean'] = np.mean(ref_vals)
            results[f'{metric}_ref_std'] = np.std(ref_vals)
            results[f'{metric}_ref_min'] = np.min(ref_vals)
            results[f'{metric}_ref_max'] = np.max(ref_vals)
            
            # Statistical tests
            u_stat, p_mw = stats.mannwhitneyu(sel_vals, ref_vals, alternative='two-sided')
            delta = cliffs_delta(sel_vals, ref_vals)
            ks_stat, ks_p = stats.ks_2samp(sel_vals, ref_vals)
            
            results[f'{metric}_mannwhitney_U'] = u_stat
            results[f'{metric}_mannwhitney_p'] = p_mw
            results[f'{metric}_cliffs_delta'] = delta
            results[f'{metric}_cliffs_interp'] = interpret_cliffs_delta(delta)
            results[f'{metric}_ks_stat'] = ks_stat
            results[f'{metric}_ks_p'] = ks_p
            
            # Percentile analysis
            ref_p25 = np.percentile(ref_vals, 25)
            ref_p50 = np.percentile(ref_vals, 50)
            ref_p75 = np.percentile(ref_vals, 75)
            ref_p90 = np.percentile(ref_vals, 90)
            ref_p95 = np.percentile(ref_vals, 95)
            
            results[f'{metric}_ref_p25'] = ref_p25
            results[f'{metric}_ref_p50'] = ref_p50
            results[f'{metric}_ref_p75'] = ref_p75
            results[f'{metric}_ref_p90'] = ref_p90
            results[f'{metric}_ref_p95'] = ref_p95
            
            pct_above_p25 = 100 * (sel_vals >= ref_p25).sum() / len(sel_vals)
            pct_above_p50 = 100 * (sel_vals >= ref_p50).sum() / len(sel_vals)
            pct_above_p75 = 100 * (sel_vals >= ref_p75).sum() / len(sel_vals)
            pct_above_p90 = 100 * (sel_vals >= ref_p90).sum() / len(sel_vals)
            pct_above_p95 = 100 * (sel_vals >= ref_p95).sum() / len(sel_vals)
            
            results[f'{metric}_pct_above_p25'] = pct_above_p25
            results[f'{metric}_pct_above_p50'] = pct_above_p50
            results[f'{metric}_pct_above_p75'] = pct_above_p75
            results[f'{metric}_pct_above_p90'] = pct_above_p90
            results[f'{metric}_pct_above_p95'] = pct_above_p95
    
    # Threshold analysis for Q99
    sel_q99 = sel_df['Q99_cms'].dropna().values
    ref_q99 = ref_df['Q99_cms'].dropna().values
    
    for thresh in Q99_THRESHOLDS:
        sel_pct = 100 * (sel_q99 >= thresh).sum() / len(sel_q99) if len(sel_q99) > 0 else 0
        ref_pct = 100 * (ref_q99 >= thresh).sum() / len(ref_q99) if len(ref_q99) > 0 else 0
        results[f'Q99_ge_{int(thresh)}_sel_pct'] = sel_pct
        results[f'Q99_ge_{int(thresh)}_ref_pct'] = ref_pct
        results[f'Q99_ge_{int(thresh)}_diff'] = sel_pct - ref_pct
    
    return results

def compute_risk_correlations(df, scenario_name, short_name):
    """Compute Risk-Q99 and Risk-Flashiness correlations."""
    results = {
        'scenario': scenario_name,
        'short_name': short_name
    }
    
    # Find risk column
    risk_col = None
    for col in ['risk_in_basin', 'flood_risk', 'risk_global', 'flood_risk_value_norm', 'flood_events_norm']:
        if col in df.columns:
            risk_col = col
            break
    
    if risk_col is None:
        results['risk_column'] = 'NOT_FOUND'
        return results
    
    results['risk_column'] = risk_col
    
    # Risk vs Q99 correlation
    valid_mask = df[risk_col].notna() & df['Q99_cms'].notna()
    if valid_mask.sum() > 10:
        risk_vals = df.loc[valid_mask, risk_col].values
        q99_vals = df.loc[valid_mask, 'Q99_cms'].values
        
        rho_q99, p_q99 = stats.spearmanr(risk_vals, q99_vals, nan_policy='omit')
        results['risk_Q99_spearman_rho'] = rho_q99
        results['risk_Q99_spearman_p'] = p_q99
    
    # Risk vs Flashiness correlation
    valid_mask_flash = df[risk_col].notna() & df['Flashiness'].notna()
    if valid_mask_flash.sum() > 10:
        risk_vals = df.loc[valid_mask_flash, risk_col].values
        flash_vals = df.loc[valid_mask_flash, 'Flashiness'].values
        
        rho_flash, p_flash = stats.spearmanr(risk_vals, flash_vals, nan_policy='omit')
        results['risk_Flashiness_spearman_rho'] = rho_flash
        results['risk_Flashiness_spearman_p'] = p_flash
    
    # Risk vs AMA correlation
    valid_mask_ama = df[risk_col].notna() & df['AMA_cms'].notna()
    if valid_mask_ama.sum() > 10:
        risk_vals = df.loc[valid_mask_ama, risk_col].values
        ama_vals = df.loc[valid_mask_ama, 'AMA_cms'].values
        
        rho_ama, p_ama = stats.spearmanr(risk_vals, ama_vals, nan_policy='omit')
        results['risk_AMA_spearman_rho'] = rho_ama
        results['risk_AMA_spearman_p'] = p_ama
    
    return results

def compute_classification_analysis(df, df_class, scenario_name, short_name, ref_df):
    """Compute flow statistics by classification."""
    results = []
    
    # Filter classification for this scenario
    df_class_scen = df_class[df_class['scenario'] == scenario_name] if 'scenario' in df_class.columns else df_class
    
    if len(df_class_scen) == 0:
        return pd.DataFrame()
    
    # Merge with flow data
    df_merged = df.merge(
        df_class_scen[['lon', 'lat', 'classification']],
        on=['lon', 'lat'],
        how='left'
    )
    
    ref_q99 = ref_df['Q99_cms'].dropna().values
    ref_flash = ref_df['Flashiness'].dropna().values
    
    for cls in ['Sentinel', 'Cascade Sentinel', 'Gap-Filler', 'Validator']:
        cls_data = df_merged[df_merged['classification'] == cls]
        
        if len(cls_data) > 0:
            q99_vals = cls_data['Q99_cms'].dropna().values
            flash_vals = cls_data['Flashiness'].dropna().values
            ama_vals = cls_data['AMA_cms'].dropna().values
            
            row = {
                'scenario': scenario_name,
                'short_name': short_name,
                'classification': cls,
                'n_sensors': len(cls_data),
                'n_with_q99': len(q99_vals)
            }
            
            if len(q99_vals) > 0:
                row['Q99_median'] = np.median(q99_vals)
                row['Q99_mean'] = np.mean(q99_vals)
                row['Q99_std'] = np.std(q99_vals)
                row['Q99_min'] = np.min(q99_vals)
                row['Q99_max'] = np.max(q99_vals)
                
                # Compare to reference
                if len(ref_q99) > 0:
                    delta = cliffs_delta(q99_vals, ref_q99)
                    row['Q99_cliffs_delta_vs_ref'] = delta
                    row['Q99_cliffs_interp_vs_ref'] = interpret_cliffs_delta(delta)
            
            if len(flash_vals) > 0:
                row['Flashiness_median'] = np.median(flash_vals)
                row['Flashiness_mean'] = np.mean(flash_vals)
                
                if len(ref_flash) > 0:
                    delta_flash = cliffs_delta(flash_vals, ref_flash)
                    row['Flashiness_cliffs_delta_vs_ref'] = delta_flash
            
            if len(ama_vals) > 0:
                row['AMA_median'] = np.median(ama_vals)
                row['AMA_mean'] = np.mean(ama_vals)
            
            results.append(row)
    
    # Also compare classifications within scenario
    sentinels = df_merged[df_merged['classification'] == 'Sentinel']['Q99_cms'].dropna().values
    cascade = df_merged[df_merged['classification'] == 'Cascade Sentinel']['Q99_cms'].dropna().values
    
    if len(sentinels) > 5 and len(cascade) > 5:
        u_stat, p_val = stats.mannwhitneyu(cascade, sentinels, alternative='two-sided')
        delta = cliffs_delta(cascade, sentinels)
        
        results.append({
            'scenario': scenario_name,
            'short_name': short_name,
            'classification': 'CASCADE_vs_SENTINEL',
            'n_sensors': len(sentinels) + len(cascade),
            'n_with_q99': len(sentinels) + len(cascade),
            'Q99_median': np.median(cascade) - np.median(sentinels),
            'Q99_mean': np.mean(cascade) - np.mean(sentinels),
            'Q99_cliffs_delta_vs_ref': delta,
            'Q99_cliffs_interp_vs_ref': interpret_cliffs_delta(delta),
            'mannwhitney_p': p_val
        })
    
    return pd.DataFrame(results)


# ================== MAIN ==================

# -------------------- STEP 1: Load Data --------------------
print("\n STEP 1: Loading data...")

# Load all scenario sensor locations
try:
    df_selected_all = pd.read_csv(SELECTED_SITES_ALL_FILE)
    scenarios_in_data = df_selected_all['scenario'].unique()
    total_sensors = len(df_selected_all)
    print(f"  [OK] Loaded {total_sensors:,} sensor locations across {len(scenarios_in_data)} scenarios")
    for scen in scenarios_in_data:
        n = (df_selected_all['scenario'] == scen).sum()
        print(f"      {scen}: {n} sensors")
except FileNotFoundError:
    raise FileNotFoundError(f"Multi-scenario output not found: {SELECTED_SITES_ALL_FILE}")

# Load master dataset
try:
    df_master = pd.read_csv(MASTER_DATASET_FILE)
    print(f"  [OK] Master dataset: {len(df_master):,} candidates")
except FileNotFoundError:
    raise FileNotFoundError(f"Master dataset not found: {MASTER_DATASET_FILE}")

# Load classification
has_classification = False
df_class = None
if os.path.exists(SENSOR_CLASSIFICATION_FILE):
    try:
        df_class = pd.read_csv(SENSOR_CLASSIFICATION_FILE)
        has_classification = True
        print(f"  [OK] Classifications loaded: {len(df_class)} records")
    except:
        pass

# -------------------- STEP 2: Create Reference Sample --------------------
print("\n STEP 2: Creating reference sample (excluding ALL scenario sensors)...")

# Get bounding box from all sensors
lon_min = df_selected_all['lon'].min() - 0.1
lon_max = df_selected_all['lon'].max() + 0.1
lat_min = df_selected_all['lat'].min() - 0.1
lat_max = df_selected_all['lat'].max() + 0.1

# Filter master to study region
df_master_region = df_master[
    (df_master['lon'] >= lon_min) & (df_master['lon'] <= lon_max) &
    (df_master['lat'] >= lat_min) & (df_master['lat'] <= lat_max)
].copy()
print(f"  Candidates in study region: {len(df_master_region):,}")

# Create set of ALL sensor coordinates (from ALL scenarios)
all_sensor_coords = set(zip(
    df_selected_all['lon'].round(6), 
    df_selected_all['lat'].round(6)
))
print(f"  Unique sensor locations across all scenarios: {len(all_sensor_coords)}")

# Mark candidates that match ANY selected sensor
df_master_region['is_selected'] = df_master_region.apply(
    lambda row: (round(row['lon'], 6), round(row['lat'], 6)) in all_sensor_coords,
    axis=1
)

# Filter to non-selected candidates only
df_candidates = df_master_region[~df_master_region['is_selected']].copy()
n_excluded = df_master_region['is_selected'].sum()
print(f"  Excluded {n_excluded} points that match ANY sensor location")
print(f"  Available candidates: {len(df_candidates):,}")

# Random sample
rng = np.random.default_rng(RANDOM_SEED)
sample_size = min(REFERENCE_SAMPLE_SIZE, len(df_candidates))
sample_idx = rng.choice(len(df_candidates), size=sample_size, replace=False)
df_reference = df_candidates.iloc[sample_idx].copy().reset_index(drop=True)

print(f"  [OK] Sampled {len(df_reference):,} reference points (spatially independent of ALL scenarios)")

# -------------------- STEP 3: Open NWM Zarr --------------------
print("\n STEP 3: Opening NWM Zarr store...")
print(f"  URL: {NWM_ZARR_URL}")

start_time = time.time()
mapper = fsspec.get_mapper(NWM_ZARR_URL, anon=True)
ds = xr.open_zarr(mapper, consolidated=True)

print(f"  [OK] Dataset loaded in {time.time()-start_time:.1f}s")
print(f"    Dimensions: {dict(ds.sizes)}")

# Load reach coordinates
reach_ids = ds['feature_id'].values
reach_lons = ds['longitude'].values
reach_lats = ds['latitude'].values
print(f"  [OK] Loaded coordinates for {len(reach_ids):,} reaches")

# -------------------- STEP 4: Snap and Query Reference Points --------------------
print("\n STEP 4: Processing REFERENCE points (10,000)...")

ref_lons = df_reference['lon'].values
ref_lats = df_reference['lat'].values

snap_results_ref = snap_points_to_nwm_reaches(
    ref_lons, ref_lats,
    reach_lons, reach_lats,
    reach_ids,
    max_distance_deg=MAX_SNAP_DISTANCE_DEG
)

df_reference['feature_id'] = snap_results_ref['feature_id'].values
df_reference['snap_distance_km'] = snap_results_ref['snap_distance_km'].values

n_snapped_ref = df_reference['feature_id'].notna().sum()
print(f"  Snapped {n_snapped_ref}/{len(df_reference)} reference points")

# Query flow data
ref_feature_ids = df_reference['feature_id'].dropna().astype(int).unique().tolist()
print(f"  Unique reaches to query: {len(ref_feature_ids)}")

print("  Querying NWM flow data (this may take 15-30 minutes for 10,000 points)...")
start_time = time.time()
ref_flow_df = compute_flow_statistics_full(
    ds, ref_feature_ids, START_DATE, END_DATE, chunk_size=CHUNK_SIZE
)
print(f"  [OK] Flow data retrieved in {(time.time()-start_time)/60:.1f} minutes")

# Join flow data to reference points
df_reference['feature_id_int'] = df_reference['feature_id'].astype('Int64')
ref_flow_df['feature_id_int'] = ref_flow_df['feature_id'].astype('Int64')

df_reference = df_reference.merge(
    ref_flow_df[['feature_id_int', 'AMA_cms', 'Q95_cms', 'Q99_cms', 'AMAX_mean_cms', 'AMAX_std_cms', 'Flashiness', 'n_hours', 'n_years']],
    on='feature_id_int',
    how='left'
)

df_ref_valid = df_reference[df_reference['Q99_cms'].notna()].copy()
n_ref_with_flow = len(df_ref_valid)
print(f"  [OK] Reference points with valid flow data: {n_ref_with_flow}")

# Save reference data
OUT_REFERENCE = os.path.join(OUT_DIR, "reference_10000_nwm_metrics.csv")
df_ref_valid.to_csv(OUT_REFERENCE, index=False)
print(f"  [OK] SAVED: {OUT_REFERENCE}")

# -------------------- STEP 5: Process Each Scenario --------------------
print("\n" + "=" * 80)
print("STEP 5: PROCESSING EACH SCENARIO")
print("=" * 80)

scenario_results = {}
all_scenario_stats = []
all_correlation_stats = []
all_classification_stats = []
all_percentile_rows = []
all_threshold_rows = []

for scenario_name in scenarios_in_data:
    print(f"\n{'-'*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'-'*60}")
    
    scenario_config = SCENARIOS.get(scenario_name, {'short_name': scenario_name.split(':')[0], 'color': 'gray'})
    short_name = scenario_config['short_name']
    
    # Filter sensors for this scenario
    df_scenario = df_selected_all[df_selected_all['scenario'] == scenario_name].copy()
    n_sensors = len(df_scenario)
    print(f"  Sensors: {n_sensors}")
    
    # Snap to NWM
    sel_lons = df_scenario['lon'].values
    sel_lats = df_scenario['lat'].values
    
    snap_results_sel = snap_points_to_nwm_reaches(
        sel_lons, sel_lats,
        reach_lons, reach_lats,
        reach_ids,
        max_distance_deg=MAX_SNAP_DISTANCE_DEG
    )
    
    df_scenario['feature_id'] = snap_results_sel['feature_id'].values
    df_scenario['snap_distance_km'] = snap_results_sel['snap_distance_km'].values
    
    n_snapped = df_scenario['feature_id'].notna().sum()
    print(f"  Snapped to NWM: {n_snapped}/{n_sensors}")
    
    # Query flow data
    sel_feature_ids = df_scenario['feature_id'].dropna().astype(int).unique().tolist()
    print(f"  Unique reaches: {len(sel_feature_ids)}")
    
    sel_flow_df = compute_flow_statistics_full(
        ds, sel_feature_ids, START_DATE, END_DATE, chunk_size=CHUNK_SIZE
    )
    
    # Join flow data
    df_scenario['feature_id_int'] = df_scenario['feature_id'].astype('Int64')
    sel_flow_df['feature_id_int'] = sel_flow_df['feature_id'].astype('Int64')
    
    df_scenario = df_scenario.merge(
        sel_flow_df[['feature_id_int', 'AMA_cms', 'Q95_cms', 'Q99_cms', 'AMAX_mean_cms', 'AMAX_std_cms', 'Flashiness', 'n_hours', 'n_years']],
        on='feature_id_int',
        how='left'
    )
    
    df_sel_valid = df_scenario[df_scenario['Q99_cms'].notna()].copy()
    n_with_flow = len(df_sel_valid)
    print(f"  Valid flow data: {n_with_flow}/{n_sensors}")
    
    # ========== SAVE SCENARIO CSV ==========
    out_scenario_file = os.path.join(OUT_DIR, f"scenario_{short_name}_nwm_metrics.csv")
    df_sel_valid.to_csv(out_scenario_file, index=False)
    print(f"  [OK] SAVED: {out_scenario_file}")
    
    # ========== COMPUTE FULL STATISTICS ==========
    print(f"  Computing statistics vs reference...")
    scenario_stats = compute_full_statistics(df_sel_valid, df_ref_valid, scenario_name, short_name)
    all_scenario_stats.append(scenario_stats)
    
    # Save individual scenario statistics
    out_stats_file = os.path.join(OUT_DIR, f"scenario_{short_name}_statistics.csv")
    pd.DataFrame([scenario_stats]).to_csv(out_stats_file, index=False)
    print(f"  [OK] SAVED: {out_stats_file}")
    
    # ========== RISK CORRELATIONS ==========
    correlation_stats = compute_risk_correlations(df_sel_valid, scenario_name, short_name)
    all_correlation_stats.append(correlation_stats)
    
    # ========== CLASSIFICATION ANALYSIS ==========
    if has_classification and df_class is not None:
        class_stats = compute_classification_analysis(df_sel_valid, df_class, scenario_name, short_name, df_ref_valid)
        if len(class_stats) > 0:
            all_classification_stats.append(class_stats)
    
    # ========== PERCENTILE ROWS ==========
    for metric in ['Q99_cms', 'AMA_cms', 'Flashiness']:
        sel_vals = df_sel_valid[metric].dropna().values
        ref_vals = df_ref_valid[metric].dropna().values
        
        if len(sel_vals) > 0 and len(ref_vals) > 0:
            for pct in [25, 50, 75, 90, 95]:
                ref_pct_val = np.percentile(ref_vals, pct)
                sel_above = 100 * (sel_vals >= ref_pct_val).sum() / len(sel_vals)
                
                all_percentile_rows.append({
                    'scenario': scenario_name,
                    'short_name': short_name,
                    'metric': metric,
                    'percentile': pct,
                    'ref_percentile_value': ref_pct_val,
                    'sel_pct_above': sel_above,
                    'expected_pct': 100 - pct
                })
    
    # ========== THRESHOLD ROWS ==========
    sel_q99 = df_sel_valid['Q99_cms'].dropna().values
    ref_q99 = df_ref_valid['Q99_cms'].dropna().values
    
    for thresh in Q99_THRESHOLDS:
        sel_pct = 100 * (sel_q99 >= thresh).sum() / len(sel_q99) if len(sel_q99) > 0 else 0
        ref_pct = 100 * (ref_q99 >= thresh).sum() / len(ref_q99) if len(ref_q99) > 0 else 0
        
        all_threshold_rows.append({
            'scenario': scenario_name,
            'short_name': short_name,
            'threshold_cms': thresh,
            'sel_count': (sel_q99 >= thresh).sum(),
            'sel_total': len(sel_q99),
            'sel_pct': sel_pct,
            'ref_count': (ref_q99 >= thresh).sum(),
            'ref_total': len(ref_q99),
            'ref_pct': ref_pct,
            'difference': sel_pct - ref_pct
        })
    
    # ========== PRINT SUMMARY ==========
    print(f"\n  Summary Statistics:")
    print(f"    Q99: Sel={scenario_stats.get('Q99_cms_sel_median',0):.2f}, Ref={scenario_stats.get('Q99_cms_ref_median',0):.2f}, delta={scenario_stats.get('Q99_cms_cliffs_delta',0):.3f} ({scenario_stats.get('Q99_cms_cliffs_interp','N/A')})")
    print(f"    Above P75: {scenario_stats.get('Q99_cms_pct_above_p75',0):.1f}% (expected: 25%)")
    
    # Store results
    scenario_results[scenario_name] = {
        'config': scenario_config,
        'df_valid': df_sel_valid,
        'stats': scenario_stats,
        'n_sensors': n_sensors,
        'n_with_flow': n_with_flow
    }

# -------------------- STEP 6: Save All Aggregated CSVs --------------------
print("\n" + "=" * 80)
print("STEP 6: SAVING ALL AGGREGATED CSV FILES")
print("=" * 80)

# All scenario statistics
OUT_STATS_ALL = os.path.join(OUT_DIR, "all_scenarios_statistics_full.csv")
pd.DataFrame(all_scenario_stats).to_csv(OUT_STATS_ALL, index=False)
print(f"  [OK] {OUT_STATS_ALL}")

# Correlation analysis
OUT_CORR = os.path.join(OUT_DIR, "correlation_analysis_all.csv")
pd.DataFrame(all_correlation_stats).to_csv(OUT_CORR, index=False)
print(f"  [OK] {OUT_CORR}")

# Classification analysis
if all_classification_stats:
    OUT_CLASS = os.path.join(OUT_DIR, "classification_flow_analysis.csv")
    pd.concat(all_classification_stats, ignore_index=True).to_csv(OUT_CLASS, index=False)
    print(f"  [OK] {OUT_CLASS}")

# Percentile analysis
OUT_PCT = os.path.join(OUT_DIR, "percentile_analysis_all.csv")
pd.DataFrame(all_percentile_rows).to_csv(OUT_PCT, index=False)
print(f"  [OK] {OUT_PCT}")

# Threshold analysis
OUT_THRESH = os.path.join(OUT_DIR, "threshold_analysis_all.csv")
pd.DataFrame(all_threshold_rows).to_csv(OUT_THRESH, index=False)
print(f"  [OK] {OUT_THRESH}")

# -------------------- STEP 7: Create Comparison Table --------------------
print("\n" + "=" * 80)
print("SCENARIO COMPARISON TABLE")
print("=" * 80)

comparison_rows = []
for scenario_name, results in scenario_results.items():
    s = results['stats']
    comparison_rows.append({
        'Scenario': results['config']['short_name'],
        'N_Sensors': results['n_sensors'],
        'N_Valid': results['n_with_flow'],
        'Q99_Sel_Median': round(s.get('Q99_cms_sel_median', 0), 2),
        'Q99_Ref_Median': round(s.get('Q99_cms_ref_median', 0), 2),
        'Q99_Cliffs_Delta': round(s.get('Q99_cms_cliffs_delta', 0), 3),
        'Q99_Effect': s.get('Q99_cms_cliffs_interp', 'N/A'),
        'Q99_p_value': round(s.get('Q99_cms_mannwhitney_p', 1), 4),
        'Flash_Sel_Median': round(s.get('Flashiness_sel_median', 0), 2),
        'Flash_Cliffs_Delta': round(s.get('Flashiness_cliffs_delta', 0), 3),
        'Pct_Above_P50': round(s.get('Q99_cms_pct_above_p50', 0), 1),
        'Pct_Above_P75': round(s.get('Q99_cms_pct_above_p75', 0), 1),
        'Pct_Above_P90': round(s.get('Q99_cms_pct_above_p90', 0), 1)
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n" + comparison_df.to_string(index=False))

OUT_COMPARISON = os.path.join(OUT_DIR, "all_scenarios_comparison.csv")
comparison_df.to_csv(OUT_COMPARISON, index=False)
print(f"\n  [OK] {OUT_COMPARISON}")

# -------------------- STEP 8: Visualizations --------------------
print("\n STEP 8: Creating visualizations...")

# Define colors
scenario_shorts = [r['config']['short_name'] for r in scenario_results.values()]
scenario_colors = [r['config']['color'] for r in scenario_results.values()]

# ========== FIGURE 1: MAIN COMPARISON ==========
fig1, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel 1: Q99 medians
ax1 = axes[0, 0]
q99_medians = [r['stats'].get('Q99_cms_sel_median', 0) for r in scenario_results.values()]
ref_q99_median = df_ref_valid['Q99_cms'].median()
bars1 = ax1.bar(scenario_shorts, q99_medians, color=scenario_colors, edgecolor='black', linewidth=1.5)
ax1.axhline(ref_q99_median, color='gray', linestyle='--', linewidth=2, label=f'Reference: {ref_q99_median:.1f}')
ax1.set_ylabel('Q99 Median (m^3/s)', fontsize=11)
ax1.set_title('Q99 by Scenario vs Reference', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3, linestyle=':')
for bar, val in zip(bars1, q99_medians):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

# Panel 2: Cliff's delta
ax2 = axes[0, 1]
deltas = [r['stats'].get('Q99_cms_cliffs_delta', 0) for r in scenario_results.values()]
bars2 = ax2.bar(scenario_shorts, deltas, color=scenario_colors, edgecolor='black', linewidth=1.5)
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.axhline(0.147, color='green', linestyle=':', linewidth=1.5, label='Small effect')
ax2.axhline(-0.147, color='green', linestyle=':', linewidth=1.5)
ax2.set_ylabel("Cliff's delta (vs Reference)", fontsize=11)
ax2.set_title('Q99 Effect Size by Scenario', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3, linestyle=':')

# Panel 3: Percentile analysis
ax3 = axes[0, 2]
pct_above_75 = [r['stats'].get('Q99_cms_pct_above_p75', 0) for r in scenario_results.values()]
bars3 = ax3.bar(scenario_shorts, pct_above_75, color=scenario_colors, edgecolor='black', linewidth=1.5)
ax3.axhline(25, color='gray', linestyle='--', linewidth=2, label='Expected (25%)')
ax3.set_ylabel('% Above Reference 75th Percentile', fontsize=11)
ax3.set_title('Sensors Above Reference P75', fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3, linestyle=':')
for bar, val in zip(bars3, pct_above_75):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

# Panel 4: Q99 boxplot
ax4 = axes[1, 0]
ref_q99 = df_ref_valid['Q99_cms'].values
all_q99 = [ref_q99] + [r['df_valid']['Q99_cms'].values for r in scenario_results.values()]
labels = ['Reference'] + scenario_shorts
bp = ax4.boxplot(all_q99, labels=labels, patch_artist=True)
bp['boxes'][0].set_facecolor('gray')
bp['boxes'][0].set_alpha(0.7)
for i, (patch, color) in enumerate(zip(bp['boxes'][1:], scenario_colors)):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Q99 (m^3/s)', fontsize=11)
ax4.set_title('Q99 Distribution by Scenario', fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle=':')

# Panel 5: Flashiness boxplot
ax5 = axes[1, 1]
ref_flash = df_ref_valid['Flashiness'].dropna().values
all_flash = [ref_flash] + [r['df_valid']['Flashiness'].dropna().values for r in scenario_results.values()]
bp5 = ax5.boxplot(all_flash, labels=labels, patch_artist=True)
bp5['boxes'][0].set_facecolor('gray')
bp5['boxes'][0].set_alpha(0.7)
for i, (patch, color) in enumerate(zip(bp5['boxes'][1:], scenario_colors)):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_ylabel('Flashiness (Q99/Q95)', fontsize=11)
ax5.set_title('Flashiness Distribution by Scenario', fontweight='bold')
ax5.grid(axis='y', alpha=0.3, linestyle=':')

# Panel 6: Summary text
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""
NWM VALIDATION v2.2-S SUMMARY
{'='*45}

REFERENCE: {n_ref_with_flow:,} random points
  (excluding ALL sensor locations)

METRIC COMPARISON (vs Reference):
"""
for scenario_name, results in scenario_results.items():
    s = results['stats']
    short = results['config']['short_name']
    summary_text += f"""
{short}: N={results['n_with_flow']}
  Q99 delta={s.get('Q99_cms_cliffs_delta',0):.3f} ({s.get('Q99_cms_cliffs_interp','N/A')})
  Above P75: {s.get('Q99_cms_pct_above_p75',0):.0f}%
"""
ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, fontsize=9, va='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig1.suptitle('NWM VALIDATION v2.2-S FULL: Multi-Scenario Comparison', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
OUT_FIG1 = os.path.join(OUT_DIR, "nwm_scenario_comparison.png")
plt.savefig(OUT_FIG1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] {OUT_FIG1}")

# ========== FIGURE 2: CDF COMPARISON ==========
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

# Q99 CDF
ax2a = axes2[0, 0]
ref_sorted = np.sort(ref_q99)
ref_cdf = np.arange(1, len(ref_sorted)+1) / len(ref_sorted)
ax2a.plot(ref_sorted, ref_cdf * 100, linewidth=2, color='gray', label='Reference', linestyle='--')
for scenario_name, results in scenario_results.items():
    sel_q99 = results['df_valid']['Q99_cms'].values
    sel_sorted = np.sort(sel_q99)
    sel_cdf = np.arange(1, len(sel_sorted)+1) / len(sel_sorted)
    ax2a.plot(sel_sorted, sel_cdf * 100, linewidth=2, color=results['config']['color'], 
              label=results['config']['short_name'])
ax2a.set_xlabel('Q99 (m^3/s)', fontsize=11)
ax2a.set_ylabel('Cumulative %', fontsize=11)
ax2a.set_title('Q99 CDF: All Scenarios vs Reference', fontweight='bold')
ax2a.legend(fontsize=9)
ax2a.grid(alpha=0.3, linestyle=':')

# AMA CDF
ax2b = axes2[0, 1]
ref_ama = df_ref_valid['AMA_cms'].values
ref_ama_sorted = np.sort(ref_ama)
ref_ama_cdf = np.arange(1, len(ref_ama_sorted)+1) / len(ref_ama_sorted)
ax2b.plot(ref_ama_sorted, ref_ama_cdf * 100, linewidth=2, color='gray', label='Reference', linestyle='--')
for scenario_name, results in scenario_results.items():
    sel_ama = results['df_valid']['AMA_cms'].values
    sel_sorted = np.sort(sel_ama)
    sel_cdf = np.arange(1, len(sel_sorted)+1) / len(sel_sorted)
    ax2b.plot(sel_sorted, sel_cdf * 100, linewidth=2, color=results['config']['color'], 
              label=results['config']['short_name'])
ax2b.set_xlabel('AMA (m^3/s)', fontsize=11)
ax2b.set_ylabel('Cumulative %', fontsize=11)
ax2b.set_title('AMA CDF: All Scenarios vs Reference', fontweight='bold')
ax2b.legend(fontsize=9)
ax2b.grid(alpha=0.3, linestyle=':')

# Flashiness CDF
ax2c = axes2[1, 0]
ref_flash_sorted = np.sort(ref_flash)
ref_flash_cdf = np.arange(1, len(ref_flash_sorted)+1) / len(ref_flash_sorted)
ax2c.plot(ref_flash_sorted, ref_flash_cdf * 100, linewidth=2, color='gray', label='Reference', linestyle='--')
for scenario_name, results in scenario_results.items():
    sel_flash = results['df_valid']['Flashiness'].dropna().values
    sel_sorted = np.sort(sel_flash)
    sel_cdf = np.arange(1, len(sel_sorted)+1) / len(sel_sorted)
    ax2c.plot(sel_sorted, sel_cdf * 100, linewidth=2, color=results['config']['color'], 
              label=results['config']['short_name'])
ax2c.set_xlabel('Flashiness (Q99/Q95)', fontsize=11)
ax2c.set_ylabel('Cumulative %', fontsize=11)
ax2c.set_title('Flashiness CDF: All Scenarios vs Reference', fontweight='bold')
ax2c.legend(fontsize=9)
ax2c.grid(alpha=0.3, linestyle=':')

# AMAX CDF
ax2d = axes2[1, 1]
ref_amax = df_ref_valid['AMAX_mean_cms'].values
ref_amax_sorted = np.sort(ref_amax)
ref_amax_cdf = np.arange(1, len(ref_amax_sorted)+1) / len(ref_amax_sorted)
ax2d.plot(ref_amax_sorted, ref_amax_cdf * 100, linewidth=2, color='gray', label='Reference', linestyle='--')
for scenario_name, results in scenario_results.items():
    sel_amax = results['df_valid']['AMAX_mean_cms'].values
    sel_sorted = np.sort(sel_amax)
    sel_cdf = np.arange(1, len(sel_sorted)+1) / len(sel_sorted)
    ax2d.plot(sel_sorted, sel_cdf * 100, linewidth=2, color=results['config']['color'], 
              label=results['config']['short_name'])
ax2d.set_xlabel('AMAX (m^3/s)', fontsize=11)
ax2d.set_ylabel('Cumulative %', fontsize=11)
ax2d.set_title('AMAX CDF: All Scenarios vs Reference', fontweight='bold')
ax2d.legend(fontsize=9)
ax2d.grid(alpha=0.3, linestyle=':')

fig2.suptitle('NWM VALIDATION: Cumulative Distribution Functions', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
OUT_FIG2 = os.path.join(OUT_DIR, "nwm_cdf_comparison.png")
plt.savefig(OUT_FIG2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] {OUT_FIG2}")

# ========== FIGURE 3: THRESHOLD ANALYSIS ==========
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))

threshold_df = pd.DataFrame(all_threshold_rows)

for idx, thresh in enumerate(Q99_THRESHOLDS):
    ax = axes3[idx // 2, idx % 2]
    
    thresh_data = threshold_df[threshold_df['threshold_cms'] == thresh]
    
    x = np.arange(len(thresh_data) + 1)
    ref_pct = thresh_data['ref_pct'].iloc[0]
    sel_pcts = [ref_pct] + thresh_data['sel_pct'].tolist()
    labels = ['Reference'] + thresh_data['short_name'].tolist()
    colors_plot = ['gray'] + scenario_colors
    
    bars = ax.bar(x, sel_pcts, color=colors_plot, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f'% with Q99 >= {int(thresh)} m^3/s', fontsize=11)
    ax.set_title(f'Threshold Analysis: Q99 >= {int(thresh)} m^3/s', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    
    for bar, val in zip(bars, sel_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

fig3.suptitle('NWM VALIDATION: Threshold Analysis', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
OUT_FIG3 = os.path.join(OUT_DIR, "nwm_threshold_analysis.png")
plt.savefig(OUT_FIG3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] {OUT_FIG3}")

# ========== FIGURE 4: SPATIAL MAPS ==========
fig4, axes4 = plt.subplots(2, 3, figsize=(20, 12))

# Reference map
ax4_ref = axes4[0, 0]
sc = ax4_ref.scatter(df_ref_valid['lon'], df_ref_valid['lat'], c=df_ref_valid['Q99_cms'],
                     cmap='YlOrRd', s=5, alpha=0.5)
plt.colorbar(sc, ax=ax4_ref, label='Q99 (m^3/s)', fraction=0.04)
ax4_ref.set_title(f'Reference (n={n_ref_with_flow:,})', fontweight='bold')
ax4_ref.set_xlabel('Longitude'); ax4_ref.set_ylabel('Latitude')
ax4_ref.grid(alpha=0.3, linestyle=':')

# Scenario maps
for idx, (scenario_name, results) in enumerate(scenario_results.items()):
    ax = axes4[(idx+1)//3, (idx+1)%3]
    df_sel = results['df_valid']
    sc = ax.scatter(df_sel['lon'], df_sel['lat'], c=df_sel['Q99_cms'],
                    cmap='YlOrRd', s=40, alpha=0.8, edgecolors='black', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Q99 (m^3/s)', fraction=0.04)
    ax.set_title(f"{results['config']['short_name']} (n={len(df_sel)})", fontweight='bold',
                 color=results['config']['color'])
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.grid(alpha=0.3, linestyle=':')

# Summary panel
ax_sum = axes4[1, 2]
ax_sum.axis('off')

fig4.suptitle('NWM Q99 Spatial Distribution: Reference vs Scenarios', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
OUT_FIG4 = os.path.join(OUT_DIR, "nwm_spatial_maps.png")
plt.savefig(OUT_FIG4, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] {OUT_FIG4}")

plt.close('all')

# -------------------- FINAL SUMMARY --------------------
print("\n" + "=" * 80)
print("NWM VALIDATION v2.2-S FULL COMPLETE")
print("=" * 80)

print(f"""
 KEY RESULTS:

1. REFERENCE DISTRIBUTION:
   - Sampled {REFERENCE_SAMPLE_SIZE:,} random candidates
   - EXCLUDED all sensor locations from ALL scenarios
   - Valid NWM data: {n_ref_with_flow:,} points

2. SCENARIO COMPARISON:
""")

for scenario_name, results in scenario_results.items():
    s = results['stats']
    short = results['config']['short_name']
    print(f"""
   {short}:
     Sensors: {results['n_sensors']} (valid: {results['n_with_flow']})
     Q99: {s.get('Q99_cms_sel_median',0):.2f} m^3/s (ref: {s.get('Q99_cms_ref_median',0):.2f})
     Cliff's delta: {s.get('Q99_cms_cliffs_delta',0):.3f} ({s.get('Q99_cms_cliffs_interp','N/A')})
     Above P50: {s.get('Q99_cms_pct_above_p50',0):.1f}% (expected: 50%)
     Above P75: {s.get('Q99_cms_pct_above_p75',0):.1f}% (expected: 25%)
     Above P90: {s.get('Q99_cms_pct_above_p90',0):.1f}% (expected: 10%)""")

print(f"""

 OUTPUT FILES:

REFERENCE DATA:
  - {OUT_REFERENCE}

SCENARIO DATA (metrics for each sensor):""")
for scenario_name, results in scenario_results.items():
    short = results['config']['short_name']
    print(f"  - scenario_{short}_nwm_metrics.csv")
    print(f"  - scenario_{short}_statistics.csv")

print(f"""
AGGREGATED ANALYSIS:
  - {OUT_STATS_ALL}
  - {OUT_COMPARISON}
  - {OUT_CORR}
  - {OUT_PCT}
  - {OUT_THRESH}""")

if has_classification:
    print(f"  - classification_flow_analysis.csv")

print(f"""
FIGURES:
  - {OUT_FIG1}
  - {OUT_FIG2}
  - {OUT_FIG3}
  - {OUT_FIG4}
""")

print("=" * 80)
