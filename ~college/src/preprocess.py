"""
Data Preprocessing Module for WiFi Fingerprint Indoor Positioning (UJIIndoorLoc)
Course: UE24CS352A - Machine Learning Mini-Project
"""

import numpy as np
import pandas as pd
import os

WAP_COUNT = 520
WAP_PREFIX = "WAP"
TARGET_COLS = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']
ALL_METADATA_COLS = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP']

def load_data(file_path):
    """
    Load UJIIndoorLoc dataset from CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df

def preprocess_rssi(df, null_val=-105, min_val=-104, max_val=0, scaling='linear', drop_unused_waps=False, active_waps=None):
    """
    Process raw RSSI columns:
    In UJIIndoorLoc:
      - Valid RSSI values: [-104, 0] dBm
      - 100 indicates 'No AP detected'
      
    Transformations:
      1. Replace 100 with null_val (-105 dBm)
      2. Normalization:
         - 'linear': scales [-105, 0] to [0, 1]
         - 'exponential': exp(RSSI / 24.0) for non-linear signal strength weighting
         - 'raw': returns raw dBm with 100 replaced by null_val
    """
    wap_cols = [c for c in df.columns if c.startswith(WAP_PREFIX)]
    if len(wap_cols) == 0:
        # Fallback if columns are numbered or indexed
        wap_cols = [f"WAP{i:03d}" for i in range(1, WAP_COUNT + 1) if f"WAP{i:03d}" in df.columns]
    
    rssi_data = df[wap_cols].values.astype(float)
    
    # Replace non-detected (100) with null_val
    rssi_data[rssi_data == 100] = null_val
    # Clip any noise outside [-105, 0]
    rssi_data = np.clip(rssi_data, null_val, max_val)
    
    if scaling == 'linear':
        # Scale to [0, 1] range: 0 is weak/not detected, 1 is strongest (0 dBm)
        norm_rssi = (rssi_data - null_val) / (max_val - null_val)
    elif scaling == 'exponential':
        # Exponential scaling emphasizing stronger signals
        norm_rssi = np.exp((rssi_data - null_val) / 20.0) / np.exp((max_val - null_val) / 20.0)
    elif scaling == 'powed':
        # Mathematical positive power transformation: ((RSSI + 105) / 105) ** 2
        norm_rssi = ((rssi_data - null_val) / (max_val - null_val)) ** 2.0
    else:
        norm_rssi = rssi_data

    if drop_unused_waps:
        if active_waps is None:
            # Active WAPs are those that have variation (detected at least once)
            active_waps = np.where(np.std(norm_rssi, axis=0) > 1e-4)[0]
        norm_rssi = norm_rssi[:, active_waps]
        wap_cols = [wap_cols[i] for i in active_waps]

    targets = {
        'coords': df[['LONGITUDE', 'LATITUDE']].values.astype(float) if 'LONGITUDE' in df.columns and 'LATITUDE' in df.columns else None,
        'building': df['BUILDINGID'].values.astype(int) if 'BUILDINGID' in df.columns else None,
        'floor': df['FLOOR'].values.astype(int) if 'FLOOR' in df.columns else None,
        'space': df['SPACEID'].values.astype(int) if 'SPACEID' in df.columns else None
    }
    
    return norm_rssi, targets, wap_cols, active_waps

def generate_synthetic_ujiindoorloc(num_samples=600, num_waps=520, random_seed=42):
    """
    Generates a realistic synthetic dataset mimicking the exact UJIIndoorLoc schema
    for testing, demonstration, and offline evaluation.
    """
    np.random.seed(random_seed)
    wap_names = [f"WAP{i:03d}" for i in range(1, num_waps + 1)]
    
    # Building layouts: 3 buildings (0, 1, 2)
    # Building 0: 4 floors, UTM Longitude [-7650, -7550], Latitude [4864800, 4864900]
    # Building 1: 4 floors, UTM Longitude [-7500, -7400], Latitude [4864900, 4865000]
    # Building 2: 5 floors, UTM Longitude [-7400, -7300], Latitude [4865000, 4865100]
    
    buildings = np.random.choice([0, 1, 2], size=num_samples, p=[0.35, 0.35, 0.30])
    floors = np.zeros(num_samples, dtype=int)
    longitudes = np.zeros(num_samples)
    latitudes = np.zeros(num_samples)
    
    for i in range(num_samples):
        b = buildings[i]
        if b == 0:
            floors[i] = np.random.choice([0, 1, 2, 3])
            longitudes[i] = np.random.uniform(-7650, -7550)
            latitudes[i] = np.random.uniform(4864800, 4864900)
        elif b == 1:
            floors[i] = np.random.choice([0, 1, 2, 3])
            longitudes[i] = np.random.uniform(-7500, -7400)
            latitudes[i] = np.random.uniform(4864900, 4865000)
        else:
            floors[i] = np.random.choice([0, 1, 2, 3, 4])
            longitudes[i] = np.random.uniform(-7400, -7300)
            latitudes[i] = np.random.uniform(4865000, 4865100)
            
    # Assign virtual AP locations in 3D space
    ap_b = np.random.choice([0, 1, 2], size=num_waps)
    ap_x = np.random.uniform(-7650, -7300, size=num_waps)
    ap_y = np.random.uniform(4864800, 4865100, size=num_waps)
    ap_floor = np.random.choice([0, 1, 2, 3, 4], size=num_waps)
    
    rssi_matrix = np.full((num_samples, num_waps), 100.0) # 100 = unobserved
    
    # Log-distance path loss simulation
    # RSSI = P0 - 10 * n * log10(d) - floor_pen
    p0 = -35.0 # signal at 1m
    path_loss_exp = 3.2
    
    for i in range(num_samples):
        b = buildings[i]
        fl = floors[i]
        px, py = longitudes[i], latitudes[i]
        
        # Calculate distances to all APs
        dx = ap_x - px
        dy = ap_y - py
        df = np.abs(ap_floor - fl) * 4.0 # 4 meters per floor
        db = (ap_b != b) * 30.0 # heavy wall/building attenuation
        
        dist = np.sqrt(dx**2 + dy**2 + df**2 + db**2)
        dist = np.maximum(dist, 1.0)
        
        rssi = p0 - 10.0 * path_loss_exp * np.log10(dist) + np.random.normal(0, 3.5, size=num_waps)
        
        # Wi-Fi card sensitivity threshold: typically -95 to -100 dBm
        detectable = (rssi >= -95.0) & (ap_b == b)
        rssi_matrix[i, detectable] = np.clip(np.round(rssi[detectable]), -104, 0)
        # unobserved remains 100.0
        
    df_dict = {w: rssi_matrix[:, j] for j, w in enumerate(wap_names)}
    df_dict['LONGITUDE'] = longitudes
    df_dict['LATITUDE'] = latitudes
    df_dict['FLOOR'] = floors
    df_dict['BUILDINGID'] = buildings
    df_dict['SPACEID'] = np.random.randint(100, 250, size=num_samples)
    df_dict['RELATIVEPOSITION'] = np.random.choice([1, 2], size=num_samples)
    df_dict['USERID'] = np.random.randint(1, 20, size=num_samples)
    df_dict['PHONEID'] = np.random.randint(1, 15, size=num_samples)
    df_dict['TIMESTAMP'] = 1371713700 + np.arange(num_samples) * 10
    
    return pd.DataFrame(df_dict)
