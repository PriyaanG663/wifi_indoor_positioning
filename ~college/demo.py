"""
Interactive Live Demonstration Script for Review & Presentation
UE24CS352A - Machine Learning Mini-Project
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Ensure package imports work regardless of current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import load_data, preprocess_rssi, generate_synthetic_ujiindoorloc
from src.models import HierarchicalIndoorPositioningSystem
from src.evaluate import calculate_positioning_metrics

def render_ascii_map(actual_x, actual_y, pred_x, pred_y, width=40, height=12):
    """
    Renders an intuitive ASCII visualizer comparing ground truth (*) vs predicted location (P).
    """
    grid = [['.' for _ in range(width)] for _ in range(height)]
    
    # Bounding box around both points with margin
    min_x = min(actual_x, pred_x) - 15
    max_x = max(actual_x, pred_x) + 15
    min_y = min(actual_y, pred_y) - 15
    max_y = max(actual_y, pred_y) + 15
    
    def to_grid(x, y):
        col = int((x - min_x) / (max_x - min_x + 1e-5) * (width - 1))
        row = int((max_y - y) / (max_y - min_y + 1e-5) * (height - 1))
        return np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)
        
    ar, ac = to_grid(actual_x, actual_y)
    pr, pc = to_grid(pred_x, pred_y)
    
    grid[ar][ac] = '*' # Ground truth
    if (pr, pc) == (ar, ac):
        grid[pr][pc] = 'O' # Overlap / exact hit
    else:
        grid[pr][pc] = 'P' # Prediction
        
    print("\n   +------------------ Spatial Map ------------------+")
    for r in range(height):
        print(f"   | {''.join(grid[r])} |")
    print("   +-------------------------------------------------+")
    print("     Legend: [*] Actual Location  [P] Predicted Location  [O] Overlap\n")

def run_live_demo():
    print("\n" + "=" * 65)
    print("   UE24CS352A LIVE DEMO: WiFi Fingerprint Indoor Positioning")
    print("=" * 65)
    
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "sample_trainingData.csv")
    test_path = os.path.join(data_dir, "sample_validationData.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("[*] Generating reference database for demo...")
        df_all = generate_synthetic_ujiindoorloc(num_samples=600, num_waps=520, random_seed=42)
        split = int(0.8 * len(df_all))
        df_train = df_all.iloc[:split].reset_index(drop=True)
        df_test = df_all.iloc[split:].reset_index(drop=True)
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
    else:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
    print(f"[+] Loaded Reference Fingerprint Database: {len(df_train)} reference points across 3 Buildings.")
    
    X_train, targets_train, _, _ = preprocess_rssi(df_train, scaling='linear')
    X_test, targets_test, _, _ = preprocess_rssi(df_test, scaling='linear')
    
    print("[+] Initializing Hierarchical Positioning Engine (Building -> Floor -> Coordinates)...")
    system = HierarchicalIndoorPositioningSystem(k_building=3, k_floor=5, k_coords=5, coord_method='wknn')
    system.fit(X_train, targets_train['building'], targets_train['floor'], targets_train['coords'])
    print("[+] Model calibrated and ready for real-time localization queries.\n")
    
    test_indices = [5, 18, 42, 67, 95]
    print(f"--- Simulating Online Query Fingerprints (Evaluating 5 Sample Queries) ---")
    
    for idx_num, idx in enumerate(test_indices, 1):
        if idx >= len(X_test):
            idx = idx % len(X_test)
        sample_rssi = X_test[idx:idx+1]
        gt_bldg = targets_test['building'][idx]
        gt_floor = targets_test['floor'][idx]
        gt_coords = targets_test['coords'][idx]
        
        start_time = time.time()
        preds = system.predict(sample_rssi)
        latency = (time.time() - start_time) * 1000.0
        
        pred_bldg = preds['building'][0]
        pred_floor = preds['floor'][0]
        pred_coords = preds['coords'][0]
        
        dist_err = np.sqrt(np.sum((pred_coords - gt_coords)**2))
        
        # Count detectable APs in this query
        detected_aps = np.sum(sample_rssi[0] > 0.05)
        
        print(f"\n[Test Case #{idx_num}] Client Query Fingerprint:")
        print(f"   Detected Access Points : {detected_aps} APs")
        print(f"   Prediction Latency     : {latency:.2f} ms")
        print(f"   Building Prediction    : Pred = Building {pred_bldg}  |  Actual = Building {gt_bldg}  {'[CORRECT]' if pred_bldg == gt_bldg else '[MISMATCH]'}")
        print(f"   Floor Prediction       : Pred = Floor {pred_floor}     |  Actual = Floor {gt_floor}     {'[CORRECT]' if pred_floor == gt_floor else '[MISMATCH]'}")
        print(f"   Estimated Coordinates  : X = {pred_coords[0]:.2f}m, Y = {pred_coords[1]:.2f}m")
        print(f"   Actual Coordinates     : X = {gt_coords[0]:.2f}m, Y = {gt_coords[1]:.2f}m")
        print(f"   2D Positioning Error   : {dist_err:.2f} meters")
        
        render_ascii_map(gt_coords[0], gt_coords[1], pred_coords[0], pred_coords[1])
        
    print("\n" + "=" * 65)
    print(" Live demonstration complete. Ready for Evaluation Panel Q&A!")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    run_live_demo()
