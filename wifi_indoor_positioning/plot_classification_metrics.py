"""
Classification Metrics Runner: Confusion Matrix and ROC Curves
Course: UE24CS352A - Machine Learning Mini-Project
Topic: Indoor Positioning System Using WiFi Fingerprint (UJIIndoorLoc)

Generates:
1. Building Classification Confusion Matrix Heatmap
2. Floor Classification Confusion Matrix Heatmap
3. Building One-vs-Rest ROC Curves & AUC Analysis
4. Floor One-vs-Rest ROC Curves & AUC Analysis
5. Per-class Precision, Recall, F1-Score, and Support Tables
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import load_data, preprocess_rssi, generate_synthetic_ujiindoorloc
from src.models import HierarchicalIndoorPositioningSystem
from src.evaluate import (
    plot_confusion_matrix,
    plot_multiclass_roc,
    print_classification_report
)

def run_classification_analysis(train_path=None, test_path=None, samples=800):
    print("=" * 65)
    print("   UE24CS352A: CLASSIFICATION METRICS & ROC / CONFUSION MATRIX")
    print("   Topic: Indoor Positioning System Using WiFi Fingerprinting")
    print("=" * 65)

    data_dir = os.path.join(BASE_DIR, "data")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    default_train = os.path.join(data_dir, "sample_trainingData.csv")
    default_test = os.path.join(data_dir, "sample_validationData.csv")

    if train_path and os.path.exists(train_path):
        print(f"[+] Loading training data: {train_path}")
        df_train = load_data(train_path)
        if test_path and os.path.exists(test_path):
            print(f"[+] Loading validation data: {test_path}")
            df_test = load_data(test_path)
        else:
            mask = np.random.rand(len(df_train)) < 0.8
            df_test = df_train[~mask].reset_index(drop=True)
            df_train = df_train[mask].reset_index(drop=True)
    elif os.path.exists(default_train) and os.path.exists(default_test):
        print(f"[+] Using dataset from: {data_dir}")
        df_train = load_data(default_train)
        df_test = load_data(default_test)
    else:
        print(f"[*] Generating {samples} sample fingerprints...")
        df_all = generate_synthetic_ujiindoorloc(num_samples=samples, num_waps=520, random_seed=42)
        split = int(0.8 * len(df_all))
        df_train = df_all.iloc[:split].reset_index(drop=True)
        df_test = df_all.iloc[split:].reset_index(drop=True)
        df_train.to_csv(default_train, index=False)
        df_test.to_csv(default_test, index=False)

    print(f"[+] Training Set: {len(df_train)} samples | Validation Set: {len(df_test)} samples")

    # Preprocessing
    print("[+] Preprocessing raw RSSI vectors (Linear Normalization [0, 1])...")
    X_train, targets_train, _, _ = preprocess_rssi(df_train, scaling='linear')
    X_test, targets_test, _, _ = preprocess_rssi(df_test, scaling='linear')

    y_train_bldg = targets_train['building']
    y_test_bldg = targets_test['building']
    y_train_floor = targets_train['floor']
    y_test_floor = targets_test['floor']
    y_train_coords = targets_train['coords']

    # Train Hierarchical Model
    print("[+] Training Hierarchical Positioning Pipeline...")
    model = HierarchicalIndoorPositioningSystem(k_building=3, k_floor=5, k_coords=5, coord_method='wknn')
    model.fit(X_train, y_train_bldg, y_train_floor, y_train_coords)

    # Predictions & Probabilities
    print("[+] Generating predictions and class probability distributions...")
    preds = model.predict(X_test)
    probs_dict = model.predict_proba(X_test)

    pred_bldg = preds['building']
    pred_floor = preds['floor']
    bldg_probs = probs_dict['building_probs']
    floor_probs = probs_dict['floor_probs']

    # 1. BUILDING CLASSIFICATION METRICS
    bldg_names = [f"Building {b}" for b in np.unique(y_test_bldg)]
    print_classification_report(y_test_bldg, pred_bldg, class_names=bldg_names, title="Building Classification Report")

    cm_bldg_path = os.path.join(results_dir, "confusion_matrix_building.png")
    plot_confusion_matrix(
        y_test_bldg, pred_bldg,
        class_names=bldg_names,
        title="Building Classification Confusion Matrix",
        save_path=cm_bldg_path,
        cmap='Blues'
    )
    print(f"[+] Building Confusion Matrix saved to: {cm_bldg_path}")

    roc_bldg_path = os.path.join(results_dir, "roc_curve_building.png")
    plot_multiclass_roc(
        y_test_bldg, bldg_probs,
        class_names=bldg_names,
        title="Building Classification One-vs-Rest ROC Curves",
        save_path=roc_bldg_path
    )
    print(f"[+] Building ROC Curves saved to: {roc_bldg_path}")

    # 2. FLOOR CLASSIFICATION METRICS
    floor_classes = np.unique(y_test_floor)
    floor_names = [f"Floor {f}" for f in floor_classes]
    print_classification_report(y_test_floor, pred_floor, class_names=floor_names, title="Floor Classification Report")

    cm_floor_path = os.path.join(results_dir, "confusion_matrix_floor.png")
    plot_confusion_matrix(
        y_test_floor, pred_floor,
        class_names=floor_names,
        title="Floor Classification Confusion Matrix",
        save_path=cm_floor_path,
        cmap='Greens'
    )
    print(f"[+] Floor Confusion Matrix saved to: {cm_floor_path}")

    roc_floor_path = os.path.join(results_dir, "roc_curve_floor.png")
    plot_multiclass_roc(
        y_test_floor, floor_probs,
        class_names=floor_names,
        title="Floor Classification One-vs-Rest ROC Curves",
        save_path=roc_floor_path
    )
    print(f"[+] Floor ROC Curves saved to: {roc_floor_path}")

    print("\n" + "=" * 65)
    print(" All Confusion Matrices and ROC Curves generated successfully!")
    print(" Generated Files in results/:")
    print(f"  1. {os.path.basename(cm_bldg_path)}")
    print(f"  2. {os.path.basename(roc_bldg_path)}")
    print(f"  3. {os.path.basename(cm_floor_path)}")
    print(f"  4. {os.path.basename(roc_floor_path)}")
    print("=" * 65 + "\n")

    return {
        'cm_bldg': cm_bldg_path,
        'roc_bldg': roc_bldg_path,
        'cm_floor': cm_floor_path,
        'roc_floor': roc_floor_path
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Confusion Matrix and ROC Curves")
    parser.add_argument('--train', type=str, default=None, help="Path to training CSV")
    parser.add_argument('--test', type=str, default=None, help="Path to validation CSV")
    parser.add_argument('--samples', type=int, default=800, help="Synthetic sample count")
    args = parser.parse_args()

    run_classification_analysis(train_path=args.train, test_path=args.test, samples=args.samples)
