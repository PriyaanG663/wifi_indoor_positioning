"""
Main Execution Script for WiFi Fingerprint Indoor Positioning
UE24CS352A - Machine Learning Mini-Project
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Ensure package imports work regardless of current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import load_data, preprocess_rssi, generate_synthetic_ujiindoorloc
from src.models import WeightedKNN, PureNumpyMLPRegressor, HierarchicalIndoorPositioningSystem
from src.evaluate import (
    calculate_positioning_metrics,
    plot_error_cdf,
    plot_confusion_matrix,
    plot_multiclass_roc,
    print_classification_report,
    print_evaluation_summary
)

def run_pipeline(train_path=None, test_path=None, synthetic_samples=1000, save_plot=True, plot_classification=True):
    print("=" * 60)
    print(" UE24CS352A - Machine Learning Mini-Project")
    print(" Indoor Positioning System Using WiFi Fingerprinting")
    print("=" * 60)

    data_dir = os.path.join(BASE_DIR, "data")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    default_train = os.path.join(data_dir, "sample_trainingData.csv")
    default_test = os.path.join(data_dir, "sample_validationData.csv")

    # 1. Dataset Loading
    if train_path and os.path.exists(train_path):
        print(f"[+] Loading training data from: {train_path}")
        df_train = load_data(train_path)
        if test_path and os.path.exists(test_path):
            print(f"[+] Loading test data from: {test_path}")
            df_test = load_data(test_path)
        else:
            print("[!] Test path not provided. Performing 80/20 train/test split...")
            mask = np.random.rand(len(df_train)) < 0.8
            df_test = df_train[~mask].reset_index(drop=True)
            df_train = df_train[mask].reset_index(drop=True)
    elif os.path.exists(default_train) and os.path.exists(default_test) and not train_path:
        print(f"[+] Found cached datasets in: {data_dir}")
        df_train = load_data(default_train)
        df_test = load_data(default_test)
    else:
        print(f"[*] No custom CSV path provided. Generating synthetic UJIIndoorLoc dataset ({synthetic_samples} samples)...")
        df_all = generate_synthetic_ujiindoorloc(num_samples=synthetic_samples, num_waps=520, random_seed=42)
        split_idx = int(0.8 * len(df_all))
        df_train = df_all.iloc[:split_idx].reset_index(drop=True)
        df_test = df_all.iloc[split_idx:].reset_index(drop=True)
        
        # Save synthetic sample datasets for inspection / repo use
        df_train.to_csv(default_train, index=False)
        df_test.to_csv(default_test, index=False)
        print(f"[+] Schema-compliant datasets saved to: {data_dir}")

    print(f"[+] Train set size: {len(df_train)} samples, Test set size: {len(df_test)} samples")

    # 2. Preprocessing
    print("[+] Preprocessing raw RSSI signals (Scaling: Linear [0, 1], Null: -105 dBm)...")
    X_train, targets_train, wap_cols, active_waps = preprocess_rssi(df_train, scaling='linear', drop_unused_waps=False)
    X_test, targets_test, _, _ = preprocess_rssi(df_test, scaling='linear', drop_unused_waps=False)

    y_train_coords = targets_train['coords']
    y_test_coords = targets_test['coords']
    y_train_bldg = targets_train['building']
    y_test_bldg = targets_test['building']
    y_train_floor = targets_train['floor']
    y_test_floor = targets_test['floor']

    # 3. Model 1: Baseline KNN (k=1)
    print("\n--- Training Model 1: 1-NN Baseline ---")
    knn1 = HierarchicalIndoorPositioningSystem(k_building=1, k_floor=1, k_coords=1, coord_method='wknn')
    knn1.fit(X_train, y_train_bldg, y_train_floor, y_train_coords)
    preds_knn1 = knn1.predict(X_test)
    metrics_knn1 = calculate_positioning_metrics(
        y_test_coords, preds_knn1['coords'],
        true_floor=y_test_floor, pred_floor=preds_knn1['floor'],
        true_bldg=y_test_bldg, pred_bldg=preds_knn1['building']
    )
    print_evaluation_summary(metrics_knn1, "1-NN Baseline")

    # 4. Model 2: Proposed Weighted k-NN (WKNN, k=5, Inverse Distance Squared)
    print("\n--- Training Model 2: Proposed Weighted k-NN (k=5, Distance-Weighted) ---")
    wknn5 = HierarchicalIndoorPositioningSystem(k_building=3, k_floor=5, k_coords=5, coord_method='wknn')
    wknn5.fit(X_train, y_train_bldg, y_train_floor, y_train_coords)
    preds_wknn5 = wknn5.predict(X_test)
    probs_wknn5 = wknn5.predict_proba(X_test)
    metrics_wknn5 = calculate_positioning_metrics(
        y_test_coords, preds_wknn5['coords'],
        true_floor=y_test_floor, pred_floor=preds_wknn5['floor'],
        true_bldg=y_test_bldg, pred_bldg=preds_wknn5['building']
    )
    print_evaluation_summary(metrics_wknn5, "Proposed WKNN (k=5)")

    # 5. Model 3: Multi-Layer Perceptron (MLP Neural Regressor)
    print("\n--- Training Model 3: Deep MLP Coordinate Regressor ---")
    mlp = HierarchicalIndoorPositioningSystem(k_building=3, k_floor=5, coord_method='mlp')
    mlp.fit(X_train, y_train_bldg, y_train_floor, y_train_coords)
    preds_mlp = mlp.predict(X_test)
    metrics_mlp = calculate_positioning_metrics(
        y_test_coords, preds_mlp['coords'],
        true_floor=y_test_floor, pred_floor=preds_mlp['floor'],
        true_bldg=y_test_bldg, pred_bldg=preds_mlp['building']
    )
    print_evaluation_summary(metrics_mlp, "MLP Neural Network")

    # 6. Comparative CDF Plot
    if save_plot:
        plot_path = os.path.join(results_dir, "positioning_error_cdf.png")
        plot_error_cdf({
            "1-NN Baseline": metrics_knn1['all_errors'],
            "Proposed WKNN (k=5)": metrics_wknn5['all_errors'],
            "MLP Regressor": metrics_mlp['all_errors']
        }, save_path=plot_path)
        print(f"\n[+] Cumulative Distribution Function (CDF) plot generated and saved to: {plot_path}")

    # 7. Classification Metrics: Confusion Matrix & ROC Curves
    if plot_classification:
        print("\n--- Generating Confusion Matrices and ROC Curves ---")
        # Building Classification
        bldg_classes = np.unique(y_test_bldg)
        bldg_names = [f"Building {b}" for b in bldg_classes]
        print_classification_report(y_test_bldg, preds_wknn5['building'], class_names=bldg_names, title="Building Classification Report")
        
        cm_bldg_path = os.path.join(results_dir, "confusion_matrix_building.png")
        plot_confusion_matrix(y_test_bldg, preds_wknn5['building'], class_names=bldg_names,
                              title="Building Classification Confusion Matrix", save_path=cm_bldg_path, cmap='Blues')
        print(f"[+] Building Confusion Matrix saved to: {cm_bldg_path}")

        roc_bldg_path = os.path.join(results_dir, "roc_curve_building.png")
        plot_multiclass_roc(y_test_bldg, probs_wknn5['building_probs'], class_names=bldg_names,
                            title="Building One-vs-Rest ROC Curves", save_path=roc_bldg_path)
        print(f"[+] Building ROC Curves saved to: {roc_bldg_path}")

        # Floor Classification
        floor_classes = np.unique(y_test_floor)
        floor_names = [f"Floor {f}" for f in floor_classes]
        print_classification_report(y_test_floor, preds_wknn5['floor'], class_names=floor_names, title="Floor Classification Report")

        cm_floor_path = os.path.join(results_dir, "confusion_matrix_floor.png")
        plot_confusion_matrix(y_test_floor, preds_wknn5['floor'], class_names=floor_names,
                              title="Floor Classification Confusion Matrix", save_path=cm_floor_path, cmap='Greens')
        print(f"[+] Floor Confusion Matrix saved to: {cm_floor_path}")

        roc_floor_path = os.path.join(results_dir, "roc_curve_floor.png")
        plot_multiclass_roc(y_test_floor, probs_wknn5['floor_probs'], class_names=floor_names,
                            title="Floor One-vs-Rest ROC Curves", save_path=roc_floor_path)
        print(f"[+] Floor ROC Curves saved to: {roc_floor_path}")

    return {
        'knn1': metrics_knn1,
        'wknn5': metrics_wknn5,
        'mlp': metrics_mlp
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Indoor Positioning System using WiFi Fingerprinting")
    parser.add_argument('--train', type=str, default=None, help="Path to trainingData.csv")
    parser.add_argument('--test', type=str, default=None, help="Path to validationData.csv")
    parser.add_argument('--samples', type=int, default=800, help="Number of synthetic samples if no CSV")
    parser.add_argument('--no-plots', action='store_true', help="Skip generating graphical plots")
    args = parser.parse_args()

    run_pipeline(
        train_path=args.train,
        test_path=args.test,
        synthetic_samples=args.samples,
        save_plot=not args.no_plots,
        plot_classification=not args.no_plots
    )
