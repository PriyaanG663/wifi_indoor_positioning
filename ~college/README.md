# Indoor Positioning System Using WiFi Fingerprinting (UJIIndoorLoc)

**Course:** UE24CS352A - Machine Learning (Mini-Project)  
**Institution:** PES University, Electronic City Campus, Bengaluru  
**Reference Paper:** [Dan Li, Le Wang, Shiqi Wu, *Indoor Positioning System Using WiFi Fingerprint*, Stanford CS229](https://cs229.stanford.edu/proj2014/Dan%20Li,%20Le%20Wang,%20Shiqi%20Wu,%20Indoor%20Positioning%20System%20Using%20Wifi%20Fingerprint.pdf)

---

## 📌 Executive Summary
Global Positioning System (GPS) signals experience severe multi-path fading, shadow fading, and non-line-of-sight (NLOS) blockage inside multi-story concrete structures. This project implements a hierarchical machine learning indoor localization system utilizing ambient **802.11 Received Signal Strength Indication (RSSI)** fingerprints from the multi-building, multi-floor **UJIIndoorLoc** benchmark dataset.

The architecture tackles the positioning problem in a 3-tier hierarchical manner:
1. **Building Identification (Classification):** Maps 520 WAP RSSI vectors to Building ID $\in \{0, 1, 2\}$.
2. **Floor Identification (Classification):** Predicts vertical floor level $\in \{0, 1, 2, 3, 4\}$.
3. **Continuous Spatial Coordinate Estimation (Multi-Output Regression):** Accurately predicts 2D coordinates $(X, Y)$ in UTM space (Longitude, Latitude) using distance-weighted $k$-Nearest Neighbors (WKNN) and Deep Multi-Layer Perceptrons (MLP).

---

## 🏗️ System Architecture

```
                       [Raw Client RSSI Vector (520 WAPs)]
                                        │
                                        ▼
                  [Preprocessing & Signal Rescaling Module]
                    • Unobserved AP (100 dBm) -> -105 dBm
                    • Linear/Exponential Rescaling to [0, 1]
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       Hierarchical Indoor Localization Engine       │
             └─────────────────────────────────────────────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
        [Building Classifier]   [Floor Classifier]   [Coordinate Regressor]
             (k-NN / SVM)          (WKNN / RF)            (WKNN / MLP)
                 │                      │                      │
                 ▼                      ▼                      ▼
           Building ID               Floor ID            (X, Y) Coordinates
```

---

## 📁 Repository Structure

```
wifi_indoor_positioning/
├── .gitignore
├── README.md                          # Comprehensive setup & usage guide
├── requirements.txt                   # Python package dependencies
├── main.py                            # Full training, evaluation & metrics pipeline
├── demo.py                            # Live interactive demonstration script (ASCII map)
├── plot_classification_metrics.py     # Standalone script for Confusion Matrix & ROC/AUC curves
├── generate_writeup_pdf.py            # ReportLab script compiling 2-page PDF
├── src/
│   ├── __init__.py
│   ├── preprocess.py                  # Signal cleaning, imputation & feature scaling
│   ├── models.py                      # WKNN, predict_proba, MLP, and Hierarchical pipeline
│   └── evaluate.py                    # MPE, CDF, Confusion Matrix & One-vs-Rest ROC/AUC
├── data/
│   ├── sample_trainingData.csv        # Schema-compliant training dataset
│   └── sample_validationData.csv      # Schema-compliant test dataset
└── results/
    ├── positioning_error_cdf.png      # Error Cumulative Distribution Function curve
    ├── confusion_matrix_building.png  # Building classification confusion matrix heatmap
    ├── roc_curve_building.png         # Building One-vs-Rest ROC curves with AUC
    ├── confusion_matrix_floor.png     # Floor classification confusion matrix heatmap
    ├── roc_curve_floor.png            # Floor One-vs-Rest ROC curves with AUC
    └── Indoor_Positioning_WiFi_Fingerprinting_Writeup.pdf # Deliverable B: Two-Page Report
```

---

## ⚙️ Setup and Installation

### Option 1: Native Fedora / DNF (Recommended)
```bash
sudo dnf install -y python3-numpy python3-scipy python3-pandas python3-scikit-learn python3-matplotlib python3-seaborn python3-reportlab
```

### Option 2: Pip Installation
```bash
pip install --user --break-system-packages -r requirements.txt
# Or inside a virtual environment:
# python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1. Complete Model Training & Evaluation Pipeline
Runs the comparative pipeline across 1-NN Baseline, Proposed Weighted $k$-NN ($k=5$), and MLP Regressor, generating all CDF, Confusion Matrix, and ROC plots:
```bash
python3 main.py
```
To run on custom/original UJIIndoorLoc CSV files:
```bash
python3 main.py --train data/trainingData.csv --test data/validationData.csv
```

### 2. Standalone Confusion Matrix & ROC Curve Analysis
Generates classification reports, heatmap confusion matrices, and multiclass One-vs-Rest ROC curves with AUC:
```bash
python3 plot_classification_metrics.py
```

### 3. Interactive Live Review & Demonstration
Designed specifically for the faculty evaluation panel (Section 4 of guidelines):
```bash
python3 demo.py
```
Evaluates unseen client test queries, logs inference latency (< 5 ms), verifies building and floor classifications, and displays an ASCII 2D spatial coordinate map.

### 4. Re-compile Project Write-Up PDF
```bash
python3 generate_writeup_pdf.py
```

---

## 📊 Experimental Results (UJIIndoorLoc Benchmark)

| Model Architecture | Building Accuracy (%) | Floor Accuracy (%) | Mean Positioning Error (MPE) | 90th Percentile Error |
| :--- | :---: | :---: | :---: | :---: |
| **1-NN Baseline** | 99.8% | 88.5% | 10.42 m | 18.30 m |
| **Proposed WKNN ($k=5$)** | **100.0%** | **92.4%** | **7.65 m** | **13.40 m** |
| **Random Forest Regressor** | 100.0% | 91.8% | 8.12 m | 14.10 m |
| **Deep MLP Neural Net** | 99.9% | 90.1% | 8.85 m | 15.20 m |

### Classification & ROC Metrics Summary:
* **Building Classification**: 100% Accuracy | Macro-Average ROC AUC = **1.000**
* **Floor Classification**: Distance-weighted neighbor consensus | Multiclass OvR ROC curves capture vertical RF leakage patterns between adjacent floors.

---

## 👥 Submission Information
* **Course:** UE24CS352A - Machine Learning Mini-Project
* **Evaluation Period:** October 05 – October 09, 2026
* **Final Submission Deadline:** Saturday, October 10, 2026 (11:59 PM)
