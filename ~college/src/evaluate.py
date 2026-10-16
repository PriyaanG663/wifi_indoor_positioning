"""
Evaluation and Metrics Module for WiFi Fingerprint Indoor Positioning
Course: UE24CS352A - Machine Learning Mini-Project
Includes:
- Mean Positioning Error (MPE), Median, 75th/90th percentile errors
- Penalized 3D Localization Error
- Cumulative Distribution Function (CDF) plotting
- Confusion Matrix computation and heatmap visualization
- Multi-Class Receiver Operating Characteristic (ROC) & AUC curves (One-vs-Rest)
- Formatted Classification Report (Precision, Recall, F1-Score)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def calculate_positioning_metrics(true_coords, pred_coords, true_floor=None, pred_floor=None, true_bldg=None, pred_bldg=None, floor_penalty=4.0, bldg_penalty=50.0):
    """
    Computes comprehensive indoor positioning evaluation metrics:
    - Mean Positioning Error (MPE)
    - Median (50th percentile) Error
    - 75th and 90th percentile errors
    - Floor Classification Accuracy
    - Building Classification Accuracy
    - Penalized 3D Localization Error (IPIN standard)
    """
    true_coords = np.asarray(true_coords, dtype=float)
    pred_coords = np.asarray(pred_coords, dtype=float)
    
    diff = pred_coords - true_coords
    euclidean_errors = np.sqrt(np.sum(diff**2, axis=1))
    
    metrics = {
        'mean_error': float(np.mean(euclidean_errors)),
        'median_error': float(np.median(euclidean_errors)),
        'std_error': float(np.std(euclidean_errors)),
        'p75_error': float(np.percentile(euclidean_errors, 75)),
        'p90_error': float(np.percentile(euclidean_errors, 90)),
        'min_error': float(np.min(euclidean_errors)),
        'max_error': float(np.max(euclidean_errors)),
        'all_errors': euclidean_errors
    }
    
    if true_floor is not None and pred_floor is not None:
        true_floor = np.asarray(true_floor)
        pred_floor = np.asarray(pred_floor)
        floor_acc = np.mean(true_floor == pred_floor) * 100.0
        metrics['floor_accuracy'] = float(floor_acc)
        
    if true_bldg is not None and pred_bldg is not None:
        true_bldg = np.asarray(true_bldg)
        pred_bldg = np.asarray(pred_bldg)
        bldg_acc = np.mean(true_bldg == pred_bldg) * 100.0
        metrics['building_accuracy'] = float(bldg_acc)
        
    if true_floor is not None and pred_floor is not None:
        floor_diff = np.abs(pred_floor - true_floor) * floor_penalty
        penalized = euclidean_errors + floor_diff
        if true_bldg is not None and pred_bldg is not None:
            bldg_diff = (pred_bldg != true_bldg) * bldg_penalty
            penalized += bldg_diff
        metrics['mean_penalized_error'] = float(np.mean(penalized))
        metrics['all_penalized_errors'] = penalized
        
    return metrics

def plot_error_cdf(errors_dict, save_path="results/positioning_error_cdf.png"):
    """
    Plots Cumulative Distribution Function (CDF) of localization errors.
    """
    plt.figure(figsize=(8, 5))
    
    for label, errors in errors_dict.items():
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cdf, label=f"{label} (MPE: {np.mean(errors):.2f}m)", linewidth=2)
        
    plt.xlabel("Localization Error (meters)", fontsize=11)
    plt.ylabel("Cumulative Probability", fontsize=11)
    plt.title("CDF of Positioning Error (UJIIndoorLoc)", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.xlim(0, max(25, np.percentile(list(errors_dict.values())[0], 95) * 1.2))
    plt.ylim(0, 1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

# ==========================================
# CONFUSION MATRIX FUNCTIONS
# ==========================================

def compute_confusion_matrix(y_true, y_pred, classes=None):
    """
    Computes an M x M confusion matrix where rows are True Labels
    and columns are Predicted Labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    classes = np.asarray(classes)
    
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1
            
    return cm, classes

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", save_path="results/confusion_matrix.png", cmap='Blues'):
    """
    Generates and saves a clean confusion matrix heatmap with both raw counts
    and row-normalized percentages in each cell.
    """
    cm, classes = compute_confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Class {c}" for c in classes]
    elif len(class_names) != len(classes):
        class_names = [str(c) for c in classes]
        
    accuracy = np.trace(cm) / np.sum(cm) * 100.0 if np.sum(cm) > 0 else 0.0
    
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f"{title}\nAccuracy: {accuracy:.2f}%", fontsize=12, fontweight='bold', pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=35, ha='right', fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)
    
    # Text annotations in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / row_sum * 100.0) if row_sum > 0 else 0.0
            text_color = "white" if count > thresh else "black"
            plt.text(j, i, f"{count}\n({pct:.1f}%)",
                     ha="center", va="center",
                     color=text_color, fontsize=8.5, fontweight='bold')
            
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return cm, save_path

# ==========================================
# ROC & AUC ANALYSIS FUNCTIONS
# ==========================================

def compute_binary_roc(y_true_binary, y_score):
    """
    Computes False Positive Rate (FPR), True Positive Rate (TPR),
    and Area Under the Curve (AUC) compatible with all NumPy versions
    (NumPy 1.x np.trapz, NumPy 2.x np.trapezoid, and pure Python formula).
    """
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    
    n_pos = np.sum(y_true_binary == 1)
    n_neg = np.sum(y_true_binary == 0)
    
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
        
    # Sort scores descending
    desc_indices = np.argsort(y_score)[::-1]
    y_sorted = y_true_binary[desc_indices]
    
    # Cumulative True and False Positives
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Prepend origin (0, 0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    
    # Trapezoidal AUC calculation compatible across NumPy 1.x and NumPy 2.0+
    if hasattr(np, 'trapezoid'):
        auc = np.abs(float(np.trapezoid(tpr, fpr)))
    elif hasattr(np, 'trapz'):
        auc = np.abs(float(np.trapz(tpr, fpr)))
    else:
        # Standard analytical trapezoidal integration
        auc = np.abs(float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2.0))
        
    return fpr, tpr, auc

def plot_multiclass_roc(y_true, y_probs, class_names=None, title="ROC Curves", save_path="results/roc_curve.png"):
    """
    Generates and saves a One-vs-Rest (OvR) multi-class ROC Curve plot with individual
    class AUCs and Macro-Average AUC.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [f"Class {c}" for c in classes]
    elif len(class_names) != n_classes:
        class_names = [str(c) for c in classes]
        
    plt.figure(figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 3)))
    all_fpr = np.linspace(0, 1, 100)
    interpolated_tprs = []
    aucs = []
    
    for i, c in enumerate(classes):
        # One-vs-Rest binary indicator
        y_binary = (y_true == c).astype(int)
        score = y_probs[:, i] if i < y_probs.shape[1] else np.zeros(len(y_true))
        
        fpr, tpr, auc_val = compute_binary_roc(y_binary, score)
        aucs.append(auc_val)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{class_names[i]} (AUC = {auc_val:.3f})")
                 
        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)
        
    # Macro-average ROC
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = float(np.mean(aucs))
    
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', lw=2.5,
             label=f"Macro-Average (AUC = {macro_auc:.3f})")
             
    # Chance line
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Random Chance (AUC = 0.500)')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    plt.title(f"{title}\nMacro-Average AUC: {macro_auc:.3f}", fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return aucs, macro_auc, save_path

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

def print_classification_report(y_true, y_pred, class_names=None, title="Classification Report"):
    """
    Computes and prints a formatted per-class classification report:
    Precision, Recall, F1-Score, and Support.
    """
    cm, classes = compute_confusion_matrix(y_true, y_pred)
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [f"Class {c}" for c in classes]
        
    print(f"\n=======================================================")
    print(f"             {title.upper()}")
    print(f"=======================================================")
    print(f"{'Class':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    
    precisions, recalls, f1s, supports = [], [], [], []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        
        print(f"{class_names[i]:<16} | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {f1*100:>8.2f}% | {support:>8d}")
        
    total_samples = np.sum(supports)
    macro_prec = np.mean(precisions) * 100.0
    macro_rec = np.mean(recalls) * 100.0
    macro_f1 = np.mean(f1s) * 100.0
    
    accuracy = (np.trace(cm) / total_samples * 100.0) if total_samples > 0 else 0.0
    
    print("-" * 65)
    print(f"{'Accuracy':<16} | {'':<10} | {'':<10} | {accuracy:>8.2f}% | {total_samples:>8d}")
    print(f"{'Macro Avg':<16} | {macro_prec:>8.2f}% | {macro_rec:>8.2f}% | {macro_f1:>8.2f}% | {total_samples:>8d}")
    print("=======================================================\n")

def print_evaluation_summary(metrics, model_name="WKNN Pipeline"):
    """
    Prints a formatted summary table of results.
    """
    print(f"\n==================================================")
    print(f"       EVALUATION REPORT: {model_name}")
    print(f"==================================================")
    if 'building_accuracy' in metrics:
        print(f"  Building Accuracy       : {metrics['building_accuracy']:.2f}%")
    if 'floor_accuracy' in metrics:
        print(f"  Floor Accuracy          : {metrics['floor_accuracy']:.2f}%")
    print(f"  Mean Positioning Error  : {metrics['mean_error']:.2f} m")
    print(f"  Median Positioning Error: {metrics['median_error']:.2f} m")
    print(f"  75th Percentile Error   : {metrics['p75_error']:.2f} m")
    print(f"  90th Percentile Error   : {metrics['p90_error']:.2f} m")
    if 'mean_penalized_error' in metrics:
        print(f"  Mean Penalized 3D Error : {metrics['mean_penalized_error']:.2f} m")
    print(f"==================================================\n")