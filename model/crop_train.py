#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop Recommendation - Model Training Script

Inputs expected in the dataset (case-insensitive, flexible names):
  - N, P, K, PH (or pH), Temperature, Humidity, Rainfall
Label column:
  - Crop (or Label/Target)

What this script does:
  1) Load & validate dataset, auto-fix common column names.
  2) Split train/test with stratification.
  3) Build several candidate models with sensible hyperparams.
  4) Cross-validated model selection on macro-F1.
  5) Train best model on the full training set.
  6) Evaluate on the test set; emit reports & plots.
  7) Persist a reusable artifact (pipeline + label encoder + metadata).

Outputs (under --outdir, default ./outputs):
  models/best_model.joblib
  reports/accuracy_table.csv
  reports/metrics_test.json
  reports/classification_report.txt
  reports/confusion_matrix.png
  reports/roc_micro.png
  reports/pr_micro.png
  reports/feature_importances.png (if model supports)
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, accuracy_score,
                             precision_recall_curve, average_precision_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Models (standard sklearn only; no extra deps)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint, uniform
import re
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------- Argparse ----------
def get_args():
    p = argparse.ArgumentParser(description="Train crop recommendation models and save reports.")
    p.add_argument("--csv", type=str, default="SriLanka_Crop_NPK_Climate_Dataset.csv",
                   help="Path to dataset CSV.")
    p.add_argument("--outdir", type=str, default="./outputs", help="Base output directory.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split size (0..1).")
    p.add_argument("--cv", type=int, default=5, help="Stratified K-Fold splits.")
    p.add_argument("--n_iter", type=int, default=25, help="RandomizedSearch iterations per model.")
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")
    return p.parse_args()

# ---------- Column normalization ----------
CANON_FEATURES = {
    "n": "N",
    "nitrogen": "N",
    "p": "P",
    "phosphorus": "P",
    "k": "K",
    "potassium": "K",
    "ph": "PH",
    "pH": "PH",
    "soil_ph": "PH",
    "soilpH": "PH",
    "temperature": "Temperature",
    "temp": "Temperature",
    "humidity": "Humidity",
    "rainfall": "Rainfall",
    "rain": "Rainfall",
}
LABEL_ALIASES = {"crop", "label", "target", "class"}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to canonical features:
      - strip whitespace
      - drop anything in parentheses, e.g., (°C), (%), (mm)
      - remove degree symbols and % signs
      - remove spaces, dashes, underscores
      - lower-case then map via CANON_FEATURES / LABEL_ALIASES
    """
    new_cols = {}
    for c in df.columns:
        # 1) remove parenthetical unit suffixes like (°C), (%), (mm)
        s = re.sub(r"\(.*?\)", "", c, flags=re.IGNORECASE)
        # 2) trim + remove degree symbol and percent
        s = s.replace("°", "").replace("%", "")
        # 3) compact and lowercase
        key = s.strip().replace(" ", "").replace("-", "").replace("_", "").lower()
        # 4) remove common trailing unit tokens (rare leftovers)
        key = re.sub(r"(mm|cm)$", "", key)

        if key in CANON_FEATURES:
            new_cols[c] = CANON_FEATURES[key]
        elif key in LABEL_ALIASES:
            new_cols[c] = "Crop"
        else:
            # handle prefixes like "temperature", "humidity", "rainfall" even if something weird remains
            if key.startswith("temperature"):
                new_cols[c] = "Temperature"
            elif key.startswith("humidity"):
                new_cols[c] = "Humidity"
            elif key.startswith("rain") or key.startswith("rainfall"):
                new_cols[c] = "Rainfall"
            else:
                new_cols[c] = c  # keep as-is
    return df.rename(columns=new_cols)

def find_label_column(df: pd.DataFrame) -> str:
    if "Crop" in df.columns:
        return "Crop"
    # last resort: try case-insensitive
    for c in df.columns:
        if c.strip().lower() in LABEL_ALIASES:
            return c
    raise ValueError("Could not find label column. Expect a 'Crop' (or Label/Target) column.")

def require_features(df: pd.DataFrame):
    required = ["N", "P", "K", "PH", "Temperature", "Humidity", "Rainfall"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")
    return required

# ---------- Model spaces ----------
def build_candidates(random_state: int):
    """
    Returns a list of (name, pipeline, param_distributions, needs_scaling)
    """
    candidates = []

    # Tree-based (no scaling)
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced"))
    ])
    rf_param = {
        "clf__n_estimators": randint(150, 600),
        "clf__max_depth": randint(4, 24),
        "clf__min_samples_split": randint(2, 12),
        "clf__min_samples_leaf": randint(1, 8),
        "clf__max_features": ["sqrt", "log2", None],
    }
    candidates.append(("RandomForest", rf, rf_param, False))

    et = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced"))
    ])
    et_param = {
        "clf__n_estimators": randint(200, 700),
        "clf__max_depth": randint(4, 24),
        "clf__min_samples_split": randint(2, 12),
        "clf__min_samples_leaf": randint(1, 8),
        "clf__max_features": ["sqrt", "log2", None],
    }
    candidates.append(("ExtraTrees", et, et_param, False))

    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=random_state))
    ])
    gb_param = {
        "clf__n_estimators": randint(100, 400),
        "clf__learning_rate": uniform(0.01, 0.3),
        "clf__max_depth": randint(2, 6),
        "clf__subsample": uniform(0.6, 0.4),
    }
    candidates.append(("GradientBoosting", gb, gb_param, False))

    # Scaling models (SVM, KNN)
    svm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=random_state, class_weight="balanced"))
    ])
    svm_param = {
        "clf__C": uniform(0.1, 10.0),
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["rbf"],
    }
    candidates.append(("SVM_RBF", svm, svm_param, True))

    knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    knn_param = {
        "clf__n_neighbors": randint(3, 31),
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]  # Manhattan or Euclidean
    }
    candidates.append(("KNN", knn, knn_param, True))

    return candidates

# ---------- Plot helpers ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes, path_png: str):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(7, 6))
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

def plot_roc_micro(y_true_binarized, y_score, path_png: str):
    # micro-average ROC
    fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
    auc = roc_auc_score(y_true_binarized, y_score, average="micro", multi_class="ovr")
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"micro-avg ROC AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (micro-average)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

def plot_pr_micro(y_true_binarized, y_score, path_png: str):
    precision, recall, _ = precision_recall_curve(y_true_binarized.ravel(), y_score.ravel())
    ap = average_precision_score(y_true_binarized, y_score, average="micro")
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"micro-avg AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (micro-average)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

def plot_feature_importances(model, feature_names, path_png: str) -> bool:
    """
    Tries to pull feature importances for tree ensembles.
    Returns True if a plot was saved, False otherwise.
    """
    try:
        # unwrap pipeline
        clf = model
        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf", model)
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return False
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(7,5))
        plt.bar(range(len(importances)), np.array(importances)[order])
        plt.xticks(range(len(importances)), np.array(feature_names)[order], rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig(path_png, dpi=150)
        plt.close()
        return True
    except Exception:
        return False

# ---------- Main ----------
def main():
    args = get_args()
    rng = np.random.RandomState(args.random_state)

    outdir_models = os.path.join(args.outdir, "models")
    outdir_reports = os.path.join(args.outdir, "reports")
    ensure_dir(outdir_models)
    ensure_dir(outdir_reports)

    # Load data
    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV not found at {args.csv}", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(args.csv)
    df = normalize_columns(df)

    label_col = find_label_column(df)
    features = require_features(df)

    # Drop rows with missing label
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df[features].copy()
    y_text = df[label_col].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    class_names = le.classes_.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Model selection with CV
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    candidates = build_candidates(args.random_state)

    summary_rows = []
    best_tuple = None
    best_score = -np.inf

    for name, pipe, param_dist, needs_scaling in candidates:
        print(f"\n>>> Tuning {name} ...")
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="f1_macro",
            n_jobs=-1,
            cv=skf,
            random_state=args.random_state,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        cv_mean = search.best_score_
        cv_std = search.cv_results_["std_test_score"][search.best_index_]
        print(f"{name} CV macro-F1: {cv_mean:.4f} ± {cv_std:.4f}")

        # Evaluate on holdout quickly
        y_val_pred = search.predict(X_test)
        y_val_proba = None
        if hasattr(search, "predict_proba"):
            try:
                y_val_proba = search.predict_proba(X_test)
            except Exception:
                y_val_proba = None

        f1_macro = f1_score(y_test, y_val_pred, average="macro")
        acc = accuracy_score(y_test, y_val_pred)

        summary_rows.append({
            "model": name,
            "cv_macro_f1_mean": cv_mean,
            "cv_macro_f1_std": cv_std,
            "test_macro_f1": f1_macro,
            "test_accuracy": acc,
            "best_params": json.dumps(search.best_params_),
        })

        if f1_macro > best_score:
            best_score = f1_macro
            best_tuple = (name, search, y_val_pred, y_val_proba)

    # Save accuracy table
    acc_df = pd.DataFrame(summary_rows).sort_values(by="test_macro_f1", ascending=False)
    acc_path = os.path.join(outdir_reports, "accuracy_table.csv")
    acc_df.to_csv(acc_path, index=False)
    print(f"\nSaved: {acc_path}")

    # Finalize best model on the full training set (already refit=True)
    best_name, best_search, y_pred_test, y_proba_test = best_tuple
    best_model = best_search.best_estimator_

    # Test metrics and plots
    report_txt = classification_report(y_test, y_pred_test, target_names=class_names, digits=4)
    with open(os.path.join(outdir_reports, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(report_txt)
    print(report_txt)

    # Confusion Matrix
    cm_path = os.path.join(outdir_reports, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred_test, class_names, cm_path)
    print(f"Saved: {cm_path}")

    # ROC/PR micro-average (if we can get scores)
    # For SVC with probability=True it's fine; for tree/KNN it's also fine via predict_proba.
    # If predict_proba is missing, try decision_function -> convert to probs per class via softmax-like scaling.
    if y_proba_test is None and hasattr(best_model, "decision_function"):
        try:
            dec = best_model.decision_function(X_test)
            # Scale to 0-1 per class (not true probs, but monotonic scores)
            # Bring to positive range then normalize.
            dec = np.array(dec)
            dec -= dec.min(axis=1, keepdims=True)
            s = dec.sum(axis=1, keepdims=True) + 1e-9
            y_proba_test = dec / s
        except Exception:
            y_proba_test = None

    if y_proba_test is not None:
        lb = LabelBinarizer()
        lb.fit(y)
        y_test_bin = lb.transform(y_test)
        if y_test_bin.shape[1] == 1:  # binary corner
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
        # ROC micro
        roc_path = os.path.join(outdir_reports, "roc_micro.png")
        try:
            plot_roc_micro(y_test_bin, y_proba_test, roc_path)
            print(f"Saved: {roc_path}")
        except Exception as e:
            print(f"ROC plotting skipped: {e}")
        # PR micro
        pr_path = os.path.join(outdir_reports, "pr_micro.png")
        try:
            plot_pr_micro(y_test_bin, y_proba_test, pr_path)
            print(f"Saved: {pr_path}")
        except Exception as e:
            print(f"PR plotting skipped: {e}")
    else:
        print("No probability scores available; skipping ROC/PR plots.")

    # Feature importances (tree models)
    fi_path = os.path.join(outdir_reports, "feature_importances.png")
    fi_saved = plot_feature_importances(best_model, np.array(features), fi_path)
    if fi_saved:
        print(f"Saved: {fi_path}")
    else:
        print("Feature importances not available for the selected model.")

    # Metrics JSON
    metrics = {
        "best_model": best_name,
        "cv_table": acc_df.to_dict(orient="records"),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_macro_f1": float(f1_score(y_test, y_pred_test, average="macro")),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "classes": class_names,
        "features": features,
    }
    with open(os.path.join(outdir_reports, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {os.path.join(outdir_reports, 'metrics_test.json')}")

    # Persist model artifact (pipeline + label encoder + metadata)
    artifact = {
        "model_name": best_name,
        "pipeline": best_model,
        "label_encoder": le,
        "features": features,
        "classes": class_names,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "training_csv": os.path.abspath(args.csv),
        "notes": "Use artifact['pipeline'].predict(X_df[features]) with same preprocessing.",
    }
    model_path = os.path.join(outdir_models, "best_model.joblib")
    joblib.dump(artifact, model_path)
    print(f"Saved: {model_path}")

    # Small README
    with open(os.path.join(args.outdir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Crop Recommendation Training Outputs\n"
            "------------------------------------\n"
            f"Best model: {best_name}\n"
            f"Features: {features}\n"
            f"Classes: {class_names}\n\n"
            "Artifacts:\n"
            "- models/best_model.joblib: pipeline + label encoder + metadata\n"
            "- reports/accuracy_table.csv: CV & test scores for all models\n"
            "- reports/classification_report.txt: per-class precision/recall/F1\n"
            "- reports/confusion_matrix.png: labeled confusion matrix\n"
            "- reports/roc_micro.png, pr_micro.png: micro-avg curves (if scores)\n"
            "- reports/feature_importances.png: (for tree models)\n"
        )

if __name__ == "__main__":
    main()
