import argparse
import json
import os
import re
import textwrap
from datetime import datetime
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")  # non-GUI backend (fix Tk errors)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    train_test_split,
    GridSearchCV,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --------- Helpers ---------

def slugify(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s.strip())
    return re.sub(r"_{2,}", "_", s).strip("_").lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: slugify(c) for c in df.columns}
    return df.rename(columns=mapping)

def find_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect canonical columns:
      Inputs: temperature, ph, nitrogen, phosphorous, potassium, soil, crop, growth_stage
      Targets: fertilizer type (classification) and yield (regression)
    """
    cols = df.columns
    synonyms = {
        "temperature": ["temp", "temperature", "avg_temp", "temperature_c", "temperature_f"],
        "ph": ["ph", "p_h", "soil_ph"],
        "nitrogen": ["n", "nitrogen", "nitro"],
        "phosphorous": ["p", "phosphorous", "phosphorus"],
        "potassium": ["k", "potassium"],
        "soil": ["soil", "soil_type", "soilname", "soil_category"],
        "crop": ["crop", "crop_type", "cropname"],
        "growth_stage": ["growth_stage", "stage", "crop_stage"],
        "fertilizer": ["fertilizer", "fertiliser", "fertilizer_type", "fertiliser_type", "fertilizername"],
        "yield": ["yield", "yield_kg_per_acre", "yield_kg_acre", "yield_kg_ac", "yield_kg",
                  "yield_kg_per_hectare", "yield_t_ha", "yield_kg_ha"],
    }
    slug_to_original = {slugify(c): c for c in cols}
    slugged = set(slug_to_original.keys())

    resolved = {}
    for canon, alts in synonyms.items():
        for alt in alts:
            s = slugify(alt)
            if s in slugged:
                resolved[canon] = slug_to_original[s]
                break

    for canon in ["temperature","ph","nitrogen","phosphorous","potassium","soil","crop","growth_stage","fertilizer","yield"]:
        if canon not in resolved:
            for c in cols:
                if canon in slugify(c):
                    resolved[canon] = c
                    break

    return resolved

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_table_as_png(df: pd.DataFrame, out_path: str, title: str = ""):
    fig_h = 0.6 * max(4, df.shape[0])
    fig_w = max(6, min(16, 0.25 * df.shape[1] * df.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    if title:
        ax.set_title(title, pad=12, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def plot_learning_curve(estimator, X, y, title, out_path, cv):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=None, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=42
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(7,5))
    plt.plot(train_sizes, train_mean, marker='o', label='Training score')
    plt.plot(train_sizes, test_mean, marker='o', label='CV score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def feature_importance_plot(model, feature_names, out_path, topn=20, title="Feature Importance"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:topn]
        names = np.array(feature_names)[idx]
        vals = importances[idx]
        plt.figure(figsize=(8, max(4, 0.3*len(idx))))
        plt.barh(range(len(idx)), vals[::-1])
        plt.yticks(range(len(idx)), names[::-1], fontsize=8)
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()

# ---- Backward-compat helpers ----
def safe_rmse(y_true, y_pred):
    """RMSE that works on old/new sklearn versions."""
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def pick_regression_scorer():
    """Choose a scoring string available in current sklearn."""
    try:
        from sklearn.metrics import get_scorer
        get_scorer("neg_root_mean_squared_error")
        return "neg_root_mean_squared_error"  # preferred
    except Exception:
        return "neg_mean_squared_error"       # fallback on older versions

# =========================
# Accuracy images 
# =========================
def plot_train_vs_test_bars(values: Dict[str, float], title: str, ylabel: str, out_path: str):
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    plt.figure(figsize=(6, 4))
    plt.bar(keys, vals)
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, max(1.0, max(vals) * 1.15))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_model_metric_comparison(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, out_path: str):
    if metric_col not in df.columns or df.shape[0] == 0:
        return
    d = df.copy().sort_values(metric_col, ascending=True)
    plt.figure(figsize=(7, max(3.8, 0.45 * len(d))))
    plt.barh(d.index.astype(str), d[metric_col].values)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# =========================
# detect leakage shortcut columns
# =========================
def detect_deterministic_categorical_features(df: pd.DataFrame, cat_cols: List[str], y_col: str) -> List[str]:
    """
    If a categorical feature maps to exactly 1 label always (ex: Crop -> Fertilizer_Type),
    it is a leakage shortcut. Return those columns.
    """
    bad = []
    for c in cat_cols:
        if c not in df.columns or y_col not in df.columns:
            continue
        try:
            nunq = df.groupby(c)[y_col].nunique()
            if len(nunq) > 0 and nunq.max() == 1:
                bad.append(c)
        except Exception:
            pass
    return bad

# --------- Main training ---------

def main():
    parser = argparse.ArgumentParser(description="Train Fertilizer Recommendation & Yield Prediction models.")
    parser.add_argument("--input", type=str, default="fertilizer_recommendation-dataset.xlsx",
                        help="Path to dataset (.xlsx or .csv).")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name (if .xlsx).")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: outputs/<timestamp>/).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for train/test split.")
    args = parser.parse_args()

    # Output folder (all files saved directly here)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("outputs", stamp)
    ensure_dir(outdir)

    # ---- Load
    ext = os.path.splitext(args.input)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if args.sheet is None:
            xls = pd.ExcelFile(args.input)
            if len(xls.sheet_names) == 0:
                raise ValueError("No sheets found in Excel file.")
            args.sheet = xls.sheet_names[0]
            print(f"[info] --sheet not provided; using first sheet: '{args.sheet}'")

        try:
            import openpyxl  # needed by pandas for .xlsx
        except ImportError as e:
            raise ImportError("pandas needs 'openpyxl' to read Excel files. Install: pip install openpyxl") from e

        df = pd.read_excel(args.input, sheet_name=args.sheet)
        if isinstance(df, dict):
            if args.sheet in df:
                df = df[args.sheet]
            else:
                first_key = next(iter(df))
                print(f"[info] Multiple sheets returned; defaulting to '{first_key}'")
                df = df[first_key]

    elif ext == ".csv":
        df = pd.read_csv(args.input)
    else:
        raise ValueError("Unsupported file type. Use .xlsx, .xls, or .csv")

    df = normalize_columns(df)

    # Detect columns
    col_res = find_columns(df)
    with open(os.path.join(outdir, "column_detection.json"), "w") as f:
        json.dump(col_res, f, indent=2)

    # Required input features (pick what's available)
    wanted_inputs = ["temperature","ph","nitrogen","phosphorous","potassium","soil","crop","growth_stage"]
    input_cols = [col_res[c] for c in wanted_inputs if c in col_res]
    if not input_cols:
        raise ValueError(
            "Could not detect any input columns. Expected some of: "
            "Temperature, PH, Nitrogen, Phosphorous, Potassium, Soil, Crop, Growth_Stage."
        )

    # Targets
    clf_target = col_res.get("fertilizer", None)
    reg_target = col_res.get("yield", None)

    # Keep input + targets
    keep_cols = input_cols + [c for c in [clf_target, reg_target] if c]
    df = df[keep_cols].copy()

    # Split numeric/categorical
    numeric_like = set(["temperature","ph","nitrogen","phosphorous","potassium"])
    numeric_cols = [c for c in input_cols if slugify(c) in numeric_like or pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in input_cols if c not in numeric_cols]

    # Preprocessors (base)
    num_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, cat_cols)
        ],
        remainder="drop"
    )

    # ---------------- Classification: FertilizerType ----------------
    clf_results = {}
    if clf_target and clf_target in df.columns:
        df_clf = df.dropna(subset=[clf_target]).copy()

        # Base features
        Xc = df_clf[input_cols]
        yc = df_clf[clf_target].astype(str)

        # ✅ FIX: remove deterministic categorical leakage features ONLY for classification
        leakage_cols = detect_deterministic_categorical_features(df_clf, cat_cols, clf_target)
        if leakage_cols:
            with open(os.path.join(outdir, "leakage_features_removed.txt"), "w") as f:
                f.write("Removed leakage features (deterministic mapping to target):\n")
                for lc in leakage_cols:
                    f.write(f"- {lc}\n")

            input_cols_clf = [c for c in input_cols if c not in leakage_cols]
            if len(input_cols_clf) == 0:
                raise ValueError(
                    "All classification input features became leakage/removed. "
                    "Please improve dataset (same crop with multiple fertilizers under different conditions)."
                )

            numeric_cols_clf = [c for c in numeric_cols if c in input_cols_clf]
            cat_cols_clf = [c for c in cat_cols if c in input_cols_clf]

            # classification-only preprocessor
            num_tf_clf = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ])
            cat_tf_clf = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            preprocessor_clf = ColumnTransformer(
                transformers=[
                    ("num", num_tf_clf, numeric_cols_clf),
                    ("cat", cat_tf_clf, cat_cols_clf)
                ],
                remainder="drop"
            )

            Xc = df_clf[input_cols_clf]

        else:
            input_cols_clf = input_cols
            numeric_cols_clf = numeric_cols
            cat_cols_clf = cat_cols
            preprocessor_clf = preprocessor

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=args.test_size, random_state=42,
            stratify=yc if yc.nunique() > 1 else None
        )

        classifiers = {
            "logreg": (
                Pipeline(steps=[("prep", preprocessor_clf),
                                ("clf", LogisticRegression(max_iter=500))]),
                {"clf__C": [0.1, 1.0, 5.0]}
            ),
            "rf": (
                Pipeline(steps=[("prep", preprocessor_clf),
                                ("clf", RandomForestClassifier(random_state=42))]),
                {"clf__n_estimators": [150, 300],
                 "clf__max_depth": [None, 8, 16],
                 "clf__min_samples_split": [2, 5]}
            ),
            "gbc": (
                Pipeline(steps=[("prep", preprocessor_clf),
                                ("clf", GradientBoostingClassifier(random_state=42))]),
                {"clf__n_estimators": [150, 300],
                 "clf__learning_rate": [0.05, 0.1],
                 "clf__max_depth": [2, 3]}
            ),
        }

        best_clf_est, best_cv_score = None, -np.inf
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if yc.nunique() > 1 else 5

        for name, (pipe, grid) in classifiers.items():
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
            gs.fit(Xc_train, yc_train)
            ypred = gs.predict(Xc_test)

            acc = accuracy_score(yc_test, ypred)
            f1m = f1_score(yc_test, ypred, average="macro")

            clf_results[name] = {
                "best_params": gs.best_params_,
                "cv_best_f1_macro": float(gs.best_score_),
                "test_accuracy": float(acc),
                "test_f1_macro": float(f1m),
            }

            if gs.best_score_ > best_cv_score:
                best_cv_score = gs.best_score_
                best_clf_est = gs.best_estimator_

        # Save metrics table
        clf_metrics_df = pd.DataFrame(clf_results).T
        clf_metrics_df.to_csv(os.path.join(outdir, "classification_metrics.csv"))

        # Save metrics as PNG + comparison charts
        clf_metrics_png_df = clf_metrics_df.reset_index().rename(columns={"index": "model"}).copy()
        save_table_as_png(clf_metrics_png_df.round(4), os.path.join(outdir, "classification_metrics.png"),
                          title="Classification Metrics (all models)")
        plot_model_metric_comparison(
            clf_metrics_df, "test_accuracy",
            "Model Comparison (Classification) - Test Accuracy",
            "Accuracy", os.path.join(outdir, "model_comparison_accuracy.png")
        )
        plot_model_metric_comparison(
            clf_metrics_df, "test_f1_macro",
            "Model Comparison (Classification) - Test F1 Macro",
            "F1 Macro", os.path.join(outdir, "model_comparison_f1_macro.png")
        )

        # Detailed report & plots for best model
        if best_clf_est is not None:
            ypred = best_clf_est.predict(Xc_test)

            # Train vs Test accuracy images
            ypred_train = best_clf_est.predict(Xc_train)
            train_acc = accuracy_score(yc_train, ypred_train)
            test_acc = accuracy_score(yc_test, ypred)
            plot_train_vs_test_bars(
                {"Train": float(train_acc), "Test": float(test_acc)},
                "Train vs Test Accuracy (Best Classifier)",
                "Accuracy",
                os.path.join(outdir, "train_vs_test_accuracy.png")
            )
            train_f1 = f1_score(yc_train, ypred_train, average="macro")
            test_f1 = f1_score(yc_test, ypred, average="macro")
            plot_train_vs_test_bars(
                {"Train": float(train_f1), "Test": float(test_f1)},
                "Train vs Test F1 Macro (Best Classifier)",
                "F1 Macro",
                os.path.join(outdir, "train_vs_test_f1_macro.png")
            )

            report = classification_report(yc_test, ypred, output_dict=True)
            report_df = pd.DataFrame(report).T
            report_df.to_csv(os.path.join(outdir, "classification_report.csv"))
            save_table_as_png(report_df.round(3), os.path.join(outdir, "classification_report.png"),
                              title="Classification Report (best model)")

            cm = confusion_matrix(yc_test, ypred, labels=np.unique(yc_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(yc_test))
            fig, ax = plt.subplots(figsize=(7, 6))
            disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
            plt.title("Confusion Matrix (best model)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=180)
            plt.close(fig)

            # ROC/PR macro (if proba available)
            if hasattr(best_clf_est, "predict_proba"):
                try:
                    yproba = best_clf_est.predict_proba(Xc_test)
                    classes = np.unique(yc_test)
                    y_true_bin = np.zeros((len(yc_test), len(classes)))
                    for i, cls in enumerate(classes):
                        y_true_bin[:, i] = (np.array(yc_test) == cls).astype(int)

                    aucs = []
                    for i in range(len(classes)):
                        proba_i = yproba[:, i]
                        try:
                            aucs.append(roc_auc_score(y_true_bin[:, i], proba_i))
                        except Exception:
                            pass
                    if len(aucs) > 0:
                        macro_auc = float(np.mean(aucs))
                        with open(os.path.join(outdir, "classification_macro_auc.txt"), "w") as f:
                            f.write(f"Macro ROC AUC: {macro_auc:.4f}")

                    precisions, r_axis = [], np.linspace(0,1,200)
                    for i in range(len(classes)):
                        proba_i = yproba[:, i]
                        p, r, _ = precision_recall_curve(y_true_bin[:, i], proba_i)
                        precisions.append(np.interp(r_axis, r[::-1], p[::-1]))
                    if len(precisions) > 0:
                        pr_mean = np.mean(np.vstack(precisions), axis=0)
                        plt.figure(figsize=(6,5))
                        plt.plot(r_axis, pr_mean)
                        plt.xlabel("Recall"); plt.ylabel("Precision")
                        plt.title("Precision-Recall (macro avg)")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(outdir, "precision_recall_macro.png"), dpi=180)
                        plt.close()
                except Exception:
                    pass

            # Learning curve
            try:
                plot_learning_curve(
                    best_clf_est, Xc_train, yc_train,
                    "Learning Curve (Classification - best)",
                    os.path.join(outdir, "learning_curve_classification.png"),
                    cv=cv
                )
            except Exception:
                pass

            # Feature importance (tree-based)
            try:
                model = best_clf_est.named_steps.get("clf", None)
                prep = best_clf_est.named_steps.get("prep", None)
                if model is not None and prep is not None:
                    num_names = list(numeric_cols_clf)
                    cat_names = []
                    if len(cat_cols_clf) > 0:
                        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
                        cat_names = list(ohe.get_feature_names_out(cat_cols_clf))
                    feat_names = num_names + cat_names
                    feature_importance_plot(
                        model, feat_names,
                        os.path.join(outdir, "feature_importance_classification.png"),
                        title="Feature Importance (Classification)"
                    )
            except Exception:
                pass

            joblib.dump(best_clf_est, os.path.join(outdir, "best_classifier.joblib"))

    # ---------------- Regression: Yield_kg_per_acre ----------------
    reg_results = {}
    if reg_target and reg_target in df.columns:
        df_reg = df.dropna(subset=[reg_target]).copy()
        Xr = df_reg[input_cols]
        yr = pd.to_numeric(df_reg[reg_target], errors="coerce")
        mask = ~yr.isna()
        Xr, yr = Xr[mask], yr[mask]

        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            Xr, yr, test_size=args.test_size, random_state=42
        )

        regressors = {
            "rf": (
                Pipeline(steps=[("prep", preprocessor),
                                ("reg", RandomForestRegressor(random_state=42))]),
                {"reg__n_estimators": [200, 400],
                 "reg__max_depth": [None, 8, 16],
                 "reg__min_samples_split": [2, 5]}
            ),
            "gbr": (
                Pipeline(steps=[("prep", preprocessor),
                                ("reg", GradientBoostingRegressor(random_state=42))]),
                {"reg__n_estimators": [200, 400],
                 "reg__learning_rate": [0.05, 0.1],
                 "reg__max_depth": [2, 3]}
            ),
            "ridge": (
                Pipeline(steps=[("prep", preprocessor),
                                ("reg", Ridge())]),
                {"reg__alpha": [0.1, 1.0, 10.0]}
            ),
            "linreg": (
                Pipeline(steps=[("prep", preprocessor),
                                ("reg", LinearRegression())]),
                {}
            ),
        }

        best_reg_est, best_cv_score = None, -np.inf
        cvr = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring_name = pick_regression_scorer()

        for name, (pipe, grid) in regressors.items():
            gs = GridSearchCV(pipe, grid, cv=cvr, scoring=scoring_name, n_jobs=-1, refit=True)
            gs.fit(Xr_train, yr_train)
            yhat = gs.predict(Xr_test)

            rmse = safe_rmse(yr_test, yhat)
            r2 = r2_score(yr_test, yhat)

            best_cv = float(gs.best_score_)  # negative error
            if scoring_name == "neg_root_mean_squared_error":
                cv_rmse = -best_cv
            else:
                cv_rmse = float(np.sqrt(-best_cv))

            reg_results[name] = {
                "best_params": gs.best_params_,
                "cv_best_score_name": scoring_name,
                "cv_best_score_value": best_cv,
                "cv_best_rmse": cv_rmse,
                "test_rmse": float(rmse),
                "test_r2": float(r2),
            }

            if gs.best_score_ > best_cv_score:
                best_cv_score = gs.best_score_
                best_reg_est = gs.best_estimator_

        reg_metrics_df = pd.DataFrame(reg_results).T
        reg_metrics_df.to_csv(os.path.join(outdir, "regression_metrics.csv"))

        # Save regression metrics as PNG + model comparison charts
        reg_metrics_png_df = reg_metrics_df.reset_index().rename(columns={"index": "model"}).copy()
        save_table_as_png(reg_metrics_png_df.round(4), os.path.join(outdir, "regression_metrics.png"),
                          title="Regression Metrics (all models)")
        plot_model_metric_comparison(
            reg_metrics_df, "test_rmse",
            "Model Comparison (Regression) - Test RMSE",
            "RMSE", os.path.join(outdir, "model_comparison_rmse.png")
        )
        plot_model_metric_comparison(
            reg_metrics_df, "test_r2",
            "Model Comparison (Regression) - Test R²",
            "R²", os.path.join(outdir, "model_comparison_r2.png")
        )

        if best_reg_est is not None:
            yhat = best_reg_est.predict(Xr_test)
            resid = yr_test - yhat
            plt.figure(figsize=(6,5))
            plt.scatter(yhat, resid, s=14, alpha=0.7)
            plt.axhline(0, color="black", lw=1)
            plt.xlabel("Predicted Yield")
            plt.ylabel("Residuals")
            plt.title("Residual Plot (best regressor)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "residuals_regression.png"), dpi=180)
            plt.close()

            # Train vs Test RMSE/R2 images
            yhat_train = best_reg_est.predict(Xr_train)
            train_rmse = safe_rmse(yr_train, yhat_train)
            test_rmse = safe_rmse(yr_test, yhat)
            plot_train_vs_test_bars(
                {"Train": float(train_rmse), "Test": float(test_rmse)},
                "Train vs Test RMSE (Best Regressor)",
                "RMSE (lower is better)",
                os.path.join(outdir, "train_vs_test_rmse.png")
            )
            train_r2 = r2_score(yr_train, yhat_train)
            test_r2 = r2_score(yr_test, yhat)
            plot_train_vs_test_bars(
                {"Train": float(train_r2), "Test": float(test_r2)},
                "Train vs Test R² (Best Regressor)",
                "R²",
                os.path.join(outdir, "train_vs_test_r2.png")
            )

            try:
                plot_learning_curve(
                    best_reg_est, Xr_train, yr_train,
                    "Learning Curve (Regression - best)",
                    os.path.join(outdir, "learning_curve_regression.png"),
                    cv=cvr
                )
            except Exception:
                pass

            try:
                model = best_reg_est.named_steps.get("reg", None)
                prep = best_reg_est.named_steps.get("prep", None)
                if model is not None and prep is not None:
                    num_names = list(numeric_cols)
                    cat_names = []
                    if len(cat_cols) > 0:
                        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
                        cat_names = list(ohe.get_feature_names_out(cat_cols))
                    feat_names = num_names + cat_names
                    feature_importance_plot(
                        model, feat_names,
                        os.path.join(outdir, "feature_importance_regression.png"),
                        title="Feature Importance (Regression)"
                    )
            except Exception:
                pass

            joblib.dump(best_reg_est, os.path.join(outdir, "best_regressor.joblib"))

    bundle = {
        "inputs_used": input_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "classification_target": clf_target,
        "regression_target": reg_target,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(bundle, f, indent=2)

    with open(os.path.join(outdir, "README.txt"), "w") as f:
        f.write(textwrap.dedent(f"""
        Artifacts generated in: {outdir}

        Files:
          - column_detection.json
          - summary.json

          - leakage_features_removed.txt             (NEW, if leakage detected)

          - classification_metrics.csv
          - classification_metrics.png
          - model_comparison_accuracy.png
          - model_comparison_f1_macro.png
          - train_vs_test_accuracy.png
          - train_vs_test_f1_macro.png

          - classification_report.csv
          - classification_report.png
          - confusion_matrix.png
          - precision_recall_macro.png              (if probabilities available)
          - learning_curve_classification.png
          - feature_importance_classification.png   (if tree-based)
          - best_classifier.joblib                  (if classification trained)

          - regression_metrics.csv
          - regression_metrics.png
          - model_comparison_rmse.png
          - model_comparison_r2.png
          - train_vs_test_rmse.png
          - train_vs_test_r2.png

          - residuals_regression.png
          - learning_curve_regression.png
          - feature_importance_regression.png       (if tree-based)
          - best_regressor.joblib                   (if regression trained)
        """).strip())

    print(f"✅ Done. Artifacts saved to: {outdir}")

if __name__ == "__main__":
    main()
