# recommendation_routes.py

import os
import re
import sys
import joblib
import torch
import subprocess
import numpy as np
import pandas as pd

from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util

recommendation_bp = Blueprint("recommendation", __name__, url_prefix="/api/recommend")

# ✅ SAFE PATHS (works no matter where you run flask from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "complete_recommendation_model.joblib")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_model.py")

# -----------------------------
# Load model and components
# -----------------------------
labour_clf = None
equip_clf = None
vectorizer = None
le_labour = None
le_equip = None
DF_labour = None
DF_equip = None
semantic_model = None

def load_models():
    global labour_clf, equip_clf, vectorizer
    global le_labour, le_equip
    global DF_labour, DF_equip
    global semantic_model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model_data = joblib.load(MODEL_PATH)

    labour_clf = model_data["classifier_model_labour"]
    equip_clf = model_data["classifier_model_equipment"]
    vectorizer = model_data["vectorizer"]
    le_labour = model_data["label_encoder_labour"]
    le_equip = model_data["label_encoder_equipment"]
    DF_labour = model_data["DF_labour"]
    DF_equip = model_data["DF_equipment"]

    semantic_model = SentenceTransformer(model_data["embedding_model_name"])
    print("✅ Recommendation models loaded")

# load at import time
load_models()


@recommendation_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Recommendation Engine"})


@recommendation_bp.post("/train")
def train():
    """
    Optional endpoint.
    Runs train_model.py and reloads the model.
    """
    try:
        if not os.path.exists(TRAIN_SCRIPT):
            return jsonify({"error": f"train_model.py not found at {TRAIN_SCRIPT}"}), 404

        # Run training script from BASE_DIR so relative paths work
        result = subprocess.run(
            [sys.executable, "-u", TRAIN_SCRIPT],
            cwd=BASE_DIR,
            check=False
        )

        if result.returncode != 0:
            return jsonify({"error": "Training failed. Check server logs."}), 500

        load_models()
        return jsonify({"status": "ok", "message": "Model retrained and reloaded"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@recommendation_bp.post("/recommend")
def recommend():
    """
    Body JSON:
    {
      "query": "harvesting labour in Kurunegala under 3000",
      "top_k": 5
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        top_k = int(data.get("top_k") or 5)

        if not query:
            return jsonify({"error": "Empty query"}), 400

        # 1) Predict labour + equipment type (TF-IDF classifier)
        qv = vectorizer.transform([query])
        labour_pred_id = labour_clf.predict(qv)[0]
        equip_pred_id = equip_clf.predict(qv)[0]

        labour_label = le_labour.inverse_transform([labour_pred_id])[0]
        equip_label = le_equip.inverse_transform([equip_pred_id])[0]

        # 2) Semantic embedding
        query_emb = semantic_model.encode(query, convert_to_tensor=True)

        # 3) Labour recommendations
        df_lab = DF_labour.copy()
        if "Labour_Type_collapsed" in df_lab.columns:
            df_lab = df_lab[df_lab["Labour_Type_collapsed"] == labour_label]

        labour_results = []
        if not df_lab.empty:
            df_lab["similarity"] = df_lab["embedding"].apply(
                lambda x: util.cos_sim(query_emb, torch.tensor(x)).item()
            )

            # optional price filter from query text
            m = re.search(r"(under|below|less than|more than|above)\s*(\d+)", query.lower())
            if m and "Hourly_Rate" in df_lab.columns:
                direction, val = m.groups()
                val = float(val)
                if direction in ["under", "below", "less than"]:
                    df_lab = df_lab[df_lab["Hourly_Rate"] <= val]
                else:
                    df_lab = df_lab[df_lab["Hourly_Rate"] >= val]

            if not df_lab.empty:
                if "Rating" in df_lab.columns:
                    df_lab["score"] = 0.7 * df_lab["similarity"] + 0.3 * (df_lab["Rating"] / 5.0)
                else:
                    df_lab["score"] = df_lab["similarity"]

                df_lab = df_lab.sort_values("score", ascending=False).head(top_k)

                # Include the same profile fields as /api/labour/search so the app
                # can show skill, experience, and jobs in Smart (recommendation) mode.
                _labour_rec_cols = [
                    "Labour_ID", "Name", "Location", "Labour_Type",
                    "Season", "Crop_Type", "Hourly_Rate", "Rating", "score",
                    "Skill_Level", "Experience_Years", "Jobs_Completed",
                    "Available_Day", "Available_Time",
                ]
                labour_results = df_lab[
                    [c for c in _labour_rec_cols if c in df_lab.columns]
                ].to_dict(orient="records")

        # 4) Equipment recommendations
        df_eq = DF_equip.copy()
        if "Equipment_Type" in df_eq.columns:
            df_eq = df_eq[df_eq["Equipment_Type"] == equip_label]

        equip_results = []
        if not df_eq.empty:
            df_eq["similarity"] = df_eq["embedding"].apply(
                lambda x: util.cos_sim(query_emb, torch.tensor(x)).item()
            )

            m = re.search(r"(under|below|less than|more than|above)\s*(\d+)", query.lower())
            if m and "Hourly_Rate_LKR" in df_eq.columns:
                direction, val = m.groups()
                val = float(val)
                if direction in ["under", "below", "less than"]:
                    df_eq = df_eq[df_eq["Hourly_Rate_LKR"] <= val]
                else:
                    df_eq = df_eq[df_eq["Hourly_Rate_LKR"] >= val]

            if not df_eq.empty:
                if "Rating" in df_eq.columns:
                    df_eq["score"] = 0.7 * df_eq["similarity"] + 0.3 * (df_eq["Rating"] / 5.0)
                else:
                    df_eq["score"] = df_eq["similarity"]

                df_eq = df_eq.sort_values("score", ascending=False).head(top_k)

                equip_results = df_eq[
                    [c for c in [
                        "Equipment_ID", "Equipment_Type", "For_Crop", "Season",
                        "Nearest_Major_District", "Condition",
                        "Hourly_Rate_LKR", "Rating", "score"
                    ] if c in df_eq.columns]
                ].to_dict(orient="records")

        # Merge in labours from Firestore that are not in the trained model
        # (e.g. newly added via the app) so they still appear in Smart search
        try:
            from labour_routes import _get_all_labour_rows, labour_row_public_visible
            firestore_labour = _get_all_labour_rows()
            model_ids = {r["Labour_ID"] for r in labour_results}
            labour_cols = [
                "Labour_ID", "Name", "Location", "Labour_Type",
                "Season", "Crop_Type", "Hourly_Rate", "Rating",
                "Skill_Level", "Experience_Years", "Jobs_Completed",
                "Available_Day", "Available_Time",
            ]
            _labour_num_cols = frozenset(
                {"Hourly_Rate", "Rating", "Experience_Years", "Jobs_Completed"}
            )
            for row in firestore_labour:
                if not labour_row_public_visible(row):
                    continue
                lid = row.get("Labour_ID") or ""
                if lid and lid not in model_ids:
                    rec = {
                        c: row.get(
                            c,
                            0 if c in _labour_num_cols else "",
                        )
                        for c in labour_cols
                    }
                    try:
                        rec["Hourly_Rate"] = float(row.get("Hourly_Rate") or 0)
                    except (TypeError, ValueError):
                        rec["Hourly_Rate"] = 0.0
                    try:
                        rec["Rating"] = float(row.get("Rating") or 0)
                    except (TypeError, ValueError):
                        rec["Rating"] = 0.0
                    try:
                        rec["Experience_Years"] = int(
                            float(row.get("Experience_Years") or 0)
                        )
                    except (TypeError, ValueError):
                        rec["Experience_Years"] = 0
                    try:
                        rec["Jobs_Completed"] = int(
                            float(row.get("Jobs_Completed") or 0)
                        )
                    except (TypeError, ValueError):
                        rec["Jobs_Completed"] = 0
                    rec["score"] = 0.25
                    labour_results.append(rec)
            labour_results.sort(key=lambda r: (-(r.get("score") or 0), -(r.get("Rating") or 0)))
            labour_results = labour_results[:top_k]
        except Exception:
            pass

        # Merge in equipments from Firestore not in the trained model
        try:
            from equipment_routes import _get_all_equipment_rows, equipment_row_public_visible
            firestore_equip = _get_all_equipment_rows()
            model_equip_ids = {r["Equipment_ID"] for r in equip_results}
            equip_cols = [
                "Equipment_ID", "Equipment_Type", "For_Crop", "Season",
                "Nearest_Major_District", "Condition",
                "Hourly_Rate_LKR", "Rating"
            ]
            for row in firestore_equip:
                if not equipment_row_public_visible(row):
                    continue
                eid = row.get("Equipment_ID") or ""
                if eid and eid not in model_equip_ids:
                    rec = {c: row.get(c, "" if c not in ("Rating", "Hourly_Rate_LKR") else 0) for c in equip_cols}
                    try:
                        rec["Hourly_Rate_LKR"] = float(row.get("Hourly_Rate_LKR") or 0)
                    except (TypeError, ValueError):
                        rec["Hourly_Rate_LKR"] = 0.0
                    try:
                        rec["Rating"] = float(row.get("Rating") or 0)
                    except (TypeError, ValueError):
                        rec["Rating"] = 0.0
                    rec["score"] = 0.25
                    equip_results.append(rec)
            equip_results.sort(key=lambda r: (-(r.get("score") or 0), -(r.get("Rating") or 0)))
            equip_results = equip_results[:top_k]
        except Exception:
            pass

        return jsonify({
            "query": query,
            "predicted_labour_type": labour_label,
            "predicted_equipment_type": equip_label,
            "labour_recommendations": labour_results,
            "equipment_recommendations": equip_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
