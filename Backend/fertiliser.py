import json
import os
from datetime import datetime
from flask import Blueprint, request, jsonify
import pandas as pd
import joblib

# ---------------------- Blueprint ---------------------- #
fertiliser_bp = Blueprint("fertiliser_bp", __name__, url_prefix="/api/fertiliser")

# ---------------------- Paths (match YOUR folder) ---------------------- #
MODELS_DIR = os.path.join("models")

SUMMARY_PATH = os.path.join(MODELS_DIR, "summary.json")
CLF_PATH = os.path.join(MODELS_DIR, "best_classifier.joblib")
REG_PATH = os.path.join(MODELS_DIR, "best_regressor.joblib")

# save last yield inside models/
LAST_YIELD_FILE = os.path.join(MODELS_DIR, "_last_yield.json")


# ---------------------- Load artifacts ---------------------- #
def load_artifacts():
    if not os.path.isdir(MODELS_DIR):
        raise RuntimeError(f"'models' folder not found at: {os.path.abspath(MODELS_DIR)}")

    if not os.path.exists(SUMMARY_PATH):
        raise RuntimeError(f"summary.json not found: {os.path.abspath(SUMMARY_PATH)}")

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    clf_model = joblib.load(CLF_PATH) if os.path.exists(CLF_PATH) else None
    reg_model = joblib.load(REG_PATH) if os.path.exists(REG_PATH) else None

    if clf_model is None and reg_model is None:
        raise RuntimeError(
            "Neither best_classifier.joblib nor best_regressor.joblib found in models/."
        )

    return summary, clf_model, reg_model


SUMMARY, CLF_MODEL, REG_MODEL = load_artifacts()


# ---------------------- Last yield helpers ---------------------- #
def save_last_yield(payload_inputs: dict, yield_kg_per_acre: float):
    try:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "yield_kg_per_acre": float(yield_kg_per_acre),
            "inputs_used": SUMMARY.get("inputs_used", []),
            "inputs_payload": payload_inputs or {},
        }
        with open(LAST_YIELD_FILE, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_last_yield():
    if not os.path.exists(LAST_YIELD_FILE):
        return None
    try:
        with open(LAST_YIELD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------- Routes ---------------------- #
@fertiliser_bp.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Fertilizer & Yield Prediction API",
        "models_dir": os.path.abspath(MODELS_DIR),
        "artifacts": {
            "summary.json": os.path.exists(SUMMARY_PATH),
            "best_classifier.joblib": os.path.exists(CLF_PATH),
            "best_regressor.joblib": os.path.exists(REG_PATH),
        },
        "endpoints": {
            "POST /api/fertiliser/predict": {
                "input_example": {
                    "temperature": 28,
                    "ph": 6.4,
                    "nitrogen": 90,
                    "phosphorous": 40,
                    "potassium": 35,
                    "crop": "paddy",
                    "growth_stage": "flowering"
                }
            },
            "POST /api/fertiliser/total-yield": {
                "body": {
                    "area_acres": 2.5,
                    "yield_kg_per_acre": "(optional) override; otherwise use the last saved prediction"
                }
            },
            "POST /api/fertiliser/total-fertiliser": {
                "body": {
                    "area_acres": 2.5,
                    "rate_kg_per_acre": 50
                }
            }
        }
    })


def build_input_dataframe(payload: dict) -> pd.DataFrame:
    """
    Build 1-row DataFrame in training feature order using SUMMARY["inputs_used"].
    """
    inputs_used = SUMMARY.get("inputs_used", [])
    row = {}
    lc_payload = {k.lower(): v for k, v in (payload or {}).items()}
    for col in inputs_used:
        row[col] = lc_payload.get(col.lower(), None)
    return pd.DataFrame([row])


@fertiliser_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Send JSON body"}), 400

    X = build_input_dataframe(data)

    result = {
        "inputs_used": SUMMARY.get("inputs_used", []),
        "fertiliser_type": None,
        "yield_kg_per_acre": None
    }

    # classification
    if CLF_MODEL is not None:
        try:
            fert_pred = CLF_MODEL.predict(X)[0]
            result["fertiliser_type"] = str(fert_pred)
        except Exception as e:
            result["fertiliser_type_error"] = str(e)

    # regression
    if REG_MODEL is not None:
        try:
            y_pred = REG_MODEL.predict(X)[0]
            y_pred = float(round(y_pred, 2))
            result["yield_kg_per_acre"] = y_pred
            save_last_yield(payload_inputs=data, yield_kg_per_acre=y_pred)
        except Exception as e:
            result["yield_error"] = str(e)

    return jsonify(result)


@fertiliser_bp.route("/total-yield", methods=["POST"])
def total_yield():
    body = request.get_json(silent=True) or {}
    if "area_acres" not in body:
        return jsonify({"error": "area_acres is required"}), 400

    area = float(body["area_acres"])

    if "yield_kg_per_acre" in body and body["yield_kg_per_acre"] is not None:
        ypa = float(body["yield_kg_per_acre"])
        source = "provided"
    else:
        last = load_last_yield()
        if not last or "yield_kg_per_acre" not in last:
            return jsonify({"error": "No last yield found. Call /predict first or provide yield_kg_per_acre."}), 400
        ypa = float(last["yield_kg_per_acre"])
        source = "last_prediction"

    total = round(ypa * area, 2)
    return jsonify({
        "area_acres": area,
        "yield_kg_per_acre": ypa,
        "total_yield_kg": total,
        "source": source
    })


@fertiliser_bp.route("/total-fertiliser", methods=["POST"])
def total_fertiliser():
    body = request.get_json(silent=True) or {}
    missing = [k for k in ["area_acres", "rate_kg_per_acre"] if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    area = float(body["area_acres"])
    rate = float(body["rate_kg_per_acre"])
    total = round(area * rate, 2)

    return jsonify({
        "area_acres": area,
        "rate_kg_per_acre": rate,
        "total_fertiliser_kg": total
    })
