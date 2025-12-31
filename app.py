import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# ---- CONFIG ----
OUTPUTS_DIR = "outputs"   # parent folder
# Auto-pick the latest timestamp folder inside outputs (e.g., outputs/20251108_113015)
MODEL_DIR = None

app = Flask(__name_


def find_latest_output_dir(base="outputs"):
    """
    Pick latest timestamp folder under outputs/.
    """
    if not os.path.isdir(base):
        return None
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


# load artifacts at startup
def load_artifacts():
    global MODEL_DIR
    if MODEL_DIR is None:
        MODEL_DIR = find_latest_output_dir(OUTPUTS_DIR)
    if MODEL_DIR is None:
        raise RuntimeError("No trained model folder found in 'outputs/'. Run train.py first.")

    summary_path = os.path.join(MODEL_DIR, "summary.json")
    clf_path = os.path.join(MODEL_DIR, "best_classifier.joblib")
    reg_path = os.path.join(MODEL_DIR, "best_regressor.joblib")

    if not os.path.exists(summary_path):
        raise RuntimeError(f"summary.json not found in {MODEL_DIR}.")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # some projects train only one of them – so load softly
    clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None
    reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None

    return summary, clf_model, reg_model


SUMMARY, CLF_MODEL, REG_MODEL = load_artifacts()

# ---- NEW: simple persistence for last yield prediction ----
LAST_YIELD_FILE = os.path.join(MODEL_DIR, "_last_yield.json")

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
        # non-fatal
        pass

def load_last_yield():
    if not os.path.exists(LAST_YIELD_FILE):
        return None
    try:
        with open(LAST_YIELD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Fertilizer & Yield Prediction API",
        "endpoints": {
            "POST /predict": {
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
            "POST /total-yield": {
                "body": {
                    "area_acres": 2.5,
                    "yield_kg_per_acre": "(optional) override; otherwise use the last saved prediction"
                }
            },
            "POST /total-fertilizer": {
                "body": {
                    "area_acres": 2.5,
                    "rate_kg_per_acre": 50
                }
            }
        }
    })


def build_input_dataframe(payload: dict) -> pd.DataFrame:
    """
    We trained the model with column names in SUMMARY["inputs_used"].
    Build a 1-row DataFrame and reorder columns to match training order.
    """
    inputs_used = SUMMARY.get("inputs_used", [])
    row = {}
    lc_payload = {k.lower(): v for k, v in (payload or {}).items()}
    for col in inputs_used:
        row[col] = lc_payload.get(col.lower(), None)
    return pd.DataFrame([row])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Send JSON body"}), 400

    # build 1-row df in the exact feature order
    X = build_input_dataframe(data)

    result = {
        "inputs_used": SUMMARY.get("inputs_used", []),
        "fertilizer_type": None,
        "yield_kg_per_acre": None
    }

    # classification
    if CLF_MODEL is not None:
        try:
            fert_pred = CLF_MODEL.predict(X)[0]
            result["fertilizer_type"] = str(fert_pred)
        except Exception as e:
            result["fertilizer_type_error"] = str(e)

    # regression
    if REG_MODEL is not None:
        try:
            y_pred = REG_MODEL.predict(X)[0]
            y_pred = float(round(y_pred, 2))
            result["yield_kg_per_acre"] = y_pred

            # ---- NEW: save locally for later /total-yield calculations
            save_last_yield(payload_inputs=data, yield_kg_per_acre=y_pred)

        except Exception as e:
            result["yield_error"] = str(e)

    return jsonify(result)


# ---- NEW: multiply last (or provided) per-acre yield by area (acres) ----
@app.route("/total-yield", methods=["POST"])
def total_yield():
    """
    Body:
    {
      "area_acres": 3.0,
      "yield_kg_per_acre": 2500.0   # optional; if omitted we use last saved prediction
    }
    Response:
    {
      "area_acres": 3.0,
      "yield_kg_per_acre": 2500.0,
      "total_yield_kg": 7500.0,
      "source": "provided" | "last_prediction"
    }
    """
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


# ---- NEW: utility — total fertilizer needed for an area given a rate per acre ----
@app.route("/total-fertilizer", methods=["POST"])
def total_fertilizer():
    """
    Body:
    {
      "area_acres": 3.0,
      "rate_kg_per_acre": 50.0
    }
    Response:
    {
      "area_acres": 3.0,
      "rate_kg_per_acre": 50.0,
      "total_fertilizer_kg": 150.0
    }
    """
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
        "total_fertilizer_kg": total
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
