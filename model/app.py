#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, math, joblib, numpy as np, pandas as pd, requests
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# ---------------------- Config ---------------------- #
ARTIFACT_PATH = os.environ.get("MODEL_PATH", os.path.join("outputs", "models", "best_model.joblib"))
PORT = int(os.environ.get("PORT", "5003"))
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")  # optional fallback

REQUIRED_FEATURES = ["N", "P", "K", "PH", "Temperature", "Humidity", "Rainfall"]

# ---------------------- Load model ---------------------- #
if not os.path.isfile(ARTIFACT_PATH):
    raise FileNotFoundError(
        f"Model artifact not found at '{ARTIFACT_PATH}'. "
        "Run train.py first or set MODEL_PATH."
    )

artifact = joblib.load(ARTIFACT_PATH)
PIPE = artifact["pipeline"]
LE = artifact["label_encoder"]
FEATURES = artifact.get("features", REQUIRED_FEATURES)
CLASSES = list(artifact.get("classes", LE.classes_))

# ---------------------- App init ---------------------- #
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)

# ---------------------- Helpers ---------------------- #
CANON_FEATURES = {
    "n": "N", "nitrogen": "N",
    "p": "P", "phosphorus": "P",
    "k": "K", "potassium": "K",
    "ph": "PH", "soilph": "PH", "soilh": "PH",
    "temperature": "Temperature", "temp": "Temperature",
    "humidity": "Humidity", "humidty": "Humidity",
    "rainfall": "Rainfall", "rain": "Rainfall"
}

def normalize_key(k: str) -> str:
    s = re.sub(r"\(.*?\)", "", k or "", flags=re.IGNORECASE)
    s = s.replace("°", "").replace("%", "")
    s = s.strip().replace(" ", "").replace("-", "").replace("_", "").lower()
    s = re.sub(r"(mm|cm)$", "", s)
    if s in CANON_FEATURES:
        return CANON_FEATURES[s]
    if s.startswith("temperature"):
        return "Temperature"
    if s.startswith("humidity"):
        return "Humidity"
    if s.startswith("rain") or s.startswith("rainfall"):
        return "Rainfall"
    return k

def coerce_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    try:
        return float(str(v).strip())
    except Exception:
        return None

def normalize_payload(obj: dict) -> dict:
    out = {}
    for k, v in obj.items():
        key = normalize_key(k)
        out[key] = coerce_float(v) if key in REQUIRED_FEATURES else v
    # allow latitude/longitude aliases
    if "lat" in obj and "latitude" not in obj:
        out["latitude"] = coerce_float(obj.get("lat"))
    if "lon" in obj and "longitude" not in obj:
        out["longitude"] = coerce_float(obj.get("lon"))
    if "Latitude" in obj and "latitude" not in out:
        out["latitude"] = coerce_float(obj.get("Latitude"))
    if "Longitude" in obj and "longitude" not in out:
        out["longitude"] = coerce_float(obj.get("Longitude"))
    return out

def rows_from_request_json(js):
    if isinstance(js, dict):
        if "instances" in js and isinstance(js["instances"], list):
            return js["instances"]
        return [js]
    if isinstance(js, list):
        return js
    raise ValueError("Send a single JSON object, a list of objects, or {\"instances\": [...]}.")

# -------- Rainfall fetchers (mm in last 24h) -------- #
def fetch_rainfall_open_meteo(lat: float, lon: float, hours: int = 24) -> float:
    """
    Open-Meteo free endpoint. Returns precipitation sum (mm) over the last `hours`.
    """
    # We fetch 48h to be safe around hour boundaries then slice last `hours`
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation"
        "&past_days=2"
        "&forecast_days=1"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    js = r.json()
    times = js.get("hourly", {}).get("time", [])
    prec = js.get("hourly", {}).get("precipitation", [])
    if not times or not prec or len(times) != len(prec):
        raise ValueError("Open-Meteo: missing hourly precipitation")
    # take last `hours` samples
    values = prec[-hours:]
    return float(np.nansum(values))

def fetch_rainfall_openweather(lat: float, lon: float, hours: int = 24) -> float:
    """
    Uses OpenWeather One Call 3.0 'hourly' to sum last 24h precipitation (mm).
    Requires OPENWEATHER_API_KEY.
    """
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not set")
    url = (
        "https://api.openweathermap.org/data/3.0/onecall"
        f"?lat={lat}&lon={lon}&exclude=minutely,daily,alerts"
        f"&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    js = r.json()
    hourly = js.get("hourly", [])
    if not hourly:
        raise ValueError("OpenWeather: no hourly data")
    # Take last `hours` entries (API returns future+past relative to current)
    # We'll filter by dt <= now and keep last 24
    now_ts = int(datetime.now(timezone.utc).timestamp())
    past_hours = [h for h in hourly if int(h.get("dt", 0)) <= now_ts]
    values = []
    for h in past_hours[-hours:]:
        # precipitation may be provided as "rain": {"1h": x} or "snow" or "precipitation"
        mm = 0.0
        if isinstance(h.get("rain"), dict) and "1h" in h["rain"]:
            mm += float(h["rain"]["1h"])
        if isinstance(h.get("snow"), dict) and "1h" in h["snow"]:
            mm += float(h["snow"]["1h"])
        # some responses have "pop" (probability) but not accumulation; ignore pop
        values.append(mm)
    return float(np.nansum(values))

def get_rainfall_mm(lat: float, lon: float, hours: int = 24) -> float:
    """
    Tries Open-Meteo first (no key). If that fails and OpenWeather key is set, tries OpenWeather.
    Returns mm over last `hours`.
    """
    try:
        return fetch_rainfall_open_meteo(lat, lon, hours=hours)
    except Exception as e1:
        if OPENWEATHER_API_KEY:
            try:
                return fetch_rainfall_openweather(lat, lon, hours=hours)
            except Exception as e2:
                raise RuntimeError(f"Rainfall fetch failed (Open-Meteo: {e1}; OpenWeather: {e2})")
        raise RuntimeError(f"Rainfall fetch failed (Open-Meteo: {e1})")

# ---------------------- Routes ---------------------- #
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_name": artifact.get("model_name", type(PIPE).__name__),
        "features": FEATURES,
        "classes": CLASSES,
        "artifact_path": os.path.abspath(ARTIFACT_PATH),
        "time": datetime.utcnow().isoformat() + "Z"
    })

@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({"classes": CLASSES})

@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "required_features": REQUIRED_FEATURES,
        "accepts": [
            "N, P, K, PH, Temperature, Humidity, Rainfall",
            "Or provide latitude/longitude to auto-fill Rainfall"
        ],
        "input_formats": ["Single JSON object", "{\"instances\": [ {...}, ... ]}", "[{...}, ...]"],
        "location_keys": ["lat, lon", "latitude, longitude"]
    })

@app.route("/rainfall", methods=["POST"])
def rainfall_endpoint():
    js = request.get_json(silent=True) or {}
    lat = js.get("latitude") or js.get("lat")
    lon = js.get("longitude") or js.get("lon")
    hours = int(js.get("hours", 24))
    if lat is None or lon is None:
        return jsonify({"error": "Provide 'lat'/'lon' or 'latitude'/'longitude'."}), 400
    lat = float(lat); lon = float(lon)
    mm = get_rainfall_mm(lat, lon, hours=hours)
    return jsonify({"lat": lat, "lon": lon, "hours": hours, "rainfall_mm": mm})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        js = request.get_json(silent=True)
        if js is None:
            return jsonify({"error": "Invalid or missing JSON"}), 400

        raw_rows = rows_from_request_json(js)
        results = []

        for row in raw_rows:
            nr = normalize_payload(row)

            # If Rainfall is missing, try to fetch with lat/lon
            rainfall = nr.get("Rainfall", None)
            if rainfall is None:
                lat = nr.get("latitude")
                lon = nr.get("longitude")
                if lat is not None and lon is not None:
                    try:
                        rainfall = get_rainfall_mm(float(lat), float(lon), hours=int(row.get("hours", 24)))
                        nr["Rainfall"] = rainfall
                    except Exception as e:
                        return jsonify({"error": f"Could not fetch rainfall: {e}"}), 502

            # Check required numeric features
            missing = [f for f in REQUIRED_FEATURES if nr.get(f, None) is None]
            if missing:
                return jsonify({
                    "error": "Missing or non-numeric features",
                    "missing": missing,
                    "hint": "Provide Rainfall or supply latitude/longitude to auto-fill it."
                }), 400

            X = pd.DataFrame([{f: nr[f] for f in FEATURES}], columns=FEATURES)
            pred_idx = PIPE.predict(X)[0]
            label = LE.inverse_transform([pred_idx])[0]

            # probs (best effort)
            probs = None
            if hasattr(PIPE, "predict_proba"):
                try:
                    probs = PIPE.predict_proba(X)[0]
                except Exception:
                    probs = None

            if probs is None and hasattr(PIPE, "decision_function"):
                try:
                    dec = PIPE.decision_function(X)
                    if dec.ndim == 1:
                        dec = np.vstack([-dec, dec]).T
                    s = np.exp(dec - dec.max(axis=1, keepdims=True))
                    probs = (s / (s.sum(axis=1, keepdims=True) + 1e-12))[0]
                except Exception:
                    probs = None

            item = {
                "prediction": label,
                "features_used": {f: float(nr[f]) for f in FEATURES},
            }
            if "latitude" in nr and "longitude" in nr:
                item["location"] = {"lat": float(nr["latitude"]), "lon": float(nr["longitude"])}
            if probs is not None:
                order = np.argsort(probs)[::-1]
                top = min(5, len(CLASSES))
                item["probs"] = [{"class": CLASSES[j], "prob": float(probs[j])} for j in order[:top]]

            results.append(item)

        return jsonify({
            "ok": True,
            "count": len(results),
            "results": results,
            "model_name": artifact.get("model_name", type(PIPE).__name__),
            "generated_at": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
