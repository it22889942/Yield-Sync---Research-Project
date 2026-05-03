#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS

PORT = int(os.environ.get("PORT", "5003"))

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)

# ✅ CORS for Flutter Web / Browser + Mobile
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ✅ import and register all model route files
from crop import crop_bp
from fertiliser import fertiliser_bp
from yieldsync_routes import yieldsync_bp
from labour_routes import labour_bp

# ✅ NEW: recommendation blueprint
from recommendation_routes import recommendation_bp

# ✅ NEW: equipment blueprint
from equipment_routes import equipment_bp

app.register_blueprint(crop_bp)
app.register_blueprint(fertiliser_bp)
app.register_blueprint(yieldsync_bp)
app.register_blueprint(recommendation_bp)
app.register_blueprint(labour_bp)

# ✅ REGISTER equipment
app.register_blueprint(equipment_bp)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Multi-Model Backend"})

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "service": "Multi-Model Backend",
        "routes": {
            "crop": [
                "/api/crop/health",
                "/api/crop/schema",
                "/api/crop/labels",
                "/api/crop/rainfall",
                "/api/crop/predict"
            ],
            "fertiliser": [
                "/api/fertiliser/",
                "/api/fertiliser/predict",
                "/api/fertiliser/total-yield",
                "/api/fertiliser/total-fertiliser"
            ],
            "yieldsync": [
                "/api/yieldsync/health",
                "/api/yieldsync/crops",
                "/api/yieldsync/markets?crop=Rice",
                "/api/yieldsync/current-price?crop=Rice&market=Colombo",
                "/api/yieldsync/predict",
                "/api/yieldsync/recommendation",
                "/api/yieldsync/history?crop=Rice&market=Colombo&days=30"
            ],
            "recommendation": [
                "/api/recommend/health",
                "/api/recommend/recommend",
                "/api/recommend/train"
            ],
            # ✅ NEW: equipment routes
            "equipment": [
                "/api/equipment/health",
                "/api/equipment/locations",
                "/api/equipment/types",
                "/api/equipment/search",
                "/api/equipment/pending",
                "POST /api/equipment/<id>/moderate"
            ],
            "labour": [
                "/api/labour/health",
                "/api/labour/locations",
                "/api/labour/skills",
                "/api/labour/search",
                "/api/labour/pending",
                "POST /api/labour/<id>/moderate"
            ],
        }
    })

if __name__ == "__main__":
    _rules = {str(r) for r in app.url_map.iter_rules()}
    _lp = "/api/labour/pending" in _rules
    _ep = "/api/equipment/pending" in _rules
    print(
        f"[YieldSync] moderation API: "
        f"/api/labour/pending={'ok' if _lp else 'MISSING — restart after code updates'}, "
        f"/api/equipment/pending={'ok' if _ep else 'MISSING'}"
    )
    # Heavy ML imports make auto-reload expensive; opt in with FLASK_RELOAD=1
    _reload = os.environ.get("FLASK_RELOAD", "").strip() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=_reload)
