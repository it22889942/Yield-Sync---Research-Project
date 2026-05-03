# # yieldsync_routes.py
# import os
# from flask import Blueprint, request, jsonify

# # Use the existing wrapper/singleton from your uploaded api.py
# # (Do NOT change logic inside api.py / predictor.py)
# from yieldsync.api import get_api

# # Optional: data updater (same logic, just exposed as endpoints)
# try:
#     from yieldsync.data_fetcher import update_this_week, update_since_date
#     HAS_FETCHER = True
# except Exception:
#     HAS_FETCHER = False


# yieldsync_bp = Blueprint("yieldsync_bp", __name__, url_prefix="/api/yieldsync")


# def _q(name, default=None):
#     """Read from query string first, then JSON body."""
#     if request.args.get(name) is not None:
#         return request.args.get(name)
#     body = request.get_json(silent=True) or {}
#     return body.get(name, default)


# @yieldsync_bp.get("/health")
# def health():
#     return jsonify({"status": "ok", "service": "YieldSync"})


# @yieldsync_bp.get("/crops")
# def crops():
#     api = get_api()
#     return jsonify({"crops": api.crops, "horizons": api.horizons})


# @yieldsync_bp.get("/markets")
# def markets():
#     crop = _q("crop")
#     if not crop:
#         return jsonify({"error": "crop is required"}), 400
#     api = get_api()
#     return jsonify({"crop": crop, "markets": api.get_markets(crop)})


# @yieldsync_bp.get("/current-price")
# def current_price():
#     crop = _q("crop")
#     market = _q("market")
#     if not crop:
#         return jsonify({"error": "crop is required"}), 400

#     api = get_api()
#     result = api.get_current_price(crop, market)  # uses existing logic
#     status = 400 if "error" in result else 200
#     return jsonify(result), status


# @yieldsync_bp.post("/predict")
# def predict():
#     crop = _q("crop")
#     market = _q("market")
#     days_ahead = _q("days_ahead", 7)

#     if not crop:
#         return jsonify({"error": "crop is required"}), 400

#     try:
#         days_ahead = int(days_ahead)
#     except Exception:
#         return jsonify({"error": "days_ahead must be an integer"}), 400

#     api = get_api()
#     result = api.predict(crop, market, days_ahead)  # existing logic
#     status = 400 if "error" in result else 200
#     return jsonify(result), status


# @yieldsync_bp.post("/recommendation")
# def recommendation():
#     crop = _q("crop")
#     market = _q("market")
#     days_ahead = _q("days_ahead", 7)
#     quantity_kg = _q("quantity_kg", 1000)

#     if not crop:
#         return jsonify({"error": "crop is required"}), 400

#     try:
#         days_ahead = int(days_ahead)
#         quantity_kg = float(quantity_kg)
#     except Exception:
#         return jsonify({"error": "days_ahead must be int and quantity_kg must be number"}), 400

#     api = get_api()
#     result = api.get_recommendation(crop, market, days_ahead, quantity_kg)
#     status = 400 if "error" in result else 200
#     return jsonify(result), status


# @yieldsync_bp.get("/history")
# def history():
#     crop = _q("crop")
#     market = _q("market")
#     days = _q("days", 30)

#     if not crop:
#         return jsonify({"error": "crop is required"}), 400

#     try:
#         days = int(days)
#     except Exception:
#         return jsonify({"error": "days must be an integer"}), 400

#     api = get_api()
#     result = api.get_price_history(crop, market, days)
#     status = 400 if "error" in result else 200
#     return jsonify(result), status


# # ----------------------------
# # Optional: Data update routes
# # ----------------------------
# @yieldsync_bp.post("/data/update-this-week")
# def api_update_this_week():
#     if not HAS_FETCHER:
#         return jsonify({"error": "data_fetcher not available (install requests + pdfplumber)"}), 400

#     # Optional custom path if your dataset lives elsewhere
#     data_path = _q("data_path")
#     update_this_week(data_path=data_path)  # existing logic
#     return jsonify({"status": "ok", "message": "Updated this week"})


# @yieldsync_bp.post("/data/update-since")
# def api_update_since():
#     if not HAS_FETCHER:
#         return jsonify({"error": "data_fetcher not available (install requests + pdfplumber)"}), 400

#     start_date = _q("start_date")
#     data_path = _q("data_path")

#     if not start_date:
#         return jsonify({"error": "start_date is required (YYYY-MM-DD)"}), 400

#     update_since_date(start_date=start_date, data_path=data_path)  # existing logic
#     return jsonify({"status": "ok", "message": f"Updated since {start_date}"})

# # ADD THIS in yieldsync_routes.py (after /history route)

# @yieldsync_bp.get("/avg-by-market")
# def avg_by_market():
#     """
#     Example:
#       /api/yieldsync/avg-by-market?crop=Radish&days=7
#     Returns per-market avg/min/max for last N days.
#     """
#     crop = _q("crop")
#     days = _q("days", 7)

#     if not crop:
#         return jsonify({"error": "crop is required"}), 400

#     try:
#         days = int(days)
#         if days <= 0:
#             raise ValueError()
#     except Exception:
#         return jsonify({"error": "days must be a positive integer"}), 400

#     api = get_api()

#     # Use YieldSyncAPI's loaded dataframe (do NOT change api.py logic)
#     try:
#         df = api._load_data()
#     except Exception as e:
#         return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

#     # Validate columns
#     needed = {"Date", "item", "market", "price"}
#     if not needed.issubset(set(df.columns)):
#         return jsonify({"error": f"Dataset missing columns: {sorted(list(needed - set(df.columns)))}"}), 500

#     # Filter crop
#     dff = df[df["item"] == crop].copy()
#     if dff.empty:
#         return jsonify({"error": f"No data for crop: {crop}"}), 400

#     dff["Date"] = dff["Date"] if str(dff["Date"].dtype).startswith("datetime") else dff["Date"]
#     # Ensure datetime
#     try:
#         import pandas as pd
#         dff["Date"] = pd.to_datetime(dff["Date"])
#     except Exception:
#         pass

#     max_date = dff["Date"].max()
#     try:
#         import pandas as pd
#         cutoff = max_date - pd.Timedelta(days=days)
#         dff = dff[dff["Date"] >= cutoff]
#     except Exception:
#         # if anything fails, just use all
#         pass

#     if dff.empty:
#         return jsonify({"error": f"No data in last {days} days for {crop}"}), 400

#     # Group by market
#     g = dff.groupby("market")["price"].agg(["mean", "min", "max", "count"]).reset_index()
#     g = g.sort_values("mean", ascending=False)

#     rows = []
#     for _, r in g.iterrows():
#         rows.append({
#             "market": str(r["market"]),
#             "avg_price": round(float(r["mean"]), 2),
#             "min_price": round(float(r["min"]), 2),
#             "max_price": round(float(r["max"]), 2),
#             "count": int(r["count"]),
#         })

#     top3 = rows[:3]

#     return jsonify({
#         "crop": crop,
#         "days": days,
#         "rows": rows,
#         "top": top3,
#         "max_date": str(getattr(max_date, "date", lambda: max_date)()),
#     })

# # ✅ ADD THIS NEW ENDPOINT (paste near other routes)
# @yieldsync_bp.get("/trends")
# def trends():
#     """
#     GET /api/yieldsync/trends?crop=All&market=All&max_points=400&recent=50
#     Returns:
#       - stats: total_records, days_of_data, first_date, last_date
#       - series: [{label, dates[], prices[]}, ...]
#       - recent: [{date, market, item, price}, ...]
#     """
#     crop = _q("crop", "All")
#     market = _q("market", "All")
#     max_points = _q("max_points", 400)
#     recent_n = _q("recent", 50)

#     try:
#         max_points = int(max_points)
#     except Exception:
#         max_points = 400

#     try:
#         recent_n = int(recent_n)
#     except Exception:
#         recent_n = 50

#     api = get_api()
#     df = api._load_data()  # uses existing cached loader in api.py

#     # Ensure Date datetime (api already does, but safe)
#     try:
#         import pandas as pd
#         df = df.copy()
#         df["Date"] = pd.to_datetime(df["Date"])
#     except Exception:
#         pass

#     if df is None or len(df) == 0:
#         return jsonify({"error": "No data available"}), 400

#     # Stats on FULL dataset (not filtered)
#     dmin = df["Date"].min()
#     dmax = df["Date"].max()
#     days_of_data = int((dmax.date() - dmin.date()).days) + 1

#     stats = {
#         "total_records": int(len(df)),
#         "days_of_data": days_of_data,
#         "first_date": str(dmin.date()),
#         "last_date": str(dmax.date()),
#     }

#     # Apply filters for charts/recent
#     f = df
#     if crop and crop != "All":
#         f = f[f["item"] == crop]
#     if market and market != "All":
#         f = f[f["market"] == market]

#     # Build time-series
#     series = []
#     crops_list = api.crops if crop == "All" else [crop]

#     for c in crops_list:
#         sub = f if crop != "All" else f[f["item"] == c]
#         if sub is None or len(sub) == 0:
#             continue

#         # Daily average
#         daily = sub.groupby("Date")["price"].mean().reset_index()
#         daily = daily.sort_values("Date")

#         # Downsample for mobile chart performance
#         n = len(daily)
#         if max_points and n > max_points:
#             import math
#             step = int(math.ceil(n / max_points))
#             daily = daily.iloc[::step, :]

#         series.append({
#             "label": c,
#             "dates": daily["Date"].dt.strftime("%Y-%m-%d").tolist(),
#             "prices": daily["price"].round(2).tolist(),
#             "count": int(len(daily)),
#         })

#     # Recent entries (filtered)
#     recent_df = f.sort_values("Date", ascending=False).head(recent_n)
#     recent_rows = []
#     for _, r in recent_df.iterrows():
#         recent_rows.append({
#             "date": str(r["Date"]),
#             "market": str(r.get("market", "")),
#             "item": str(r.get("item", "")),
#             "price": float(r.get("price", 0)),
#         })

#     return jsonify({
#         "crop": crop,
#         "market": market,
#         "stats": stats,
#         "series": series,
#         "recent": recent_rows,
#     }), 200
# yieldsync_routes.py
import os
from flask import Blueprint, request, jsonify

# Use the existing wrapper/singleton from your uploaded api.py
# (Do NOT change logic inside api.py / predictor.py)
from yieldsync.api import get_api

# Optional: data updater (same logic, just exposed as endpoints)
try:
    from yieldsync.data_fetcher import update_this_week, update_since_date, update_complete_weeks_only
    HAS_FETCHER = True
except Exception:
    HAS_FETCHER = False

# Optional: model retraining
try:
    from yieldsync.trainer import retrain_models
    HAS_TRAINER = True
except Exception:
    HAS_TRAINER = False


yieldsync_bp = Blueprint("yieldsync_bp", __name__, url_prefix="/api/yieldsync")


def _q(name, default=None):
    """Read from query string first, then JSON body."""
    if request.args.get(name) is not None:
        return request.args.get(name)
    body = request.get_json(silent=True) or {}
    return body.get(name, default)


@yieldsync_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "YieldSync"})


@yieldsync_bp.get("/crops")
def crops():
    api = get_api()
    return jsonify({"crops": api.crops, "horizons": api.horizons})


@yieldsync_bp.get("/markets")
def markets():
    crop = _q("crop")
    if not crop:
        return jsonify({"error": "crop is required"}), 400
    api = get_api()
    return jsonify({"crop": crop, "markets": api.get_markets(crop)})


@yieldsync_bp.get("/current-price")
def current_price():
    crop = _q("crop")
    market = _q("market")
    if not crop:
        return jsonify({"error": "crop is required"}), 400

    api = get_api()
    result = api.get_current_price(crop, market)  # uses existing logic
    status = 400 if "error" in result else 200
    return jsonify(result), status


@yieldsync_bp.post("/predict")
def predict():
    crop = _q("crop")
    market = _q("market")
    days_ahead = _q("days_ahead", 7)

    if not crop:
        return jsonify({"error": "crop is required"}), 400

    try:
        days_ahead = int(days_ahead)
    except Exception:
        return jsonify({"error": "days_ahead must be an integer"}), 400

    api = get_api()
    result = api.predict(crop, market, days_ahead)  # existing logic
    status = 400 if "error" in result else 200
    return jsonify(result), status


@yieldsync_bp.post("/recommendation")
def recommendation():
    crop = _q("crop")
    market = _q("market")
    days_ahead = _q("days_ahead", 7)
    quantity_kg = _q("quantity_kg", 1000)

    if not crop:
        return jsonify({"error": "crop is required"}), 400

    try:
        days_ahead = int(days_ahead)
        quantity_kg = float(quantity_kg)
    except Exception:
        return jsonify({"error": "days_ahead must be int and quantity_kg must be number"}), 400

    api = get_api()
    result = api.get_recommendation(crop, market, days_ahead, quantity_kg)
    status = 400 if "error" in result else 200
    return jsonify(result), status


@yieldsync_bp.get("/history")
def history():
    crop = _q("crop")
    market = _q("market")
    days = _q("days", 30)

    if not crop:
        return jsonify({"error": "crop is required"}), 400

    try:
        days = int(days)
    except Exception:
        return jsonify({"error": "days must be an integer"}), 400

    api = get_api()
    result = api.get_price_history(crop, market, days)
    status = 400 if "error" in result else 200
    return jsonify(result), status


# ----------------------------
# Optional: Data update routes
# ----------------------------
@yieldsync_bp.post("/data/update-this-week")
def api_update_this_week():
    if not HAS_FETCHER:
        return jsonify({"error": "data_fetcher not available (install requests + pdfplumber)"}), 400

    # Optional custom path if your dataset lives elsewhere
    data_path = _q("data_path")
    update_this_week(data_path=data_path)  # existing logic
    return jsonify({"status": "ok", "message": "Updated this week"})


@yieldsync_bp.post("/data/update-since")
def api_update_since():
    if not HAS_FETCHER:
        return jsonify({"error": "data_fetcher not available (install requests + pdfplumber)"}), 400

    start_date = _q("start_date")
    data_path = _q("data_path")

    if not start_date:
        return jsonify({"error": "start_date is required (YYYY-MM-DD)"}), 400

    update_since_date(start_date=start_date, data_path=data_path)  # existing logic
    return jsonify({"status": "ok", "message": f"Updated since {start_date}"})


@yieldsync_bp.post("/data/update-smart")
def api_update_smart():
    """
    SMART UPDATE: Only fetches COMPLETE weeks (excludes current incomplete week).
    
    This is the recommended way to update data - it automatically:
    - Detects the last complete week (last Sunday)
    - Fetches only complete weeks to avoid incomplete data
    - Auto-detects start date from existing data
    - Returns detailed status info
    
    Usage:
      POST /api/yieldsync/data/update-smart
      
      Optional body:
      {
        "start_date": "2026-01-01",  // optional, auto-detects if not provided
        "data_path": "/custom/path"   // optional
      }
    """
    if not HAS_FETCHER:
        return jsonify({"error": "data_fetcher not available (install requests + pdfplumber)"}), 400

    start_date = _q("start_date")  # Optional
    data_path = _q("data_path")     # Optional
    
    try:
        result = update_complete_weeks_only(start_date=start_date, data_path=data_path)
        
        # Return non-error HTTP status for expected "no new data" states.
        if result.get("status") in ("success", "no_update_needed", "no_data"):
            return jsonify(result), 200
        return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "Failed to update data"
        }), 500


@yieldsync_bp.post("/models/retrain")
def api_retrain_models():
    """
    RETRAIN MODELS: Retrain all price forecasting models with latest data.
    
    Use this after updating data with /data/update-smart to ensure models
    learn from the freshest market patterns.
    
    ⚠️ WARNING: This can take 10-30 minutes depending on data size!
    Models trained:
    - Per-crop models (Rice, Beetroot, Radish, Red Onion)
    - Per-market models (20+ markets)
    - Multi-horizon (7, 14, 30, 60, 84 days)
    - Total: ~200+ models
    
    Usage:
      POST /api/yieldsync/models/retrain
      
      Optional body:
      {
        "data_path": "/custom/path",      // optional
        "save_dir": "/custom/models"      // optional
      }
      
    Returns:
      {
        "status": "success",
        "models_trained": 215,
        "duration_minutes": 18.5,
        "timestamp": "2026-02-27T10:30:00Z"
      }
    """
    if not HAS_TRAINER:
        return jsonify({
            "error": "Model trainer not available",
            "hint": "Check if scikit-learn, lightgbm, tensorflow are installed"
        }), 400
    
    data_path = _q("data_path")
    save_dir = _q("save_dir")
    
    # Auto-detect paths if not provided
    # NOTE: this routes file lives in Backend/, so default assets should resolve
    # under Backend/yieldsync/... (not project_root/yieldsync/...).
    if data_path is None:
        import os
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(
            backend_dir,
            "yieldsync",
            "data",
            "full_history_features_real_weather.csv",
        )
    
    if save_dir is None:
        import os
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(backend_dir, "yieldsync", "models", "saved_models")
    
    try:
        import time
        start_time = time.time()
        
        # Run training (this will take a while!)
        result = retrain_models(
            price_data_path=data_path,
            save_dir=save_dir,
            progress_callback=None  # Could add logging here
        )
        
        duration_mins = (time.time() - start_time) / 60
        
        # Enhance result with timing info
        result["duration_minutes"] = round(duration_mins, 2)
        result["data_path"] = data_path
        result["save_dir"] = save_dir
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "message": "Model retraining failed"
        }), 500


# ----------------------------
# Analytics routes
# ----------------------------

@yieldsync_bp.get("/avg-by-market")
def avg_by_market():
    """
    Example:
      /api/yieldsync/avg-by-market?crop=Radish&days=7
    Returns per-market avg/min/max for last N days.
    """
    crop = _q("crop")
    days = _q("days", 7)

    if not crop:
        return jsonify({"error": "crop is required"}), 400

    try:
        days = int(days)
        if days <= 0:
            raise ValueError()
    except Exception:
        return jsonify({"error": "days must be a positive integer"}), 400

    api = get_api()

    # Use YieldSyncAPI's loaded dataframe (do NOT change api.py logic)
    try:
        df = api._load_data()
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

    # Validate columns
    needed = {"Date", "item", "market", "price"}
    if not needed.issubset(set(df.columns)):
        return jsonify({"error": f"Dataset missing columns: {sorted(list(needed - set(df.columns)))}"}), 500

    # Filter crop
    dff = df[df["item"] == crop].copy()
    if dff.empty:
        return jsonify({"error": f"No data for crop: {crop}"}), 400

    dff["Date"] = dff["Date"] if str(dff["Date"].dtype).startswith("datetime") else dff["Date"]
    # Ensure datetime
    try:
        import pandas as pd
        dff["Date"] = pd.to_datetime(dff["Date"])
    except Exception:
        pass

    max_date = dff["Date"].max()
    try:
        import pandas as pd
        cutoff = max_date - pd.Timedelta(days=days)
        dff = dff[dff["Date"] >= cutoff]
    except Exception:
        # if anything fails, just use all
        pass

    if dff.empty:
        return jsonify({"error": f"No data in last {days} days for {crop}"}), 400

    # Group by market
    g = dff.groupby("market")["price"].agg(["mean", "min", "max", "count"]).reset_index()
    g = g.sort_values("mean", ascending=False)

    rows = []
    for _, r in g.iterrows():
        rows.append({
            "market": str(r["market"]),
            "avg_price": round(float(r["mean"]), 2),
            "min_price": round(float(r["min"]), 2),
            "max_price": round(float(r["max"]), 2),
            "count": int(r["count"]),
        })

    top3 = rows[:3]

    return jsonify({
        "crop": crop,
        "days": days,
        "rows": rows,
        "top": top3,
        "max_date": str(getattr(max_date, "date", lambda: max_date)()),
    })

# ✅ ADD THIS NEW ENDPOINT (paste near other routes)
@yieldsync_bp.get("/trends")
def trends():
    """
    GET /api/yieldsync/trends?crop=All&market=All&max_points=400&recent=50
    Returns:
      - stats: total_records, days_of_data, first_date, last_date
      - series: [{label, dates[], prices[]}, ...]
      - recent: [{date, market, item, price}, ...]
    """
    crop = _q("crop", "All")
    market = _q("market", "All")
    max_points = _q("max_points", 400)
    recent_n = _q("recent", 50)

    try:
        max_points = int(max_points)
    except Exception:
        max_points = 400

    try:
        recent_n = int(recent_n)
    except Exception:
        recent_n = 50

    api = get_api()
    df = api._load_data()  # uses existing cached loader in api.py

    # Ensure Date datetime (api already does, but safe)
    try:
        import pandas as pd
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass

    if df is None or len(df) == 0:
        return jsonify({"error": "No data available"}), 400

    # Stats on FULL dataset (not filtered)
    dmin = df["Date"].min()
    dmax = df["Date"].max()
    days_of_data = int((dmax.date() - dmin.date()).days) + 1

    stats = {
        "total_records": int(len(df)),
        "days_of_data": days_of_data,
        "first_date": str(dmin.date()),
        "last_date": str(dmax.date()),
    }

    # Apply filters for charts/recent
    f = df
    if crop and crop != "All":
        f = f[f["item"] == crop]
    if market and market != "All":
        f = f[f["market"] == market]

    # Build time-series
    series = []
    crops_list = api.crops if crop == "All" else [crop]

    for c in crops_list:
        sub = f if crop != "All" else f[f["item"] == c]
        if sub is None or len(sub) == 0:
            continue

        # Daily average
        daily = sub.groupby("Date")["price"].mean().reset_index()
        daily = daily.sort_values("Date")

        # Downsample for mobile chart performance
        n = len(daily)
        if max_points and n > max_points:
            import math
            step = int(math.ceil(n / max_points))
            daily = daily.iloc[::step, :]

        series.append({
            "label": c,
            "dates": daily["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "prices": daily["price"].round(2).tolist(),
            "count": int(len(daily)),
        })

    # Recent entries (filtered)
    recent_df = f.sort_values("Date", ascending=False).head(recent_n)
    recent_rows = []
    for _, r in recent_df.iterrows():
        recent_rows.append({
            "date": str(r["Date"]),
            "market": str(r.get("market", "")),
            "item": str(r.get("item", "")),
            "price": float(r.get("price", 0)),
        })

    return jsonify({
        "crop": crop,
        "market": market,
        "stats": stats,
        "series": series,
        "recent": recent_rows,
    }), 200
