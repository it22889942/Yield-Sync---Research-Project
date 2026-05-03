# equipment_routes.py — Firestore as source of truth
import os
import re
import time
import logging
import statistics
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)
equipment_bp = Blueprint("equipment", __name__, url_prefix="/api/equipment")

# In-memory cache to reduce Firestore reads (TTL seconds)
_EQUIPMENT_CACHE_TTL = 3600  # 1 hour
_equipment_cache = {"rows": None, "ts": 0}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EQUIPMENT_COLLECTION = "equipments"
SERVICE_KEY_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

_db = None

def _get_db():
    global _db
    if _db is not None:
        return _db
    key_path = SERVICE_KEY_PATH
    if not os.path.exists(key_path):
        key_path = os.path.join(os.getcwd(), "Backend", "serviceAccountKey.json")
    if not os.path.exists(key_path):
        key_path = os.path.join(os.getcwd(), "serviceAccountKey.json")
    if not os.path.exists(key_path):
        raise FileNotFoundError(
            f"Firebase key not found. Tried: {SERVICE_KEY_PATH}. Put serviceAccountKey.json in Backend folder."
        )
    import firebase_admin
    from firebase_admin import credentials, firestore
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(key_path))
    _db = firestore.client()
    return _db

def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if hasattr(x, "__float__") and x != x:  # NaN
            return ""
    except Exception:
        pass
    return str(x).strip()

def _safe_num(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if hasattr(x, "__float__") and x != x:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default

def _format_phone(v) -> str:
    if v is None:
        return "—"
    try:
        if hasattr(v, "__float__") and v != v:
            return "—"
    except Exception:
        pass
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.0+)?", s):
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    digits = re.sub(r"\D", "", s)
    if not digits:
        return "—"
    if len(digits) == 9:
        digits = "0" + digits
    return digits

def _main_location(loc: str) -> str:
    loc = _safe_str(loc)
    if not loc:
        return ""
    return loc.split("-")[0].strip()

def _doc_to_row(doc_id, data):
    """Normalize Firestore doc to a dict with expected keys."""
    def g(*keys, default=""):
        for k in keys:
            if k in data and data[k] is not None:
                v = data[k]
                try:
                    if v == "" or (isinstance(v, float) and (v != v)):
                        return default
                    if hasattr(v, "isoformat"):  # Firestore Timestamp etc.
                        return str(v)
                    return str(v).strip() if not isinstance(v, (int, float)) else str(v)
                except Exception:
                    return default
        return default

    def gn(*keys, default=0.0):
        for k in keys:
            if k in data and data[k] is not None:
                try:
                    return float(data[k])
                except (TypeError, ValueError):
                    pass
        return default

    nearest = g("Nearest_Major_District", "Nearest_Major_District", default="")
    main_dist = _main_location(nearest) if nearest else ""

    return {
        "Equipment_ID": g("Equipment_ID", "equipmentId", default=doc_id),
        "Equipment_Type": g("Equipment_Type", "Equipment_Type"),
        "For_Crop": g("For_Crop", "For_Crop"),
        "Equipment_Owner_Name": g("Equipment_Owner_Name", "Equipment_Owner_Name"),
        "Nearest_Major_District": nearest,
        "Main_District": main_dist,
        "Season": g("Season", "Season"),
        "Condition": g("Condition", "Condition"),
        "Owner_Type": g("Owner_Type", "Owner_Type"),
        "Last_Booked_By": g("Last_Booked_By", "Last_Booked_By"),
        "Registration_No": g("Registration_No", "Registration_No"),
        "Available_Day": g("Available_Day", "Available_Day"),
        "Available_Time": g("Available_Time", "Available_Time"),
        "Owner_Contact_No": data.get("Owner_Contact_No"),
        "Insurance_Valid_Till": data.get("Insurance_Valid_Till"),
        "Hourly_Rate_LKR": gn("Hourly_Rate_LKR", "Hourly_Rate_LKR"),
        "Daily_Rate_LKR": gn("Daily_Rate_LKR", "Daily_Rate_LKR"),
        "Rating": gn("Rating", "Rating"),
        "Past_Bookings": gn("Past_Bookings", "Past_Bookings"),
        "Success_Rate_pct": gn("Success_Rate_pct", "Success_Rate_pct"),
        "Distance_km_from_Main_District": gn("Distance_km_from_Main_District", "Distance_km_from_Main_District"),
        "Manufacture_Year": int(gn("Manufacture_Year", "Manufacture_Year")),
        "Power_Capacity_HP": gn("Power_Capacity_HP", "Power_Capacity_HP"),
        "Maintenance_Cost_LKR_Per_Month": gn("Maintenance_Cost_LKR_Per_Month", "Maintenance_Cost_LKR_Per_Month"),
        "Avg_Usage_Hours_Per_Month": gn("Avg_Usage_Hours_Per_Month", "Avg_Usage_Hours_Per_Month"),
        "Downtime_Days_Per_Month": gn("Downtime_Days_Per_Month", "Downtime_Days_Per_Month"),
        "Owner_Experience_Years": int(gn("Owner_Experience_Years", "Owner_Experience_Years")),
        "Moderation_Status": g("moderation_status", "Moderation_Status", default="approved"),
    }

def _fetch_all_equipment_rows_from_firestore():
    """Actually stream all equipment docs from Firestore (no cache)."""
    db = _get_db()
    rows = []
    for doc in db.collection(EQUIPMENT_COLLECTION).stream():
        doc_id = doc.id
        data = doc.to_dict() or {}
        row = _doc_to_row(doc_id, data)
        if not row["Equipment_ID"]:
            row["Equipment_ID"] = doc_id
        rows.append(row)
    return rows


def _get_all_equipment_rows():
    """Return all equipment rows, using a short TTL cache to reduce Firestore reads."""
    global _equipment_cache
    now = time.time()
    if _equipment_cache["rows"] is not None and (now - _equipment_cache["ts"]) < _EQUIPMENT_CACHE_TTL:
        return _equipment_cache["rows"]
    rows = _fetch_all_equipment_rows_from_firestore()
    _equipment_cache = {"rows": rows, "ts": now}
    return rows

def equipment_row_public_visible(row: dict) -> bool:
    """Approved listings only; pending/rejected hidden from public search."""
    s = str(row.get("Moderation_Status", "approved")).strip().lower()
    return s not in ("pending", "rejected")


def _invalidate_equipment_cache():
    global _equipment_cache
    _equipment_cache = {"rows": None, "ts": 0}


def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _tokenize(q: str):
    q = _norm(q)
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    return [t for t in q.split() if t]


def _positive_rates(values):
    out = []
    for v in values:
        try:
            fv = float(v)
            if fv > 0:
                out.append(fv)
        except (TypeError, ValueError):
            pass
    return out


def _equipment_peer_rates(rows, district: str, equipment_type: str, rate_key: str):
    dist = _norm(district)
    typ = _norm(equipment_type)

    exact = _positive_rates(
        r.get(rate_key)
        for r in rows
        if _norm(str(r.get("Main_District", ""))) == dist
        and _norm(str(r.get("Equipment_Type", ""))) == typ
    )
    if len(exact) >= 5:
        return exact

    by_dist = _positive_rates(
        r.get(rate_key)
        for r in rows
        if _norm(str(r.get("Main_District", ""))) == dist
    )
    if len(by_dist) >= 5:
        return by_dist

    by_type = _positive_rates(
        r.get(rate_key)
        for r in rows
        if _norm(str(r.get("Equipment_Type", ""))) == typ
    )
    if len(by_type) >= 5:
        return by_type

    return _positive_rates(r.get(rate_key) for r in rows)


def _rate_outlier(value: float, peer_rates: list[float], label: str):
    if value <= 0 or len(peer_rates) < 3:
        return {
            f"{label}_outlier": False,
            f"{label}_median": None,
            f"{label}_low_threshold": None,
            f"{label}_high_threshold": None,
            f"{label}_outlier_reason": "",
        }

    median_rate = float(statistics.median(peer_rates))
    low = round(median_rate * 0.60, 2)
    high = round(median_rate * 1.80, 2)
    is_outlier = value < low or value > high

    reason = ""
    if is_outlier:
        direction = "below" if value < low else "above"
        reason = (
            f"{label.capitalize()} rate is {direction} expected range for similar posts "
            f"(median={median_rate:.2f}, range={low:.2f}-{high:.2f})."
        )

    return {
        f"{label}_outlier": is_outlier,
        f"{label}_median": round(median_rate, 2),
        f"{label}_low_threshold": low,
        f"{label}_high_threshold": high,
        f"{label}_outlier_reason": reason,
    }

@equipment_bp.get("/health")
def health():
    try:
        _get_db()
        return jsonify({"status": "ok", "service": "Equipment Search"})
    except Exception as e:
        logger.exception("Equipment health check failed")
        return jsonify({"error": str(e), "service": "Equipment Search"}), 503

@equipment_bp.get("/locations")
def locations():
    try:
        rows = _get_all_equipment_rows()
        locs = sorted({_safe_str(r.get("Main_District", "")) for r in rows if _safe_str(r.get("Main_District", ""))})
        return jsonify({"locations": locs})
    except FileNotFoundError as e:
        logger.warning("Equipment locations: %s", e)
        return jsonify({"error": str(e), "locations": []}), 503
    except Exception as e:
        logger.exception("Equipment locations failed")
        return jsonify({"error": str(e)}), 500

@equipment_bp.get("/types")
def types():
    try:
        rows = _get_all_equipment_rows()
        out = sorted({_safe_str(r.get("Equipment_Type", "")) for r in rows if _safe_str(r.get("Equipment_Type", ""))})
        return jsonify({"types": out})
    except FileNotFoundError as e:
        logger.warning("Equipment types: %s", e)
        return jsonify({"error": str(e), "types": []}), 503
    except Exception as e:
        logger.exception("Equipment types failed")
        return jsonify({"error": str(e)}), 500

@equipment_bp.get("/item/<equip_id>")
def item(equip_id):
    eid = _safe_str(equip_id)
    db = _get_db()
    doc_ref = db.collection(EQUIPMENT_COLLECTION).document(eid)
    doc = doc_ref.get()
    if not doc or not doc.exists:
        return jsonify({"error": "not_found"}), 404
    r = _doc_to_row(doc.id, doc.to_dict() or {})
    if not r["Equipment_ID"]:
        r["Equipment_ID"] = doc.id

    ins = r.get("Insurance_Valid_Till")
    if ins is not None:
        if hasattr(ins, "date") and callable(getattr(ins, "date")):
            try:
                ins = ins.date().isoformat()
            except Exception:
                ins = str(ins)
        elif hasattr(ins, "isoformat") and callable(getattr(ins, "isoformat")):
            try:
                ins = ins.isoformat()
            except Exception:
                ins = str(ins)
        else:
            ins = str(ins)
    else:
        ins = ""

    return jsonify({
        "id": _safe_str(r.get("Equipment_ID")),
        "equipment_type": _safe_str(r.get("Equipment_Type")),
        "for_crop": _safe_str(r.get("For_Crop")),
        "hourly_rate": _safe_num(r.get("Hourly_Rate_LKR")),
        "daily_rate": _safe_num(r.get("Daily_Rate_LKR")),
        "rating": _safe_num(r.get("Rating")),
        "past_bookings": int(_safe_num(r.get("Past_Bookings"), 0)),
        "success_rate_pct": _safe_num(r.get("Success_Rate_pct")),
        "distance_km": _safe_num(r.get("Distance_km_from_Main_District")),
        "nearest_major_district": _safe_str(r.get("Nearest_Major_District")),
        "main_district": _safe_str(r.get("Main_District")),
        "season": _safe_str(r.get("Season")),
        "owner_name": _safe_str(r.get("Equipment_Owner_Name")) or "—",
        "owner_contact": _format_phone(r.get("Owner_Contact_No")),
        "condition": _safe_str(r.get("Condition")),
        "manufacture_year": int(_safe_num(r.get("Manufacture_Year"), 0)),
        "power_hp": _safe_num(r.get("Power_Capacity_HP")),
        "maintenance_cost_month": _safe_num(r.get("Maintenance_Cost_LKR_Per_Month")),
        "avg_usage_hours_month": _safe_num(r.get("Avg_Usage_Hours_Per_Month")),
        "downtime_days_month": _safe_num(r.get("Downtime_Days_Per_Month")),
        "owner_type": _safe_str(r.get("Owner_Type")),
        "owner_experience_years": int(_safe_num(r.get("Owner_Experience_Years"), 0)),
        "last_booked_by": _safe_str(r.get("Last_Booked_By")),
        "insurance_valid_till": ins,
        "registration_no": _safe_str(r.get("Registration_No")),
        "available_day": _safe_str(r.get("Available_Day")),
        "available_time": _safe_str(r.get("Available_Time")),
    })

@equipment_bp.post("/search")
def search():
    """
    Body:
    {
      "query": "tractor",
      "location": "Kurunegala",
      "type": "Harvester",
      "top_k": 20
    }
    """
    try:
        rows = _get_all_equipment_rows()
        rows = [r for r in rows if equipment_row_public_visible(r)]
    except FileNotFoundError as e:
        logger.warning("Equipment search: %s", e)
        return jsonify({"error": str(e), "count": 0, "items": []}), 503
    except Exception as e:
        logger.exception("Equipment search failed")
        return jsonify({"error": str(e)}), 500

    data = request.get_json(silent=True) or {}
    query = _safe_str(data.get("query"))
    location = _safe_str(data.get("location"))
    typ = _safe_str(data.get("type"))
    top_k = int(data.get("top_k") or 20)

    out = [r for r in rows]

    if location:
        out = [r for r in out if _norm(str(r.get("Main_District", ""))) == location.lower()]

    if typ:
        out = [r for r in out if typ.lower() in _norm(str(r.get("Equipment_Type", "")))]

    toks = _tokenize(query)
    if toks:
        def any_match(r):
            for col in ["Equipment_Type", "For_Crop", "Nearest_Major_District", "Condition", "Owner_Type"]:
                val = _norm(str(r.get(col, "")))
                if any(t in val for t in toks):
                    return True
            return False
        out = [r for r in out if any_match(r)]

    if not out:
        return jsonify({"count": 0, "items": []})

    def score_row(r):
        rating = float(r.get("Rating", 0) or 0) / 5.0
        success = float(r.get("Success_Rate_pct", 0) or 0) / 100.0
        bookings = float(r.get("Past_Bookings", 0) or 0)
        bookings_boost = min(1.0, bookings / 200.0)
        return (0.60 * rating) + (0.25 * success) + (0.15 * bookings_boost)

    for r in out:
        r["_score"] = score_row(r)
    out.sort(key=lambda r: (-r["_score"], -float(r.get("Rating", 0)), -int(r.get("Past_Bookings", 0))))
    out = out[:top_k]

    items = []
    for r in out:
        items.append({
            "id": _safe_str(r.get("Equipment_ID")),
            "equipment_type": _safe_str(r.get("Equipment_Type")),
            "for_crop": _safe_str(r.get("For_Crop")),
            "location": _safe_str(r.get("Main_District")),
            "nearest_major_district": _safe_str(r.get("Nearest_Major_District")),
            "hourly_rate": float(r.get("Hourly_Rate_LKR", 0) or 0),
            "daily_rate": float(r.get("Daily_Rate_LKR", 0) or 0),
            "rating": float(r.get("Rating", 0) or 0),
            "past_bookings": int(r.get("Past_Bookings", 0) or 0),
            "owner_name": _safe_str(r.get("Equipment_Owner_Name")) or "—",
            "owner_contact": _format_phone(r.get("Owner_Contact_No")),
            "available_day": _safe_str(r.get("Available_Day")),
            "available_time": _safe_str(r.get("Available_Time")),
            "condition": _safe_str(r.get("Condition")),
            "score": float(r.get("_score", 0)),
        })

    return jsonify({"count": len(items), "items": items})


@equipment_bp.get("/pending")
def list_pending():
    """Equipment ads awaiting admin approval."""
    try:
        rows = _fetch_all_equipment_rows_from_firestore()
    except FileNotFoundError as e:
        logger.warning("Equipment pending: %s", e)
        return jsonify({"error": str(e), "items": []}), 503
    except Exception as e:
        logger.exception("Equipment pending failed")
        return jsonify({"error": str(e)}), 500

    pending = [
        r for r in rows
        if str(r.get("Moderation_Status", "")).strip().lower() == "pending"
    ]

    items = []
    for r in pending:
        hourly_rate = float(r.get("Hourly_Rate_LKR", 0) or 0)
        daily_rate = float(r.get("Daily_Rate_LKR", 0) or 0)
        district = _safe_str(r.get("Main_District"))
        equipment_type = _safe_str(r.get("Equipment_Type"))

        hourly_fraud = _rate_outlier(
            value=hourly_rate,
            peer_rates=_equipment_peer_rates(
                rows=rows,
                district=district,
                equipment_type=equipment_type,
                rate_key="Hourly_Rate_LKR",
            ),
            label="hourly_price",
        )
        daily_fraud = _rate_outlier(
            value=daily_rate,
            peer_rates=_equipment_peer_rates(
                rows=rows,
                district=district,
                equipment_type=equipment_type,
                rate_key="Daily_Rate_LKR",
            ),
            label="daily_price",
        )

        items.append({
            "id": _safe_str(r.get("Equipment_ID")),
            "equipment_type": equipment_type,
            "for_crop": _safe_str(r.get("For_Crop")),
            "location": district,
            "nearest_major_district": _safe_str(r.get("Nearest_Major_District")),
            "hourly_rate": hourly_rate,
            "daily_rate": daily_rate,
            "rating": float(r.get("Rating", 0) or 0),
            "past_bookings": int(r.get("Past_Bookings", 0) or 0),
            "owner_name": _safe_str(r.get("Equipment_Owner_Name")) or "—",
            "available_day": _safe_str(r.get("Available_Day")),
            "available_time": _safe_str(r.get("Available_Time")),
            "condition": _safe_str(r.get("Condition")),
            "moderation_status": "pending",
            **hourly_fraud,
            **daily_fraud,
        })

    return jsonify({"count": len(items), "items": items})


@equipment_bp.post("/<doc_id>/moderate")
def moderate(doc_id: str):
    """Set moderation_status to approved or rejected (doc id = Firestore document id)."""
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "").strip().lower()
    if action not in ("approve", "reject"):
        return jsonify({"error": 'action must be "approve" or "reject"'}), 400

    try:
        db = _get_db()
        ref = db.collection(EQUIPMENT_COLLECTION).document(doc_id)
        snap = ref.get()
        if not snap.exists:
            return jsonify({"error": "not_found"}), 404
        new_status = "approved" if action == "approve" else "rejected"
        ref.update({"moderation_status": new_status})
        _invalidate_equipment_cache()
        return jsonify({"ok": True, "id": doc_id, "moderation_status": new_status})
    except Exception as e:
        logger.exception("Equipment moderate failed")
        return jsonify({"error": str(e)}), 500
