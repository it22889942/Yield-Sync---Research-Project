# labour_routes.py — Firestore as source of truth
import os
import re
import time
import logging
import statistics
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)
labour_bp = Blueprint("labour", __name__, url_prefix="/api/labour")

# In-memory cache to reduce Firestore reads (TTL seconds)
_LABOUR_CACHE_TTL = 3600  # 1 hour
_labour_cache = {"rows": None, "ts": 0}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABOUR_COLLECTION = "labours"
SERVICE_KEY_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

# Optional: fallback Excel paths if Firestore unavailable
CANDIDATES = [
    os.path.join(BASE_DIR, "data", "Labour_Expanded_clean_generated.xlsx"),
    os.path.join(BASE_DIR, "Labour_Expanded_clean_generated.xlsx"),
    os.path.join(os.getcwd(), "data", "Labour_Expanded_clean_generated.xlsx"),
    os.path.join(os.getcwd(), "Labour_Expanded_clean_generated.xlsx"),
]

_db = None

def _get_db():
    global _db
    if _db is not None:
        return _db
    key_path = SERVICE_KEY_PATH
    if not os.path.exists(key_path):
        # Try from cwd (e.g. if server is run from project root)
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

def _doc_to_row(doc_id, data):
    """Normalize Firestore doc to a dict with expected keys (Excel-style names)."""
    def g(*keys, default=""):
        for k in keys:
            if k in data and data[k] is not None:
                v = data[k]
                try:
                    if v == "" or (isinstance(v, float) and (v != v)):  # NaN
                        return default
                    # Firestore Timestamp or other types -> string
                    if hasattr(v, "isoformat"):
                        return str(v)
                    return str(v).strip() if not isinstance(v, (int, float)) else str(v)
                except Exception:
                    return default
        return default

    def gn(*keys, default=0):
        for k in keys:
            if k in data and data[k] is not None:
                try:
                    v = data[k]
                    if isinstance(v, (int, float)) and v == v:  # no NaN
                        return float(v)
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return default

    return {
        "Labour_ID": g("Labour_ID", "labourId", default=doc_id),
        "Name": g("Name", "name"),
        "Labour_Type": g("Labour_Type", "Labour_Type"),
        "Skill_Level": g("Skill_Level", "Skill_Level"),
        "Location": g("Location", "Location"),
        "Season": g("Season", "Season"),
        "Crop_Type": g("Crop_Type", "Crop_Type"),
        "Available_Day": g("Available_Day", "Available_Day"),
        "Available_Time": g("Available_Time", "Available_Time"),
        "Hourly_Rate": gn("Hourly_Rate", "Hourly_Rate"),
        "Rating": gn("Rating", "Rating"),
        "Experience_Years": int(gn("Experience_Years", "Experience_Years")),
        "Jobs_Completed": int(gn("Jobs_Completed", "Jobs_Completed")),
        "Moderation_Status": g("moderation_status", "Moderation_Status", default="approved"),
    }

def _fetch_all_labour_rows_from_firestore():
    """Actually stream all labour docs from Firestore (no cache)."""
    db = _get_db()
    coll = db.collection(LABOUR_COLLECTION)
    rows = []
    for doc in coll.stream():
        doc_id = doc.id
        data = doc.to_dict() or {}
        row = _doc_to_row(doc_id, data)
        if not row["Labour_ID"]:
            row["Labour_ID"] = doc_id
        rows.append(row)
    return rows


def _get_all_labour_rows():
    """Return all labour rows, using a short TTL cache to reduce Firestore reads."""
    global _labour_cache
    now = time.time()
    if _labour_cache["rows"] is not None and (now - _labour_cache["ts"]) < _LABOUR_CACHE_TTL:
        return _labour_cache["rows"]
    rows = _fetch_all_labour_rows_from_firestore()
    _labour_cache = {"rows": rows, "ts": now}
    return rows

def labour_row_public_visible(row: dict) -> bool:
    """Approved listings only; pending/rejected hidden from public search."""
    s = str(row.get("Moderation_Status", "approved")).strip().lower()
    return s not in ("pending", "rejected")


def _invalidate_labour_cache():
    global _labour_cache
    _labour_cache = {"rows": None, "ts": 0}


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


def _labour_peer_rates(rows, location: str, labour_type: str):
    loc = _norm(location)
    typ = _norm(labour_type)

    exact = _positive_rates(
        r.get("Hourly_Rate")
        for r in rows
        if _norm(str(r.get("Location", ""))) == loc
        and _norm(str(r.get("Labour_Type", ""))) == typ
    )
    if len(exact) >= 5:
        return exact

    by_loc = _positive_rates(
        r.get("Hourly_Rate")
        for r in rows
        if _norm(str(r.get("Location", ""))) == loc
    )
    if len(by_loc) >= 5:
        return by_loc

    by_type = _positive_rates(
        r.get("Hourly_Rate")
        for r in rows
        if _norm(str(r.get("Labour_Type", ""))) == typ
    )
    if len(by_type) >= 5:
        return by_type

    return _positive_rates(r.get("Hourly_Rate") for r in rows)


def _price_outlier(hourly_rate: float, peer_rates: list[float]):
    """Simple robust check against peer median.
    Outlier if rate is below 60% or above 180% of median.
    """
    if hourly_rate <= 0 or len(peer_rates) < 3:
        return {
            "price_outlier": False,
            "price_median": None,
            "price_low_threshold": None,
            "price_high_threshold": None,
            "price_outlier_reason": "",
        }

    median_rate = float(statistics.median(peer_rates))
    low = round(median_rate * 0.60, 2)
    high = round(median_rate * 1.80, 2)
    is_outlier = hourly_rate < low or hourly_rate > high

    reason = ""
    if is_outlier:
        direction = "below" if hourly_rate < low else "above"
        reason = (
            f"Hourly rate is {direction} expected range for similar posts "
            f"(median={median_rate:.2f}, range={low:.2f}-{high:.2f})."
        )

    return {
        "price_outlier": is_outlier,
        "price_median": round(median_rate, 2),
        "price_low_threshold": low,
        "price_high_threshold": high,
        "price_outlier_reason": reason,
    }

def _keyword_score(row, toks):
    if not toks:
        return 0.0
    text = " ".join([
        str(row.get("Name", "")),
        str(row.get("Labour_Type", "")),
        str(row.get("Skill_Level", "")),
        str(row.get("Crop_Type", "")),
        str(row.get("Season", "")),
        str(row.get("Location", "")),
    ]).lower()
    hit = sum(1 for t in toks if t in text)
    return hit / max(1, len(toks))

@labour_bp.get("/health")
def health():
    try:
        _get_db()
        return jsonify({"status": "ok", "service": "Labour Search"})
    except Exception as e:
        logger.exception("Labour health check failed")
        return jsonify({"error": str(e), "service": "Labour Search"}), 503

@labour_bp.get("/locations")
def locations():
    try:
        rows = _get_all_labour_rows()
        locs = sorted({str(r["Location"]).strip() for r in rows if str(r.get("Location", "")).strip()})
        return jsonify({"locations": locs})
    except FileNotFoundError as e:
        logger.warning("Labour locations: %s", e)
        return jsonify({"error": str(e), "locations": []}), 503
    except Exception as e:
        logger.exception("Labour locations failed")
        return jsonify({"error": str(e)}), 500

@labour_bp.get("/skills")
def skills():
    try:
        rows = _get_all_labour_rows()
        skills_list = sorted({str(r["Labour_Type"]).strip() for r in rows if str(r.get("Labour_Type", "")).strip()})
        return jsonify({"skills": skills_list})
    except FileNotFoundError as e:
        logger.warning("Labour skills: %s", e)
        return jsonify({"error": str(e), "skills": []}), 503
    except Exception as e:
        logger.exception("Labour skills failed")
        return jsonify({"error": str(e)}), 500

@labour_bp.post("/search")
def search():
    """
    Body:
    {
      "query": "paddy harvesting",
      "location": "Kurunegala",
      "skill": "Rice Harvesting",   // optional
      "top_k": 20
    }
    """
    try:
        rows = _get_all_labour_rows()
        rows = [r for r in rows if labour_row_public_visible(r)]
    except FileNotFoundError as e:
        logger.warning("Labour search: %s", e)
        return jsonify({
            "error": str(e),
            "query": "",
            "location": "",
            "skill": None,
            "count": 0,
            "items": []
        }), 503
    except Exception as e:
        logger.exception("Labour search failed")
        return jsonify({"error": str(e)}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    location = (data.get("location") or "").strip()
    skill = (data.get("skill") or "").strip()
    top_k = int(data.get("top_k") or 20)

    out = [r for r in rows]

    if location:
        out = [r for r in out if _norm(str(r.get("Location", ""))) == location.lower()]

    if skill:
        s = skill.lower()
        out = [
            r for r in out
            if s in _norm(str(r.get("Labour_Type", "")))
            or s in _norm(str(r.get("Skill_Level", "")))
            or s in _norm(str(r.get("Crop_Type", "")))
        ]

    toks = _tokenize(query)
    if toks:
        def any_match(r):
            for col in ["Name", "Labour_Type", "Skill_Level", "Crop_Type", "Season", "Location"]:
                val = _norm(str(r.get(col, "")))
                if any(t in val for t in toks):
                    return True
            return False
        out = [r for r in out if any_match(r)]

    if not out:
        return jsonify({
            "query": query,
            "location": location,
            "skill": skill or None,
            "count": 0,
            "items": []
        })

    def score_row(r):
        k = _keyword_score(r, toks)
        rating = float(r.get("Rating", 0) or 0) / 5.0
        jobs = float(r.get("Jobs_Completed", 0) or 0)
        jobs_boost = min(1.0, jobs / 200.0)
        return (0.55 * k) + (0.35 * rating) + (0.10 * jobs_boost)

    for r in out:
        r["_score"] = score_row(r)
    out.sort(key=lambda r: (-r["_score"], -float(r.get("Rating", 0)), -int(r.get("Jobs_Completed", 0))))
    out = out[:top_k]

    items = []
    for r in out:
        items.append({
            "id": str(r.get("Labour_ID", "")),
            "name": str(r.get("Name", "")),
            "location": str(r.get("Location", "")),
            "labour_type": str(r.get("Labour_Type", "")),
            "skill_level": str(r.get("Skill_Level", "")),
            "hourly_rate": float(r.get("Hourly_Rate", 0) or 0),
            "rating": float(r.get("Rating", 0) or 0),
            "jobs_completed": int(r.get("Jobs_Completed", 0) or 0),
            "experience_years": int(r.get("Experience_Years", 0) or 0),
            "season": str(r.get("Season", "")),
            "crop_type": str(r.get("Crop_Type", "")),
            "available_day": str(r.get("Available_Day", "")),
            "available_time": str(r.get("Available_Time", "")),
            "score": float(r.get("_score", 0)),
        })

    return jsonify({
        "query": query,
        "location": location,
        "skill": skill or None,
        "count": len(items),
        "items": items
    })


@labour_bp.get("/pending")
def list_pending():
    """Labour ads awaiting admin approval."""
    try:
        rows = _fetch_all_labour_rows_from_firestore()
    except FileNotFoundError as e:
        logger.warning("Labour pending: %s", e)
        return jsonify({"error": str(e), "items": []}), 503
    except Exception as e:
        logger.exception("Labour pending failed")
        return jsonify({"error": str(e)}), 500

    pending = [
        r for r in rows
        if str(r.get("Moderation_Status", "")).strip().lower() == "pending"
    ]

    items = []
    for r in pending:
        hourly_rate = float(r.get("Hourly_Rate", 0) or 0)
        peers = _labour_peer_rates(
            rows=rows,
            location=str(r.get("Location", "")),
            labour_type=str(r.get("Labour_Type", "")),
        )
        fraud = _price_outlier(hourly_rate, peers)
        items.append({
            "id": str(r.get("Labour_ID", "")),
            "name": str(r.get("Name", "")),
            "location": str(r.get("Location", "")),
            "labour_type": str(r.get("Labour_Type", "")),
            "skill_level": str(r.get("Skill_Level", "")),
            "hourly_rate": hourly_rate,
            "rating": float(r.get("Rating", 0) or 0),
            "jobs_completed": int(r.get("Jobs_Completed", 0) or 0),
            "experience_years": int(r.get("Experience_Years", 0) or 0),
            "season": str(r.get("Season", "")),
            "crop_type": str(r.get("Crop_Type", "")),
            "available_day": str(r.get("Available_Day", "")),
            "available_time": str(r.get("Available_Time", "")),
            "moderation_status": "pending",
            **fraud,
        })

    return jsonify({"count": len(items), "items": items})


@labour_bp.post("/<doc_id>/moderate")
def moderate(doc_id: str):
    """Set moderation_status to approved or rejected (doc id = Firestore document id)."""
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "").strip().lower()
    if action not in ("approve", "reject"):
        return jsonify({"error": 'action must be "approve" or "reject"'}), 400

    try:
        db = _get_db()
        ref = db.collection(LABOUR_COLLECTION).document(doc_id)
        snap = ref.get()
        if not snap.exists:
            return jsonify({"error": "not_found"}), 404
        new_status = "approved" if action == "approve" else "rejected"
        ref.update({"moderation_status": new_status})
        _invalidate_labour_cache()
        return jsonify({"ok": True, "id": doc_id, "moderation_status": new_status})
    except Exception as e:
        logger.exception("Labour moderate failed")
        return jsonify({"error": str(e)}), 500
