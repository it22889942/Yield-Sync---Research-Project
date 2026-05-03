#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------------
# CONFIG (CHANGE ONLY THIS PART)
# ----------------------------
JSON_PATH = "crop_recommendations.json"   # your json file
COLLECTION = "croprecomdation"           # Firestore collection name
SERVICE_KEY = "serviceAccountKey.json"   # Firebase service account json path
BATCH_SIZE = 450                         # safe batch size

def normalize_value(v):
    """
    Firestore-safe basic normalization.
    """
    if v is None:
        return None
    # Convert any non-JSON types (if exist) to string
    if isinstance(v, (str, int, float, bool, list, dict)):
        return v
    return str(v)

def normalize_dict(obj):
    """
    Recursively normalize nested maps/lists for Firestore.
    """
    if isinstance(obj, dict):
        return {str(k): normalize_dict(normalize_value(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_dict(normalize_value(x)) for x in obj]
    return normalize_value(obj)

def main():
    # Init Firebase
    cred = credentials.Certificate(SERVICE_KEY)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # Load JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise Exception("❌ JSON root must be an object/map like { 'RICE': {...}, ... }")

    print("✅ JSON loaded.")
    print(f"📌 Documents found: {list(data.keys())}")

    batch = db.batch()
    count = 0

    for doc_id, doc_data in data.items():
        doc_id = str(doc_id).strip()
        if not doc_id:
            continue

        # Ensure Firestore-safe nested data
        doc_data = normalize_dict(doc_data)

        # Add/keep docId inside document (optional but useful)
        doc_data["docId"] = doc_id

        doc_ref = db.collection(COLLECTION).document(doc_id)
        batch.set(doc_ref, doc_data, merge=True)

        count += 1

        if count % BATCH_SIZE == 0:
            batch.commit()
            batch = db.batch()
            print(f"✅ Uploaded {count} documents...")

    if count % BATCH_SIZE != 0:
        batch.commit()

    print(f"🎉 DONE! Total uploaded: {count} into '{COLLECTION}' collection.")

if __name__ == "__main__":
    main()
