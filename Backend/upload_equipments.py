#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------------
# CONFIG (CHANGE ONLY THIS PART)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILENAME = "Equipment_Final_fixed.xlsx"
EXCEL_CANDIDATES = [
    os.path.join(BASE_DIR, "data", EXCEL_FILENAME),
    os.path.join(BASE_DIR, EXCEL_FILENAME),
    os.path.join(os.path.dirname(BASE_DIR), EXCEL_FILENAME),
    os.path.join(os.path.dirname(BASE_DIR), "data", EXCEL_FILENAME),
]
SHEET_NAME = None  # None = first sheet
COL_ID = "Equipment_ID"  # <-- Excel ID column (change if different)
COLLECTION = "equipments"  # Firestore collection name
SERVICE_KEY = os.path.join(BASE_DIR, "serviceAccountKey.json")

def clean_field_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def normalize_value(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return str(v)
    return v

def read_excel_safely(path: str, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    if isinstance(df, dict):
        first_key = list(df.keys())[0]
        print(f"ℹ️ Multiple sheets found. Using first sheet: {first_key}")
        df = df[first_key]
    if not isinstance(df, pd.DataFrame):
        raise Exception(f"❌ Excel read failed. Got: {type(df)}")
    return df

def main():
    # Init Firebase
    if not os.path.exists(SERVICE_KEY):
        raise FileNotFoundError(
            f"Firebase key not found: {SERVICE_KEY}\n"
            "Put serviceAccountKey.json in the Backend folder."
        )
    cred = credentials.Certificate(SERVICE_KEY)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    # Read Excel (first path that exists)
    excel_path = None
    for p in EXCEL_CANDIDATES:
        if os.path.exists(p):
            excel_path = p
            break
    if not excel_path:
        raise FileNotFoundError(
            f"Equipment Excel file not found. Tried:\n  " + "\n  ".join(EXCEL_CANDIDATES)
            + "\n\nPut Equipment_Final_fixed.xlsx in Backend/ or Backend/data/"
        )
    print(f"📂 Using Excel: {excel_path}")
    df = read_excel_safely(excel_path, SHEET_NAME)
    df.columns = [c.strip() for c in df.columns]

    print("✅ Excel loaded.")
    print("📌 Columns:", df.columns.tolist())

    # Validate ID column
    if COL_ID not in df.columns:
        raise Exception(
            f"❌ ID column '{COL_ID}' not found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"➡️ Fix: set COL_ID to correct column name."
        )

    # Clean column names (safe keys)
    cleaned_cols = {c: clean_field_name(c) for c in df.columns}
    df = df.rename(columns=cleaned_cols)

    id_col_clean = clean_field_name(COL_ID)
    if id_col_clean not in df.columns:
        raise Exception(
            f"❌ Cleaned ID column '{id_col_clean}' not found.\n"
            f"Available: {list(df.columns)}"
        )

    # Upload in batches
    batch = db.batch()
    count = 0

    for _, row in df.iterrows():
        doc_id = str(row.get(id_col_clean, "")).strip()
        if not doc_id or doc_id.lower() == "nan":
            continue

        data = {k: normalize_value(v) for k, v in row.to_dict().items()}

        # Keep both standardized + original-ish id field
        data["equipmentId"] = doc_id
        data[id_col_clean] = doc_id

        doc_ref = db.collection(COLLECTION).document(doc_id)
        batch.set(doc_ref, data, merge=True)

        count += 1
        if count % 450 == 0:
            batch.commit()
            batch = db.batch()
            print(f"✅ Uploaded {count} records...")

    if count % 450 != 0:
        batch.commit()

    print(f"🎉 DONE! Total uploaded: {count} into '{COLLECTION}' collection.")

if __name__ == "__main__":
    main()
