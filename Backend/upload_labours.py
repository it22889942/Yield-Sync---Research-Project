import os
import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILENAME = "Labour_Expanded_clean_generated.xlsx"
# Try these locations (script dir, data subfolder, project root)
EXCEL_CANDIDATES = [
    os.path.join(BASE_DIR, "data", EXCEL_FILENAME),
    os.path.join(BASE_DIR, EXCEL_FILENAME),
    os.path.join(os.path.dirname(BASE_DIR), EXCEL_FILENAME),
    os.path.join(os.path.dirname(BASE_DIR), "data", EXCEL_FILENAME),
]
SHEET_NAME = None                          # can be None (auto) or 0 or "Sheet1"
COL_ID = "Labour_ID"                       # Excel ID column name
COLLECTION = "labours"                     # Firestore collection name
SERVICE_KEY = os.path.join(BASE_DIR, "serviceAccountKey.json")

def clean_field_name(s: str) -> str:
    """Clean column names to safer Firestore keys (optional but good)."""
    s = str(s).strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def normalize_value(v):
    """Convert NaN to None and numpy types to python native."""
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return str(v)
    return v

def read_excel_safely(path: str, sheet_name):
    """
    If sheet_name=None, pandas may return dict of DataFrames (all sheets).
    This function always returns a single DataFrame (first sheet).
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    # If multiple sheets returned as dict -> pick first sheet
    if isinstance(df, dict):
        # df = {"Sheet1": DataFrame, "Sheet2": DataFrame, ...}
        first_key = list(df.keys())[0]
        print(f"ℹ️ Multiple sheets detected. Using first sheet: {first_key}")
        df = df[first_key]

    if not isinstance(df, pd.DataFrame):
        raise Exception(f"❌ Excel read failed. Got type: {type(df)}")

    return df

def main():
    # ----------------------------
    # Init Firebase
    # ----------------------------
    if not os.path.exists(SERVICE_KEY):
        raise FileNotFoundError(
            f"Firebase key not found: {SERVICE_KEY}\n"
            "Put serviceAccountKey.json in the Backend folder."
        )
    cred = credentials.Certificate(SERVICE_KEY)

    # Avoid "already initialized" error if running multiple times
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # ----------------------------
    # Read Excel (first path that exists)
    # ----------------------------
    excel_path = None
    for p in EXCEL_CANDIDATES:
        if os.path.exists(p):
            excel_path = p
            break
    if not excel_path:
        raise FileNotFoundError(
            f"Labour Excel file not found. Tried:\n  " + "\n  ".join(EXCEL_CANDIDATES)
            + "\n\nPut Labour_Expanded_clean_generated.xlsx in Backend/ or Backend/data/"
        )
    print(f"📂 Using Excel: {excel_path}")
    df = read_excel_safely(excel_path, SHEET_NAME)

    print("✅ Excel loaded.")
    print("📌 Columns:", df.columns.tolist())

    # ----------------------------
    # Validate ID column exists
    # ----------------------------
    if COL_ID not in df.columns:
        raise Exception(
            f"❌ ID column '{COL_ID}' not found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"➡️ Fix: Set COL_ID to the correct column name."
        )

    # ----------------------------
    # Clean column names (optional)
    # ----------------------------
    cleaned_cols = {c: clean_field_name(c) for c in df.columns}
    df = df.rename(columns=cleaned_cols)

    id_col_clean = clean_field_name(COL_ID)

    if id_col_clean not in df.columns:
        raise Exception(
            f"❌ Cleaned ID column '{id_col_clean}' not found after rename.\n"
            f"Available columns after clean: {list(df.columns)}"
        )

    # ----------------------------
    # Upload to Firestore (batch)
    # ----------------------------
    batch = db.batch()
    count = 0

    for _, row in df.iterrows():
        labour_id = str(row.get(id_col_clean, "")).strip()

        if not labour_id or labour_id.lower() == "nan":
            continue

        data = {}
        for k, v in row.to_dict().items():
            data[k] = normalize_value(v)

        # store doc id + original key
        data["labourId"] = labour_id
        data["Labour_ID"] = labour_id

        doc_ref = db.collection(COLLECTION).document(labour_id)
        batch.set(doc_ref, data, merge=True)

        count += 1

        # Firestore batch write limit ~500, keep safe at 450
        if count % 450 == 0:
            batch.commit()
            batch = db.batch()
            print(f"✅ Uploaded {count} records...")

    # final commit
    if count % 450 != 0:
        batch.commit()

    print(f"🎉 DONE! Total uploaded: {count} into '{COLLECTION}' collection.")

if __name__ == "__main__":
    main()
