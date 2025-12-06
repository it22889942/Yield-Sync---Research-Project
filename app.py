from flask import Flask, request, jsonify, render_template
import joblib
import re
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# -----------------------------
# Load model and components
# -----------------------------
# Make sure this path matches what you saved in train.py
model_data = joblib.load('models/complete_recommendation_model.joblib')

labour_clf = model_data['classifier_model_labour']
equip_clf = model_data['classifier_model_equipment']
vectorizer = model_data['vectorizer']

le_labour = model_data['label_encoder_labour']
le_equip = model_data['label_encoder_equipment']

DF_labour = model_data['DF_labour']
DF_equip = model_data['DF_equipment']

semantic_model = SentenceTransformer(model_data['embedding_model_name'])

print("✅ Models Loaded Successfully")

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    """Render the frontend HTML page."""
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle recommendation queries from frontend."""
    data = request.get_json()
    query = data.get('query', '')
    top_k = int(data.get('top_k', 5))

    if not query.strip():
        return jsonify({'error': 'Empty query'}), 400

    # -----------------------------------
    # 1. Predict Labour Type & Equipment
    # -----------------------------------
    qv = vectorizer.transform([query])

    labour_pred_id = labour_clf.predict(qv)[0]
    equip_pred_id = equip_clf.predict(qv)[0]

    labour_label = le_labour.inverse_transform([labour_pred_id])[0]
    equip_label = le_equip.inverse_transform([equip_pred_id])[0]

    # -----------------------------------
    # 2. Semantic embedding of the query
    # -----------------------------------
    query_emb = semantic_model.encode(query, convert_to_tensor=True)

    # -----------------------------------
    # 3. Labour recommendations
    # -----------------------------------
    df_lab = DF_labour.copy()
    if 'Labour_Type_collapsed' in df_lab.columns:
        df_lab = df_lab[df_lab['Labour_Type_collapsed'] == labour_label]

    if df_lab.empty:
        labour_results = []
    else:
        # similarity
        df_lab['similarity'] = df_lab['embedding'].apply(
            lambda x: util.cos_sim(query_emb, torch.tensor(x)).item()
        )

        # rate filter: uses Hourly_Rate if present
        m = re.search(r'(under|below|less than|more than|above)\s*(\d+)', query.lower())
        if m and 'Hourly_Rate' in df_lab.columns:
            direction, val = m.groups()
            val = float(val)
            if direction in ['under', 'below', 'less than']:
                df_lab = df_lab[df_lab['Hourly_Rate'] <= val]
            else:
                df_lab = df_lab[df_lab['Hourly_Rate'] >= val]

        if df_lab.empty:
            labour_results = []
        else:
            # score = 0.7 * similarity + 0.3 * rating (if available)
            if 'Rating' in df_lab.columns:
                df_lab['score'] = 0.7 * df_lab['similarity'] + 0.3 * (df_lab['Rating'] / 5.0)
            else:
                df_lab['score'] = df_lab['similarity']

            df_lab = df_lab.sort_values(by='score', ascending=False).head(top_k)

            columns_to_send = []
            for c in ['Name', 'Location', 'Labour_Type', 'Season', 'Crop_Type', 'Hourly_Rate', 'Rating', 'score']:
                if c in df_lab.columns:
                    columns_to_send.append(c)

            labour_results = df_lab[columns_to_send].to_dict(orient='records')

    # -----------------------------------
    # 4. Equipment recommendations
    # -----------------------------------
    df_eq = DF_equip.copy()
    if 'Equipment_Type' in df_eq.columns:
        df_eq = df_eq[df_eq['Equipment_Type'] == equip_label]

    if df_eq.empty:
        equip_results = []
    else:
        df_eq['similarity'] = df_eq['embedding'].apply(
            lambda x: util.cos_sim(query_emb, torch.tensor(x)).item()
        )

        # reuse same numeric filter, if dataset has Hourly_Rate_LKR
        m = re.search(r'(under|below|less than|more than|above)\s*(\d+)', query.lower())
        if m and 'Hourly_Rate_LKR' in df_eq.columns:
            direction, val = m.groups()
            val = float(val)
            if direction in ['under', 'below', 'less than']:
                df_eq = df_eq[df_eq['Hourly_Rate_LKR'] <= val]
            else:
                df_eq = df_eq[df_eq['Hourly_Rate_LKR'] >= val]

        if df_eq.empty:
            equip_results = []
        else:
            if 'Rating' in df_eq.columns:
                df_eq['score'] = 0.7 * df_eq['similarity'] + 0.3 * (df_eq['Rating'] / 5.0)
            else:
                df_eq['score'] = df_eq['similarity']

            df_eq = df_eq.sort_values(by='score', ascending=False).head(top_k)

            equip_columns_to_send = []
            # choose some sensible columns if they exist
            for c in [
                'Equipment_Type', 'For_Crop', 'Season',
                'Nearest_Major_District', 'Condition',
                'Hourly_Rate_LKR', 'Rating', 'score'
            ]:
                if c in df_eq.columns:
                    equip_columns_to_send.append(c)

            equip_results = df_eq[equip_columns_to_send].to_dict(orient='records')

    # -----------------------------------
    # 5. Return results as JSON
    # -----------------------------------
    return jsonify({
        'query': query,
        'predicted_labour_type': labour_label,
        'predicted_equipment_type': equip_label,
        'labour_recommendations': labour_results,
        'equipment_recommendations': equip_results
    })


# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
