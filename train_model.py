# train.py
# 100% ML Recommendation Engine (Local VS Code Version)
# ----------------------------------------------------

import os, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

# ---------------------------
# Step 1 - Folder Setup
# ---------------------------
os.makedirs('models', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('embeddings', exist_ok=True)

# ---------------------------
# Step 2 - Load Datasets
# ---------------------------
LABOUR_DATA_PATH = 'data/Labour_Expanded_clean_generated.xlsx'
EQUIP_DATA_PATH  = 'data/Equipment_Final_clean_generated.xlsx'

print("📥 Reading labour dataset from:", LABOUR_DATA_PATH)
DF_labour = pd.read_excel(LABOUR_DATA_PATH)
DF_labour.columns = [c.strip() for c in DF_labour.columns]

print("📥 Reading equipment dataset from:", EQUIP_DATA_PATH)
DF_equip = pd.read_excel(EQUIP_DATA_PATH)
DF_equip.columns = [c.strip() for c in DF_equip.columns]

# Fill missing essential columns for labour
for col in ['Labour_Type', 'Location', 'Season', 'Crop_Type',
            'Hourly_Rate', 'Rating', 'Skill_Level', 'Name']:
    if col not in DF_labour.columns:
        DF_labour[col] = 'unknown'
    DF_labour[col] = DF_labour[col].fillna('unknown')

# Fill missing essential columns for equipment
for col in ['Equipment_Type', 'For_Crop', 'Season',
            'Nearest_Major_District', 'Condition']:
    if col not in DF_equip.columns:
        DF_equip[col] = 'unknown'
    DF_equip[col] = DF_equip[col].fillna('unknown')

print("✅ Labour Data:", DF_labour.shape)
print("✅ Equipment Data:", DF_equip.shape)

# ---------------------------
# Step 3 - Feature Engineering
# ---------------------------
# Labour cleaning
for col in ['Labour_Type', 'Season', 'Crop_Type', 'Skill_Level', 'Location']:
    DF_labour[col] = DF_labour[col].astype(str).str.strip()

DF_labour['Hourly_Rate'] = pd.to_numeric(DF_labour['Hourly_Rate'], errors='coerce').fillna(0)
DF_labour['Rating']      = pd.to_numeric(DF_labour['Rating'], errors='coerce').fillna(0)

# Equipment cleaning
for col in ['Equipment_Type', 'For_Crop', 'Season',
            'Nearest_Major_District', 'Condition']:
    DF_equip[col] = DF_equip[col].astype(str).str.strip()

if 'Hourly_Rate_LKR' in DF_equip.columns:
    DF_equip['Hourly_Rate_LKR'] = pd.to_numeric(
        DF_equip['Hourly_Rate_LKR'], errors='coerce'
    ).fillna(0)
if 'Rating' in DF_equip.columns:
    DF_equip['Rating'] = pd.to_numeric(
        DF_equip['Rating'], errors='coerce'
    ).fillna(0)

# Request-style text for LABOUR (what user will type)
DF_labour['request_text'] = (
    DF_labour['Labour_Type'] + ' ' +
    DF_labour['Season'] + ' ' +
    DF_labour['Crop_Type'] + ' ' +
    DF_labour['Skill_Level'] + ' ' +
    DF_labour['Location']
)

# ⚠️ IMPORTANT FIX:
# Request-style text for EQUIPMENT – now includes Equipment_Type as well
DF_equip['request_text'] = (
    DF_equip['Equipment_Type'] + ' ' +            # <-- added
    DF_equip['For_Crop'] + ' ' +
    DF_equip['Season'] + ' ' +
    DF_equip['Nearest_Major_District'] + ' ' +
    DF_equip['Condition']
)

# Collapse rare labour types
vc = DF_labour['Labour_Type'].value_counts()
rare = vc[vc < 5].index.tolist()
DF_labour['Labour_Type_collapsed'] = DF_labour['Labour_Type'].apply(
    lambda x: x if x not in rare else 'other'
)

labour_label_col    = 'Labour_Type_collapsed'
equipment_label_col = 'Equipment_Type'

print("✅ Feature Engineering Done")

# ---------------------------
# Step 4 - Shared TF-IDF Vectorizer
# ---------------------------
# Fit TF-IDF over ALL request_text (labour + equipment)
all_text = pd.concat(
    [DF_labour['request_text'], DF_equip['request_text']],
    ignore_index=True
).values

vectorizer = TfidfVectorizer(
    stop_words=None,          # keep stopwords for better phrase matching
    ngram_range=(1, 3),       # keep phrases
    max_df=0.98,              # allow domain words
    min_df=1,                 # keep rare keywords
    max_features=60000        # bigger vocabulary
)

print("🔤 Fitting shared TF-IDF on labour + equipment text ...")
vectorizer.fit(all_text)
print("✅ TF-IDF Vocabulary Size:", len(vectorizer.vocabulary_))

# ---------------------------
# Step 5 - Prepare LABOUR Data for Classification
# ---------------------------
X_labour_text = DF_labour['request_text'].values
y_labour      = DF_labour[labour_label_col].values

le_labour = LabelEncoder()
y_labour_enc = le_labour.fit_transform(y_labour)

X_labour_train_text, X_labour_test_text, y_labour_train, y_labour_test = train_test_split(
    X_labour_text,
    y_labour_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_labour_enc
)

X_labour_train = vectorizer.transform(X_labour_train_text)
X_labour_test  = vectorizer.transform(X_labour_test_text)

print(f"🧩 Labour - Training Samples: {X_labour_train.shape[0]}, Test Samples: {X_labour_test.shape[0]}")
print("✅ Labour TF-IDF Matrices:", X_labour_train.shape, X_labour_test.shape)

# ---------------------------
# Step 6 - Model Comparison (Labour Type)
# ---------------------------
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=400, random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        eval_metric='mlogloss', n_jobs=-1, random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.1, random_state=42
    )
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, clf in models.items():
    print(f"⚙️ Evaluating {name} for Labour_Type ...")
    scores = cross_val_score(
        clf, X_labour_train, y_labour_train,
        cv=skf, scoring='accuracy', n_jobs=-1
    )
    results.append({'model': name, 'mean_acc': scores.mean(), 'std_acc': scores.std()})

res_df = pd.DataFrame(results).sort_values('mean_acc', ascending=False)
print("\n📊 Labour Type Model Comparison:\n", res_df)

plt.figure(figsize=(7, 5))
sns.barplot(data=res_df, x='model', y='mean_acc')
plt.title('Labour Type Model Accuracy Comparison (5-Fold CV)')
plt.ylabel('Mean Accuracy')
plt.savefig('outputs/plots/model_comparison_labour.png')
plt.close()

# ---------------------------
# Step 7 - Train Best Model (Labour Type)
# ---------------------------
best_name_labour = res_df.iloc[0]['model']
best_labour_clf = models[best_name_labour]
best_labour_clf.fit(X_labour_train, y_labour_train)
print(f"🏆 Best Labour Model Trained: {best_name_labour}")

# Optional: save labour-only model
joblib.dump(
    {
        'vectorizer': vectorizer,
        'label_encoder_labour': le_labour,
        'labour_model': best_labour_clf
    },
    f'models/best_labour_model_{best_name_labour}.joblib'
)

# ---------------------------
# Step 8 - Evaluate Labour Model
# ---------------------------
y_labour_pred = best_labour_clf.predict(X_labour_test)
print("✅ Labour Test Accuracy:", round(accuracy_score(y_labour_test, y_labour_pred), 3))
print(
    classification_report(
        y_labour_test, y_labour_pred,
        target_names=le_labour.classes_
    )
)

cm_labour = confusion_matrix(y_labour_test, y_labour_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_labour, annot=False, cmap='Blues')
plt.title(f'Confusion Matrix - Labour ({best_name_labour})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('outputs/plots/confusion_matrix_labour.png')
plt.close()

# ---------------------------
# Step 9 - Prepare EQUIPMENT Data & Train Model
# ---------------------------
X_equip_text = DF_equip['request_text'].values
y_equip      = DF_equip[equipment_label_col].values

le_equip = LabelEncoder()
y_equip_enc = le_equip.fit_transform(y_equip)

X_equip_train_text, X_equip_test_text, y_equip_train, y_equip_test = train_test_split(
    X_equip_text,
    y_equip_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_equip_enc
)

X_equip_train = vectorizer.transform(X_equip_train_text)
X_equip_test  = vectorizer.transform(X_equip_test_text)

print(f"🧩 Equipment - Training Samples: {X_equip_train.shape[0]}, Test Samples: {X_equip_test.shape[0]}")
print("✅ Equipment TF-IDF Matrices:", X_equip_train.shape, X_equip_test.shape)

# Use same model type that won for labour to keep config simple
if best_name_labour == 'RandomForest':
    equip_clf = RandomForestClassifier(
        n_estimators=400, random_state=42, n_jobs=-1
    )
elif best_name_labour == 'XGBoost':
    equip_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        eval_metric='mlogloss', n_jobs=-1, random_state=42
    )
else:  # LightGBM
    equip_clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.1, random_state=42
    )

print("⚙️ Training equipment model ...")
equip_clf.fit(X_equip_train, y_equip_train)

# Evaluate equipment model
y_equip_pred = equip_clf.predict(X_equip_test)
print("✅ Equipment Test Accuracy:", round(accuracy_score(y_equip_test, y_equip_pred), 3))
print(
    classification_report(
        y_equip_test, y_equip_pred,
        target_names=le_equip.classes_
    )
)

cm_equip = confusion_matrix(y_equip_test, y_equip_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_equip, annot=False, cmap='Greens')
plt.title(f'Confusion Matrix - Equipment ({best_name_labour}-based)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('outputs/plots/confusion_matrix_equipment.png')
plt.close()

# ---------------------------
# Step 10 - Semantic Model + Save Everything
# ---------------------------
print("🔁 Computing semantic embeddings (labour + equipment)...")
model_semantic = SentenceTransformer('all-MiniLM-L6-v2')

# Labour embeddings
DF_labour['text_for_embed'] = DF_labour['request_text'] + ' ' + DF_labour['Labour_Type']
DF_labour['embedding'] = DF_labour['text_for_embed'].apply(
    lambda x: model_semantic.encode(x, convert_to_tensor=True)
)

# Equipment embeddings
DF_equip['text_for_embed'] = (
    DF_equip['Equipment_Type'] + ' ' +
    DF_equip['For_Crop'] + ' ' +
    DF_equip['Season'] + ' ' +
    DF_equip['Nearest_Major_District']
)
DF_equip['embedding'] = DF_equip['text_for_embed'].apply(
    lambda x: model_semantic.encode(x, convert_to_tensor=True)
)

# Convert embeddings to numpy for saving
DF_labour_save = DF_labour.copy()
DF_labour_save['embedding'] = DF_labour_save['embedding'].apply(lambda x: x.cpu().numpy())

DF_equip_save = DF_equip.copy()
DF_equip_save['embedding'] = DF_equip_save['embedding'].apply(lambda x: x.cpu().numpy())

all_models = {
    # Classifiers
    'classifier_model_labour': best_labour_clf,
    'classifier_model_equipment': equip_clf,

    # Vectorizer + Encoders
    'vectorizer': vectorizer,
    'label_encoder_labour': le_labour,
    'label_encoder_equipment': le_equip,

    # Data with embeddings
    'DF_labour': DF_labour_save,
    'DF_equipment': DF_equip_save,

    # Semantic model info
    'embedding_model_name': 'all-MiniLM-L6-v2'
}

joblib.dump(all_models, 'models/complete_recommendation_model.joblib')
print("✅ Complete Labour + Equipment Model Saved (joblib) Successfully")
