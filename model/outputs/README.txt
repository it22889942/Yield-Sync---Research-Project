Crop Recommendation Training Outputs
------------------------------------
Best model: GradientBoosting
Features: ['N', 'P', 'K', 'PH', 'Temperature', 'Humidity', 'Rainfall']
Classes: ['Beetroot', 'Onion', 'Radish', 'chickpea', 'maize', 'rice']

Artifacts:
- models/best_model.joblib: pipeline + label encoder + metadata
- reports/accuracy_table.csv: CV & test scores for all models
- reports/classification_report.txt: per-class precision/recall/F1
- reports/confusion_matrix.png: labeled confusion matrix
- reports/roc_micro.png, pr_micro.png: micro-avg curves (if scores)
- reports/feature_importances.png: (for tree models)
