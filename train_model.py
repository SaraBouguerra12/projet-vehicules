"""
Projet 16 : Classification des types de véhicules
Entraînement + comparaison modèles + MLflow
"""

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import mlflow
import mlflow.sklearn


# ─────────────────────────────
# MLflow setup
# ─────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("vehicle-classification")


print("=" * 60)
print("  PROJET 16 - Classification des Types de Véhicules")
print("=" * 60)


# ─────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────
df = pd.read_csv("vehicle_data.csv")

features = [col for col in df.columns if col != "CLASS"]
X = df[features].values
y = df["CLASS"].values


# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ─────────────────────────────
# 2. MODELS COMPARISON
# ─────────────────────────────
models = [
    ("RandomForest", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )),
    
    ("SVM", SVC(kernel="rbf", C=1, gamma="scale")),
    
    ("KNN", KNeighborsClassifier(n_neighbors=5))
]


results = []


for name, model in models:

    with mlflow.start_run(run_name=name):

        print(f"\n🚀 Training {name}...")

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)

        # ───── LOG MLflow ─────
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", accuracy)

        if name == "RandomForest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 15)

        if name == "SVM":
            mlflow.log_param("kernel", "rbf")

        if name == "KNN":
            mlflow.log_param("n_neighbors", 5)

        mlflow.sklearn.log_model(model, "model")

        print(f"✅ {name} Accuracy: {accuracy:.4f}")

        results.append((name, accuracy))


# ─────────────────────────────
# 3. BEST MODEL
# ─────────────────────────────
best_model = max(results, key=lambda x: x[1])

print("\n🏆 BEST MODEL:")
print(best_model)


# ─────────────────────────────
# 4. FINAL EVALUATION (best model report)
# ─────────────────────────────
best_name = best_model[0]

if best_name == "RandomForest":
    final_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
elif best_name == "SVM":
    final_model = SVC(kernel="rbf", C=1, gamma="scale")
else:
    final_model = KNeighborsClassifier(n_neighbors=5)

final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)


print("\n📊 Classification Report (BEST MODEL):")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# Cross validation
cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5)
print("\nCV mean:", cv_scores.mean())


# ─────────────────────────────
# 5. SAVE BEST MODEL
# ─────────────────────────────
joblib.dump(final_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")


metadata = {
    "features": features,
    "best_model": best_name,
    "accuracy": float(best_model[1])
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


print("\n✅ Training terminé avec comparaison MLflow")
