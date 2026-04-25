"""
Projet 16 : Classification des types de véhicules
Backend Flask - API REST
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np

app = Flask(__name__)

# ─────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

FEATURES = metadata['features']
CLASSES = metadata['classes']

# Descriptions des classes pour l'affichage
CLASS_INFO = {
    'bus': {
        'label': 'Bus',
        'emoji': '🚌',
        'description': 'Grand véhicule de transport en commun',
        'color': '#FF6B35'
    },
    'van': {
        'label': 'Van / Fourgonnette',
        'emoji': '🚐',
        'description': 'Véhicule utilitaire polyvalent',
        'color': '#4ECDC4'
    },
    'saab': {
        'label': 'Voiture (Saab)',
        'emoji': '🚗',
        'description': 'Véhicule de tourisme de type Saab',
        'color': '#45B7D1'
    },
    'opel': {
        'label': 'Voiture (Opel)',
        'emoji': '🚙',
        'description': 'Véhicule de tourisme de type Opel',
        'color': '#96CEB4'
    }
}


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html',
                           features=FEATURES,
                           accuracy=metadata['accuracy'],
                           cv_mean=metadata['cv_mean'])


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        data = request.get_json()

        # Extraire les valeurs dans le bon ordre
        values = []
        for feature in FEATURES:
            val = data.get(feature)
            if val is None:
                return jsonify({'error': f'Champ manquant : {feature}'}), 400
            values.append(float(val))

        # Préparer et prédire
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction_encoded = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]

        predicted_class = label_encoder.inverse_transform([prediction_encoded])[0]

        # Construire les probabilités pour chaque classe
        proba_dict = {}
        for i, cls in enumerate(label_encoder.classes_):
            proba_dict[cls] = round(float(probabilities[i]) * 100, 1)

        result = {
            'prediction': predicted_class,
            'info': CLASS_INFO.get(predicted_class, {}),
            'probabilities': proba_dict,
            'confidence': round(float(max(probabilities)) * 100, 1)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/info')
def info():
    """Informations sur le modèle"""
    return jsonify({
        'model': 'Random Forest Classifier',
        'accuracy': metadata['accuracy'],
        'cv_mean': metadata['cv_mean'],
        'classes': CLASSES,
        'n_features': len(FEATURES),
        'features': FEATURES
    })


@app.route('/example/<vehicle_type>')
def get_example(vehicle_type):
    """Retourne des valeurs d'exemple pour un type de véhicule"""
    examples = {
        'bus': {
            'COMPACTNESS': 95, 'CIRCULARITY': 40, 'DISTANCE_CIRCULARITY': 75,
            'RADIUS_RATIO': 270, 'PR_AXIS_ASPECT_RATIO': 90, 'MAX_LENGTH_ASPECT_RATIO': 12,
            'SCATTER_RATIO': 900, 'ELONGATEDNESS': 35, 'PR_AXIS_RECT': 145,
            'MAX_LENGTH_RECT': 195, 'SCALED_VARIANCE_MAJOR': 220, 'SCALED_VARIANCE_MINOR': 215,
            'SCALED_RADIUS_OF_GYRATION': 185, 'SKEWNESS_MAJOR': 5, 'SKEWNESS_MINOR': 7,
            'KURTOSIS_MAJOR': 7, 'KURTOSIS_MINOR': 12, 'HOLLOWS_RATIO': 195
        },
        'van': {
            'COMPACTNESS': 92, 'CIRCULARITY': 45, 'DISTANCE_CIRCULARITY': 80,
            'RADIUS_RATIO': 250, 'PR_AXIS_ASPECT_RATIO': 100, 'MAX_LENGTH_ASPECT_RATIO': 10,
            'SCATTER_RATIO': 950, 'ELONGATEDNESS': 30, 'PR_AXIS_RECT': 150,
            'MAX_LENGTH_RECT': 200, 'SCALED_VARIANCE_MAJOR': 240, 'SCALED_VARIANCE_MINOR': 210,
            'SCALED_RADIUS_OF_GYRATION': 190, 'SKEWNESS_MAJOR': 6, 'SKEWNESS_MINOR': 8,
            'KURTOSIS_MAJOR': 8, 'KURTOSIS_MINOR': 10, 'HOLLOWS_RATIO': 200
        },
        'saab': {
            'COMPACTNESS': 85, 'CIRCULARITY': 50, 'DISTANCE_CIRCULARITY': 85,
            'RADIUS_RATIO': 230, 'PR_AXIS_ASPECT_RATIO': 110, 'MAX_LENGTH_ASPECT_RATIO': 8,
            'SCATTER_RATIO': 850, 'ELONGATEDNESS': 40, 'PR_AXIS_RECT': 130,
            'MAX_LENGTH_RECT': 175, 'SCALED_VARIANCE_MAJOR': 200, 'SCALED_VARIANCE_MINOR': 195,
            'SCALED_RADIUS_OF_GYRATION': 170, 'SKEWNESS_MAJOR': 4, 'SKEWNESS_MINOR': 6,
            'KURTOSIS_MAJOR': 6, 'KURTOSIS_MINOR': 9, 'HOLLOWS_RATIO': 185
        },
        'opel': {
            'COMPACTNESS': 88, 'CIRCULARITY': 48, 'DISTANCE_CIRCULARITY': 82,
            'RADIUS_RATIO': 240, 'PR_AXIS_ASPECT_RATIO': 105, 'MAX_LENGTH_ASPECT_RATIO': 9,
            'SCATTER_RATIO': 880, 'ELONGATEDNESS': 38, 'PR_AXIS_RECT': 138,
            'MAX_LENGTH_RECT': 180, 'SCALED_VARIANCE_MAJOR': 210, 'SCALED_VARIANCE_MINOR': 200,
            'SCALED_RADIUS_OF_GYRATION': 178, 'SKEWNESS_MAJOR': 5, 'SKEWNESS_MINOR': 7,
            'KURTOSIS_MAJOR': 7, 'KURTOSIS_MINOR': 11, 'HOLLOWS_RATIO': 192
        }
    }

    if vehicle_type in examples:
        return jsonify(examples[vehicle_type])
    return jsonify({'error': 'Type de véhicule inconnu'}), 404


if __name__ == '__main__':
    print("🚗 Démarrage du serveur Flask...")
    print("📊 Modèle chargé avec succès")
    print(f"🎯 Accuracy : {metadata['accuracy']}%")
    print("🌐 Accès : http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
