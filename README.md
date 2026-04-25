# 🚗 Projet 16 — Classification des Types de Véhicules

> Module : Machine Learning Avancée | 

---

## 📋 Description

Ce projet utilise un algorithme de **Random Forest** pour classifier automatiquement le type d'un véhicule (Bus, Van, Saab, Opel) à partir de ses caractéristiques techniques extraites par analyse d'image (silhouette).

---

## 🏗️ Architecture du Projet

```
projet_vehicules/
├── app.py              → Backend Flask (API REST)
├── train_model.py      → Entraînement du modèle ML
├── vehicle_data.csv    → Dataset (948 véhicules, 18 features)
├── model.pkl           → Modèle Random Forest sauvegardé
├── scaler.pkl          → Normaliseur (StandardScaler)
├── label_encoder.pkl   → Encodeur des labels
├── metadata.json       → Infos du modèle (accuracy, features...)
├── requirements.txt    → Dépendances Python
└── templates/
    └── index.html      → Interface web (Frontend)
```

---

## ⚙️ Technologies Utilisées

| Couche | Technologie |
|--------|-------------|
| **Frontend** | HTML5 + CSS3 + JavaScript |
| **Backend** | Python + Flask |
| **ML** | Scikit-learn (Random Forest) |
| **Data** | Pandas + NumPy |
| **Sauvegarde modèle** | Joblib |

---

## 🚀 Installation et Lancement

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Entraîner le modèle

```bash
python train_model.py
```

### 3. Lancer l'application

```bash
python app.py
```

### 4. Ouvrir dans le navigateur

```
http://localhost:5000
```

---

## 📊 Résultats du Modèle

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 82.63% |
| **Cross-validation (5-fold)** | 84.83% ± 2.97% |
| **Algorithme** | Random Forest (100 arbres) |

### Performance par classe :

| Classe | Précision | Rappel | F1-Score |
|--------|-----------|--------|----------|
| Bus    | 0.93      | 0.83   | 0.88     |
| Van    | 0.80      | 0.83   | 0.81     |
| Saab   | 0.86      | 0.88   | 0.87     |
| Opel   | 0.73      | 0.77   | 0.75     |

---

## 🔌 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web principale |
| `/predict` | POST | Prédiction (JSON → résultat) |
| `/info` | GET | Informations sur le modèle |
| `/example/<type>` | GET | Valeurs d'exemple (bus/van/saab/opel) |

### Exemple d'appel API :

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "COMPACTNESS": 95,
    "CIRCULARITY": 40,
    "DISTANCE_CIRCULARITY": 75,
    "RADIUS_RATIO": 270,
    ...
  }'
```

### Réponse :

```json
{
  "prediction": "bus",
  "confidence": 91.5,
  "info": {
    "label": "Bus",
    "emoji": "🚌",
    "description": "Grand véhicule de transport en commun"
  },
  "probabilities": {
    "bus": 91.5,
    "van": 5.2,
    "saab": 2.1,
    "opel": 1.2
  }
}
```

---

## 🔬 Features (18 caractéristiques)

| Feature | Description |
|---------|-------------|
| COMPACTNESS | Compacité de la silhouette |
| CIRCULARITY | Degré de circularité |
| DISTANCE_CIRCULARITY | Circularité basée sur la distance |
| RADIUS_RATIO | Ratio des rayons max/min |
| PR_AXIS_ASPECT_RATIO | Ratio d'aspect de l'axe principal |
| MAX_LENGTH_ASPECT_RATIO | Ratio longueur maximale |
| SCATTER_RATIO | Ratio de dispersion |
| ELONGATEDNESS | Degré d'élongation |
| PR_AXIS_RECT | Rectangularité de l'axe principal |
| MAX_LENGTH_RECT | Rectangularité longueur max |
| SCALED_VARIANCE_MAJOR | Variance axe majeur |
| SCALED_VARIANCE_MINOR | Variance axe mineur |
| SCALED_RADIUS_OF_GYRATION | Rayon de gyration |
| SKEWNESS_MAJOR | Asymétrie axe majeur |
| SKEWNESS_MINOR | Asymétrie axe mineur |
| KURTOSIS_MAJOR | Kurtosis axe majeur |
| KURTOSIS_MINOR | Kurtosis axe mineur |
| HOLLOWS_RATIO | Ratio des creux |

---


