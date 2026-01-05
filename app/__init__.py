from flask import Flask
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MongoDB setup
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
    client.server_info()
    db = client['brain_tumor_db']
    users_col = db['users']
    history_col = db['history']
    MONGODB_AVAILABLE = True
except Exception as e:
    print(f"Warning: MongoDB not available. Error: {e}")
    client = None
    db = None
    users_col = None
    history_col = None
    MONGODB_AVAILABLE = False

# Load models
DEFAULT_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
EYE_LABELS = ['0', '1', '2', '3', '4'] # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative
LUNG_LABELS = ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma']

class_labels = DEFAULT_LABELS
brain_model = None
eye_model = None
lung_model = None

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
brain_model_path = os.path.join(base_dir, 'models', 'brain_tumor_classifier_v2.keras')
eye_model_path = os.path.join(base_dir, 'models', 'diabetic_retinopathy_model.keras')
lung_model_path = os.path.join(base_dir, 'models', 'lung_cancer_model.keras')

# Load Brain Model
if os.path.exists(brain_model_path):
    try:
        brain_model = load_model(brain_model_path, compile=False)
        print(f"Successfully loaded brain model from: {brain_model_path}")
    except Exception as e:
        print(f"Failed to load brain model: {e}")
else:
    print(f"Brain model not found at {brain_model_path}")

# Load Eye Model
if os.path.exists(eye_model_path):
    try:
        eye_model = load_model(eye_model_path, compile=False)
        print(f"Successfully loaded eye model from: {eye_model_path}")
    except Exception as e:
        print(f"Failed to load eye model: {e}")
else:
    print(f"Eye model not found at {eye_model_path}")

# Load Lung Model
if os.path.exists(lung_model_path):
    try:
        lung_model = load_model(lung_model_path, compile=False)
        print(f"Successfully loaded lung model from: {lung_model_path}")
    except Exception as e:
        print(f"Failed to load lung model: {e}")
else:
    print(f"Lung model not found at {lung_model_path}")


from app import routes
