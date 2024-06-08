import os
from firebase_admin import storage

import firebase_admin
from firebase_admin import credentials, storage

# Path to your service account key file
SERVICE_ACCOUNT_KEY_PATH = 'CODE\serviceAccountKey.json'

# Initialize Firebase Admin SDK
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'djangopneumonia.appspot.com'
})


LOCAL_MODEL_PATH = 'finalapp/ml_models/oursecondmodel.pth'
REMOTE_MODEL_PATH = 'oursecondmodel.pth'  # The path in your Firebase Storage

def download_model():
    """Download the model from Firebase Storage if not present locally."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        bucket = storage.bucket()
        blob = bucket.blob(REMOTE_MODEL_PATH)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print(f"Model downloaded to {LOCAL_MODEL_PATH}.")
    else:
        print(f"Model already present at {LOCAL_MODEL_PATH}.")
