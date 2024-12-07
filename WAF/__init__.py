from urllib.parse import unquote
from flask import request, jsonify
import joblib
import numpy as np
import pickle
import os
# Custom tokenization function to capture SQL injection patterns
def custom_tokenizer(text):
    return text.split()

# Custom Unpickler to load the custom tokenizer
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'custom_tokenizer':
            return custom_tokenizer
        return super().find_class(module, name)

# Load the saved model
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file paths
model_path = os.path.join(current_dir, 'nb.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    content = f.read(10)
    print(content)

# Load the saved model and vectorizer
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


try:
    with open(vectorizer_path, 'rb') as f:
        vectorizer = CustomUnpickler(f).load()
    print("Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    vectorizer = None

# Function to preprocess the input path for the model
def preprocess_path(path):
    try:
        # Decode URL-encoded characters
        decoded_path = unquote(path)
        print(f"Decoded Path: {decoded_path}")  # Debugging

        # Extract the last segment of the path
        last_segment = decoded_path.split('/')[-1]
        print(f"Last Segment: {last_segment}")  # Debugging

        # Convert to a format suitable for the model
        if vectorizer:
            vectorized_path = vectorizer.transform([last_segment]).toarray()
            print(f"Vectorized Path: {vectorized_path}")  # Debugging

            # Check if the vectorized path is all zeros
            if not np.any(vectorized_path):
                print("Warning: Vectorized path is all zeros. No meaningful tokens detected.")
                return None
            
            return vectorized_path
        else:
            print("Vectorizer not loaded.")
            return None

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# SQL Injection Detection Function using the model
def detect_sql_injection(path):
    preprocessed_path = preprocess_path(path)
    if preprocessed_path is None:
        return False  # No meaningful tokens, assume no SQL injection

    try:
        prediction = model.predict(preprocessed_path) if model else [0]
        print(f"Prediction: {prediction}")  # Debugging
        return prediction[0] == 1  # Assuming 1 indicates SQL injection
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False

# Middleware function to monitor requests and detect SQL injection
def monitor_sql_injection(app):
    @app.before_request
    def monitor_request():
        # Get the URL path
        path = request.path
        print(f"Checking URL: {path}")  # Debugging

        # Detect SQL injection
            # Detect SQL injection
        if detect_sql_injection(path):
            return """
            <html>
                <head><title>Access Denied :Rusicade WAF</title></head>
                <body>
                    <h1 style="color:red"> Rusicade WAF - Web Application Firewall</h1>
                    <h1>Error: Potential SQL Injection Detected!</h1>
                    <p>Your request has been blocked due to suspicious activity.</p>
                </body>
            </html>
            """, 400  # Return HTML error page with 400 status code