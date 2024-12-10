from django.http import HttpResponse
from WAF import SQLInjectionWAF
import os
from urllib.parse import unquote
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'sqli.pkl')
vectorizer_path = os.path.join(current_dir, 'models', 'vectorizerSqli.pkl')

def preprocess_path(path, vectorizer):
    try:
        # Decode URL-encoded characters
        decoded_path = unquote(path)
        print(f"Decoded Path: {decoded_path}")  # Debugging

        # Extract the last segment of the path
        last_segment = decoded_path.split('/')[-2] if '/' in decoded_path else decoded_path
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

class RusicadeWAFMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.waf = SQLInjectionWAF(model_path, vectorizer_path)

    def __call__(self, request):
        # Get the client's IP address
        client_ip = request.META.get('REMOTE_ADDR')
        print(f"Client IP: {client_ip}")  # Debugging

        # Get the URL path
        path = request.path
        print(f"Checking URL: {path}")  # Debugging

        # Preprocess the path
        preprocessed_path = preprocess_path(path, self.waf.vectorizer)

        # Detect SQL injection
        if self.waf.detect(preprocessed_path):
            return HttpResponse("""
                <html>
                    <head><title>Access Denied :Rusicade WAF</title></head>
                    <body>
                        <h1 style="color:red"> Rusicade WAF - Web Application Firewall</h1>
                        <h1>Error: Potential SQL Injection Detected!</h1>
                        <p>Your request has been blocked due to suspicious activity.</p>
                    </body>
                </html>
            """, status=400)  # Return HTML error page with 400 status code

        response = self.get_response(request)
        return response