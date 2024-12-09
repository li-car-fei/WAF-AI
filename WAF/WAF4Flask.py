# Middleware function to monitor requests and detect SQL injection
from flask import request, jsonify
from WAF import SQLInjectionWAF
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir,'models', 'sqli.pkl')
vectorizer_path = os.path.join(current_dir,'models','vectorizerSqli.pkl')
blocked_ips_file = os.path.join(current_dir, 'blocked_ips.json')

def load_blocked_ips():
    if os.path.exists(blocked_ips_file):
        with open(blocked_ips_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_blocked_ips(blocked_ips):
    with open(blocked_ips_file, 'w') as f:
        json.dump(list(blocked_ips), f)

blocked_ips = load_blocked_ips()

def rusicadeWAF(app):
    waf = SQLInjectionWAF(model_path, vectorizer_path)

    @app.before_request
    def monitor_request():
        # Get the client's IP address
        client_ip = request.remote_addr
        print(f"Client IP: {client_ip}")  # Debugging

        # Check if the IP is already blocked
        if client_ip in blocked_ips:
            return """
            <html>
                <head><title>Access Denied :Rusicade WAF</title></head>
                <body>
                    <center> <h1 style="color:red"> Rusicade WAF - Web Application Firewall</h1> </center> 
                    <h1>Error: Your IP has been blocked!</h1>
                    <p>Your request has been blocked due to suspicious activity.</p>
                    <p>Please contact the administrator for further assistance.</p>
                </body>
            </html>
            """, 403  # Return HTML error page with 403 status code

        # Get the URL path
        path = request.path
        print(f"Checking URL: {path}")  # Debugging

        # Detect SQL injection
        if waf.detect(path):
            # Block the IP address
            blocked_ips.add(client_ip)
            save_blocked_ips(blocked_ips)
            return """
            <html>
                <head><title>Access Denied :Rusicade WAF</title></head>
                <body>
                    <center> <h1 style="color:red"> Rusicade WAF - Web Application Firewall</h1> </center> 
                    <h1>Error: Potential SQL Injection Detected!</h1>
                    <p>Your request has been blocked due to suspicious activity.</p>
                </body>
            </html>
            """, 400  # Return HTML error page with 400 status code