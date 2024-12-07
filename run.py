from flask import Flask, jsonify
from WAF import monitor_sql_injection  # Import the monitoring logic

app = Flask(__name__)

# Apply the monitoring middleware
monitor_sql_injection(app)

# Define Flask Routes
@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/user/<username>')
def user_profile(username):
    # Example route to simulate user profile response
    return jsonify({"user": username})

# Run Flask Application
if __name__ == "__main__":
    app.run(port=5000, debug=True)