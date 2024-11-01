from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import json
import os
import time

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html")

@app.route("/api/portfolio")
def portfolio_api():
    """API endpoint to fetch live portfolio data."""
    if os.path.exists("data/portfolio.json"):
        with open("data/portfolio.json") as f:
            portfolio_data = json.load(f)
    else:
        portfolio_data = {"balance": 0, "open_trades": [], "metrics": {}}
    return jsonify(portfolio_data)

def emit_portfolio_update():
    """Emit portfolio updates to the client every 5 seconds."""
    while True:
        if os.path.exists("data/portfolio.json"):
            with open("data/portfolio.json") as f:
                portfolio_data = json.load(f)
            socketio.emit("portfolio_update", portfolio_data)
        time.sleep(5)

@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    print("Client connected.")

if __name__ == "__main__":
    # Start emitting updates in the background
    socketio.start_background_task(emit_portfolio_update)
    socketio.run(app, debug=True)
