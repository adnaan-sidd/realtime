from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

# Sample endpoints to display data
@app.route("/")
def index():
    """Main dashboard page showing current portfolio status and trade activity."""
    # Load sample data for trades and portfolio from JSON files (these should be updated live)
    if os.path.exists("data/portfolio.json"):
        with open("data/portfolio.json") as f:
            portfolio_data = json.load(f)
    else:
        portfolio_data = {"balance": 0, "open_trades": [], "metrics": {}}

    if os.path.exists("logs/main_log.txt"):
        with open("logs/main_log.txt") as f:
            logs = f.readlines()[-20:]  # Show the last 20 log entries
    else:
        logs = ["No logs available."]

    return render_template("index.html", portfolio=portfolio_data, logs=logs)

@app.route("/api/portfolio")
def portfolio_api():
    """API endpoint to fetch live portfolio data."""
    if os.path.exists("data/portfolio.json"):
        with open("data/portfolio.json") as f:
            portfolio_data = json.load(f)
    else:
        portfolio_data = {"balance": 0, "open_trades": [], "metrics": {}}
    return jsonify(portfolio_data)

@app.route("/api/logs")
def logs_api():
    """API endpoint to fetch the latest logs."""
    if os.path.exists("logs/main_log.txt"):
        with open("logs/main_log.txt") as f:
            logs = f.readlines()[-20:]
    else:
        logs = ["No logs available."]
    return jsonify({"logs": logs})

if __name__ == "__main__":
    app.run(debug=True)
