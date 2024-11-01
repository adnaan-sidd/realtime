import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def is_market_open():
    """Check if the Forex market is open based on weekday and time."""
    now = datetime.utcnow()
    weekday = now.weekday()  # Monday = 0, Sunday = 6
    hour = now.hour

    # Market is generally open from Sunday 5 PM EST to Friday 5 PM EST
    # Adjust for UTC: Open Sunday 10 PM UTC, Close Friday 10 PM UTC
    if weekday in range(0, 5):  # Monday to Friday
        if (weekday < 4) or (weekday == 4 and hour < 22):  # Ensure Friday closes at 22:00 UTC
            return True
    elif weekday == 6 and hour >= 22:  # Sunday 10 PM UTC start
        return True
    return False

def run_main_script():
    """Run the main.py script if the market is open."""
    if is_market_open():
        logging.info("Market is open, running main.py...")
        result = subprocess.run(["python3", "main.py"], capture_output=True, text=True)

        if result.returncode == 0:
            logging.info("main.py executed successfully.")
        else:
            logging.error(f"main.py execution failed with error: {result.stderr}")

        # Log output
        with open("logs/main_log.txt", "a") as log_file:
            log_file.write(result.stdout + "\n")
            log_file.write(result.stderr + "\n")
    else:
        logging.info("Market is closed. main.py will not run.")

if __name__ == "__main__":
    logging.info("Market monitoring started. Checking every 15 minutes.")
    while True:
        run_main_script()
        time.sleep(900)  # Check every 15 minutes (900 seconds)
