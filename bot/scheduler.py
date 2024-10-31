import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

def run_main_script():
    """Run the main.py script every 2 hours."""
    while True:
        logging.info("Starting main.py...")
        result = subprocess.run(["python3", "main.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("main.py executed successfully.")
        else:
            logging.error(f"main.py execution failed with error: {result.stderr}")

        # Log output
        with open("logs/main_log.txt", "a") as log_file:
            log_file.write(result.stdout + "\n")
            log_file.write(result.stderr + "\n")

        logging.info("Waiting for 2 hours before the next execution.")
        time.sleep(7200)  # 7200 seconds = 2 hours

if __name__ == "__main__":
    run_main_script()
