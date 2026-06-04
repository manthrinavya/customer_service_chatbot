import schedule
import time
import subprocess

def update_job():
    print("Running scheduled knowledge base update...")
    subprocess.run(["python", "update_db.py"])

# Run every day at 2 AM
schedule.every().day.at("02:00").do(update_job)

print("Scheduler started. Updates will run daily at 2:00 AM")
while True:
    schedule.run_pending()
    time.sleep(60)