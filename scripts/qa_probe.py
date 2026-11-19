"""
Test probe for starting Streamlit server and verifying health and HTTP endpoints.
"""
import subprocess
import time
import urllib.request
import sys
import os

print("Starting Streamlit process on port 8501...")
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=os.path.abspath(".")
)

url = "http://127.0.0.1:8501/_stcore/health"
success = False
try:
    for attempt in range(20):
        time.sleep(1)
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                status = resp.status
                body = resp.read().decode("utf-8")
                print(f"Attempt {attempt+1}: Streamlit health responded with status {status}: {body.strip()}")
                if status == 200:
                    success = True
                    break
        except Exception as e:
            print(f"Attempt {attempt+1}: waiting for server ({e})...")

    if success:
        with urllib.request.urlopen("http://127.0.0.1:8501/", timeout=5) as resp:
            html = resp.read().decode("utf-8")
            has_title = "<title>" in html
            print(f"Main page HTTP 200 OK: {len(html)} bytes, has_title: {has_title}")
    else:
        print("Failed to reach Streamlit health endpoint within 20 seconds.")
finally:
    proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=4)
        print("Process terminated.")
        if stdout:
            print("STDOUT preview:\n", stdout[:400])
        if stderr:
            print("STDERR preview:\n", stderr[:400])
    except Exception:
        proc.kill()
