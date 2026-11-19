"""
Minimal script to isolate where app.py blocks or hangs during execution.
"""
import sys
import os
import time

print("Step 1: Setting environment variables...")
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
sys.path.insert(0, os.path.abspath("."))

print("Step 2: Testing top-level imports...")
t0 = time.time()
import app
print(f"Imported app in {time.time() - t0:.2f}s")
