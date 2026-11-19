"""
End-to-end programmatic functional testing of all 16 Kallvettu Streamlit studios
using Streamlit's official AppTest framework (streamlit.testing.v1).
"""
import sys
import os
import time
import traceback

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

from streamlit.testing.v1 import AppTest

STUDIOS = [
    "📊 Dashboard",
    "🏛️ Inscription Studio",
    "🔬 Digital Restoration Studio",
    "✍️ Transcription Studio",
    "📜 Translation Studio",
    "📖 Lexical Analysis",
    "📑 Research Dossier",
    "🏷️ Glyph Annotation Studio",
    "🗺️ India Epigraphical Geographic Atlas",
    "📖 Classical Tamil Dictionary",
    "🔍 Epigraphical Search Engine",
    "🕸️ Historical Knowledge Graph",
    "🔎 IIIF Deep-Zoom Inscription Viewer",
    "💡 Photometric RTI & Relighting",
    "🔬 Research Evaluation Studio",
    "⚙️ Settings"
]

print("=" * 70)
print("KALLVETTU END-TO-END WORKSPACE FUNCTIONAL TEST HARNESS")
print("=" * 70)

results = {}

try:
    print("\n[INIT] Initializing AppTest from app.py...")
    t0 = time.time()
    at = AppTest.from_file(os.path.abspath("app.py"), default_timeout=45)
    at.run()
    init_time = time.time() - t0
    print(f"[INIT] App initial run completed in {init_time:.2f}s.")

    if at.exception:
        print("[ERROR] Initial load threw exception:", at.exception)
    else:
        print("[SUCCESS] Initial load clean without unhandled exceptions.")

    # Locate sidebar navigation radio
    nav_radio = None
    for r in at.radio:
        if "Dashboard" in str(r.options):
            nav_radio = r
            break

    if not nav_radio:
        print("[WARN] Could not find nav_radio in at.radio. Available radios:", len(at.radio))

    for studio_name in STUDIOS:
        print(f"\n" + "-" * 60)
        print(f"Testing Studio: {studio_name}")
        print("-" * 60)
        t_start = time.time()
        studio_status = "PASS"
        error_msg = None

        try:
            # Switch studio in session state or radio
            at.session_state["nav_studio_selector"] = studio_name
            if nav_radio:
                try:
                    nav_radio.set_value(studio_name)
                except Exception:
                    pass
            at.run(timeout=45)
            elapsed = time.time() - t_start

            # Check for exceptions
            if at.exception:
                exc_list = at.exception
                exc_texts = [str(e.value) for e in exc_list]
                print(f"  [FAIL] Uncaught exception in {studio_name}: {exc_texts}")
                studio_status = "FAIL"
                error_msg = "; ".join(exc_texts)
            else:
                # Count elements
                n_markdown = len(at.markdown)
                n_buttons = len(at.button)
                n_inputs = len(at.text_input) + len(at.text_area)
                n_metrics = len(at.metric)
                n_dataframes = len(at.dataframe)
                print(f"  [PASS] Rendered successfully in {elapsed:.2f}s | "
                      f"Metrics: {n_metrics}, Markdown: {n_markdown}, "
                      f"Buttons: {n_buttons}, Inputs: {n_inputs}, DataFrames: {n_dataframes}")

            results[studio_name] = {
                "status": studio_status,
                "elapsed_s": round(elapsed, 2),
                "error": error_msg,
                "metrics_count": len(at.metric),
                "buttons_count": len(at.button),
                "markdown_count": len(at.markdown)
            }

        except Exception as ex:
            elapsed = time.time() - t_start
            print(f"  [EXCEPTION] Studio test raised: {ex}")
            traceback.print_exc()
            results[studio_name] = {
                "status": "FAIL",
                "elapsed_s": round(elapsed, 2),
                "error": str(ex)
            }

except Exception as e:
    print("[FATAL] Test harness aborted:", e)
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST SUMMARY SUMMARY TABLE")
print("=" * 70)
passed_count = sum(1 for r in results.values() if r["status"] == "PASS")
total_count = len(results)
print(f"Total Studios Tested: {total_count} | Passed: {passed_count} | Failed: {total_count - passed_count}\n")
for name, data in results.items():
    status = data["status"]
    elapsed = data.get("elapsed_s", 0)
    err = f" (Error: {data['error']})" if data.get("error") else ""
    print(f"  [{status}] {name:<40} ({elapsed}s){err}")
