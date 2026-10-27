"""
run_all.py
Runs the full pipeline in order: data steps 01-06 (Weeks 7-8) and model steps 07-10.
Usage (from the project folder):  python code/run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_load_raw_data.py", "02_clean_transform.py", "03_add_generated_fields.py", "04_compute_metrics.py",
         "05_simulate_validation_outcomes.py", "06_validate.py",
         "07_build_model_table.py", "08_train_evaluate_model.py", "09_validate_model.py",
         "10_train_performance_model.py"]
for step in STEPS:
    print(f"\n=== Running {step} ===")
    if subprocess.run([sys.executable, str(HERE / step)], cwd=HERE).returncode != 0:
        sys.exit(f"Stopped: {step} failed.")
print("\nPipeline finished. Final tables: data/final/  Model data and forecasts: data/model/  "
      "Withheld outcomes: data/withheld/  Logs and results: outputs/")
