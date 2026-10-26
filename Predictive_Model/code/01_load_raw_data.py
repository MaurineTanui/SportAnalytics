"""
01_load_raw_data.py
Step 1. Acquisition. Loads the 12 synthetic CSV files submitted in Week 5 (data/raw/), checks that they
are complete and unchanged (SHA-256 fingerprint), and records an inventory in outputs/raw_inventory.csv.
The raw files are never edited; every later step writes to new folders.
"""
import hashlib
import sys
import pandas as pd
import config as C

C.OUTPUTS.mkdir(parents=True, exist_ok=True)
rows, problems = [], []
for name, expected in C.RAW_FILES.items():
    path = C.RAW / name
    if not path.exists():
        problems.append(f"missing file: {name}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    df = pd.read_csv(path)
    rows.append({"file": name, "rows": len(df), "columns": df.shape[1], "sha256": digest,
                 "matches_week5_submission": digest == expected})
    if digest != expected:
        problems.append(f"{name} differs from the Week 5 submission")
inv = pd.DataFrame(rows)
inv.to_csv(C.OUTPUTS / "raw_inventory.csv", index=False)
print(inv[["file", "rows", "columns", "matches_week5_submission"]].to_string(index=False))
if problems:
    sys.exit("[01] stopped: " + "; ".join(problems))
print("[01] all 12 Week 5 files present and unchanged")
