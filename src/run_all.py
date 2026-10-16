"""runs the four scripts in order. python src/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
steps = ["make_splits.py", "run_baselines.py", "explore_pos.py", "error_analysis.py"]

fail = []
for name in steps:
    print("\n=== %s ===" % name, flush=True)
    r = subprocess.run([PY, os.path.join(HERE, name)], cwd=os.getcwd())
    if r.returncode != 0:
        fail.append(name)

print("\n%d of %d steps failed%s" % (len(fail), len(steps), (": " + ", ".join(fail)) if fail else ""))
sys.exit(1 if fail else 0)
