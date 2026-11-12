"""Run a careport module unchanged (runpy, as `python -m`) and report its peak resident memory and wall time.
Usage: python run_measured.py <module> [args...]"""
import resource
import runpy
import sys
import time

t0 = time.time()
mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]
code = 0
try:
    runpy.run_module(mod, run_name="__main__")
except SystemExit as e:
    code = e.code or 0
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # ru_maxrss is in KiB on Linux
print(f"MEASURED module={mod} peak_rss_GB={peak_gb:.2f} wall_s={time.time() - t0:.0f} exit={code}", file=sys.stderr, flush=True)
sys.exit(code)
