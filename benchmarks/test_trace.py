import sys, traceback
from benchmarks.n10_rigorous_benchmark import run_bench_25q

try:
    res = run_bench_25q()
    print("Success:", res)
except Exception as e:
    print("Caught error:", type(e), e)
    traceback.print_exc()
