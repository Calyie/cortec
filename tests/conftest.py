"""Make the `cortec` and `cortec_hybrid` packages importable regardless of the working directory.

Both packages live under `src/` at the repository root, beside this `tests/` folder. Without this the suite only
collected when pytest was invoked from the root with the packages installed; run it from elsewhere
and you got a collection ERROR, not results, which is how a defect once stayed hidden: a suite that
cannot be collected reports no failures.
"""
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
