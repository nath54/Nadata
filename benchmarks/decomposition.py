"""
decomposition.py — explains why the Python-facade column can be FASTER
than the C++ column in results.md, even though both run the same engine.

The three layers measured (same process, same machine):

  1. engine   — StructInstance.get_attribute + unwrap: the actual work
  2. facade   — Nadata.get_attribute: symbol lookup + wrapper layers
  3. C shim   — nd_get_attribute: facade call + UTF-8 + malloc + NdValue

Takeaway: the ENGINE cost is identical for every language. The C++/C
column pays a constant per-call BOUNDARY TAX (~1-2 us: crossing the
Python/C heap, converting the result string, allocating the NdValue)
that the Python facade never pays because its values already ARE
Python objects. ~1-2 us per call is an excellent FFI price — it buys
hosts a Python-free process and a stable ABI at hundreds of thousands
of ops/sec.

Usage: python benchmarks/decomposition.py
(run from the repository root; needs the dev shim built)
"""

# Import Modules
import ctypes
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["NADATA_HOME"] = os.getcwd()

base = Path(__file__).resolve().parent.parent
base_py = Path(sys.base_prefix)

if sys.platform == "win32":
    build_dir = base / "native" / "build" / "win32"
else:
    build_dir = (base / "native" / "build" /
                 ("darwin" if sys.platform == "darwin" else "linux"))

os.add_dll_directory(str(build_dir))
os.add_dll_directory(str(base_py))
os.add_dll_directory(str(base_py / "DLLs"))

N = 100000

dll = ctypes.PyDLL(str(build_dir /
    ("nadata.dll" if sys.platform == "win32"
     else "libnadata.dylib" if sys.platform == "darwin"
     else "libnadata.so")), winmode=0)
dll.nd_load.restype = ctypes.c_int
dll.nd_get_attribute.restype = ctypes.c_int
dll.nd_free_value.restype = None


class NdValue(ctypes.Structure):
    _fields_ = [("tag", ctypes.c_int), ("as", ctypes.c_void_p * 4)]


def us_per_op(start: float, ops: int = N) -> float:
    """Microseconds per operation for a finished timed loop."""

    return (time.perf_counter() - start) * 1e6 / ops


def main() -> None:
    """Runs the three-layer decomposition and prints the verdict."""

    db = ctypes.c_void_p()
    assert dll.nd_load(b"./examples/ex0_simple", ctypes.byref(db)) == 0

    out = NdValue()
    assert dll.nd_get_attribute(db, b"tom", b"birthplace",
                                ctypes.byref(out)) == 0

    # 3. C shim: facade call + UTF-8 + malloc + NdValue + free.
    start = time.perf_counter()

    for _ in range(N):
        dll.nd_get_attribute(db, b"tom", b"birthplace", ctypes.byref(out))
        dll.nd_free_value(ctypes.byref(out))

    shim = us_per_op(start)

    # 2. Python facade + 1. engine.
    from nadata.api import Nadata, _unwrap_value

    app: Nadata = Nadata.load(path="./examples/ex0_simple")

    start = time.perf_counter()

    for _ in range(N):
        app.get_attribute(symbol="tom", name="birthplace")

    facade = us_per_op(start)

    engine_get = app.get(symbol="tom").get_attribute

    start = time.perf_counter()

    for _ in range(N):
        _unwrap_value(engine_get("birthplace"))

    engine = us_per_op(start)

    print(f"engine only (instance.get_attribute + unwrap):   "
          f"{engine:.2f} us/op")
    print(f"python facade (+ symbol lookup, + wrapper):      "
          f"{facade:.2f} us/op")
    print(f"C shim (facade call + UTF-8 + malloc + NdValue): "
          f"{shim:.2f} us/op")
    print(f"per-op shim tax vs engine:                       "
          f"{shim - engine:+.2f} us")


if __name__ == "__main__":
    main()
