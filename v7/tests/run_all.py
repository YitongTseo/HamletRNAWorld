"""Minimal test runner — this project has no pytest in the shared venv.

Several test modules (test_flask_body, test_flask_embedder, test_sigma_controllers)
were written pytest-style with no `if __name__ == "__main__"` block, so running
them directly executed NOTHING and exited 0 — silently green. This runner
imports every tests/test_*.py and calls each test_* function, so those modules
are actually exercised.

    /home/web/.venv/bin/python tests/run_all.py [substring ...]
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    files = sorted((ROOT / "tests").glob("test_*.py"))
    if argv:
        files = [f for f in files if any(a in f.stem for a in argv)]

    total = passed = 0
    failures: list[str] = []
    for f in files:
        try:
            mod = load(f)
        except Exception:
            failures.append(f"{f.stem}: IMPORT FAILED")
            print(f"\n=== {f.stem} ===\n  IMPORT FAILED")
            traceback.print_exc()
            continue
        fns = [(k, v) for k, v in sorted(vars(mod).items())
               if k.startswith("test_") and callable(v)]
        if not fns:
            continue
        print(f"\n=== {f.stem} ({len(fns)} tests) ===")
        for name, fn in fns:
            total += 1
            try:
                fn()
                passed += 1
                print(f"  PASS {name}")
            except Exception as e:
                failures.append(f"{f.stem}::{name}")
                print(f"  FAIL {name}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}\n{passed}/{total} passed")
    if failures:
        print("failed:")
        for x in failures:
            print(f"  - {x}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
