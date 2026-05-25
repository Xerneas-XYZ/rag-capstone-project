import importlib.util
import sys
import traceback
from pathlib import Path


def run():
    # Ensure project root is on sys.path so `app` package imports work
    project_root = Path(".").resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    path = Path("tests") / "test_rewriter.py"
    spec = importlib.util.spec_from_file_location("test_rewriter", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_rewriter"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        traceback.print_exc()
        raise

    try:
        mod.test_rewrite_query_returns_stripped_content()
        mod.test_rewrite_query_invoked_with_expected_args()
    except AssertionError:
        traceback.print_exc()
        print("TESTS FAILED")
        raise SystemExit(1)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run()
