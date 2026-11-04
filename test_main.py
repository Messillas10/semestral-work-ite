"""Run every test of the project.

Each test is a function whose name starts with "test_" inside a module of
TEST_MODULES. A test passes if it finishes without an AssertionError.

Usage:
    python test_main.py
"""

import sys

from tests import test_data, test_model

TEST_MODULES = [test_data, test_model]


def run_tests():
    """Run all test functions, print one line per test and a summary.

    Returns:
        Number of tests that failed.
    """
    failed = 0
    total = 0
    for module in TEST_MODULES:
        for name in dir(module):
            test = getattr(module, name)
            if not (name.startswith("test_") and callable(test)):
                continue
            total += 1
            try:
                test()
                print(f"PASS  {module.__name__}.{name}")
            except AssertionError as error:
                failed += 1
                print(f"FAIL  {module.__name__}.{name}  {error}")
            except Exception as error:  # UNEXPECTED CRASH, NOT A FAILED CHECK
                failed += 1
                print(f"ERROR {module.__name__}.{name}  {type(error).__name__}: {error}")
    print(f"\n{total - failed}/{total} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_tests() else 0)
