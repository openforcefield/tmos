import sys

# qcarchivetesting registers a broken pytest11 entry point in this environment.
# Unregister it before collection so it doesn't prevent tests from running.
sys.modules.setdefault("qcarchivetesting", type(sys)("qcarchivetesting"))
sys.modules.setdefault(
    "qcarchivetesting.pytest_config", type(sys)("qcarchivetesting.pytest_config")
)
