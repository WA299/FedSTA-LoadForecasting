"""Keep test checkpoint files in the workspace, avoiding Windows TEMP ACLs."""

from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    if config.option.basetemp is None:
        root = Path(__file__).resolve().parents[1] / ".pytest_tmp"
        root.mkdir(exist_ok=True)
        config.option.basetemp = str(root / uuid4().hex)
