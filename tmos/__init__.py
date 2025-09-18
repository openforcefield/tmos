"""Python script to determine the oxidation state of a metal complex"""

# Add imports here
import shutil
import os
from .tmos import *
from .utils import configure_logger as configure_logger

from ._version import __version__ as __version__

# Only check for obabel if not in documentation building environment
if not (os.environ.get("READTHEDOCS") or os.environ.get("SPHINX_BUILD")):
    if shutil.which("obabel") is None:
        raise ImportError(
            "The Open Babel executable 'obabel' is not installed or not found in PATH. Please install Open Babel to proceed."
        )
