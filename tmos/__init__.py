"""Python script to determine the oxidation state of a metal complex"""

# Add imports here
import shutil
from .tmos import *

from ._version import __version__ as __version__

if shutil.which("obabel") is None:
    raise ImportError(
        "The Open Babel executable 'obabel' is not installed or not found in PATH. Please install Open Babel to proceed."
    )
