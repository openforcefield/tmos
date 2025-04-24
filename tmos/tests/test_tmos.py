"""
Unit and regression test for the tmos package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import tmos


def test_tmos_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "tmos" in sys.modules
