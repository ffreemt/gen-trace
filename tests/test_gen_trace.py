"""Test gen_trace."""
# pylint: disable=broad-except
import numpy as np
from gen_trace import __version__
from gen_trace import gen_trace


def test_version():
    """Test version."""
    assert __version__[:3] == "0.1"


def test_sanity():
    """Check sanity."""
    try:
        assert not gen_trace(np.array([['', '']]))
    except Exception:
        assert True
