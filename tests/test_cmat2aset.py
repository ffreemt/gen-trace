"""Test cmat2aset."""
import numpy as np
from gen_trace.cmat2aset import cmat2aset


def test_hero():
    """Test hero.npy."""
    cmat_ = np.load("tests/hero.npy")

    aset = cmat2aset(cmat_)

    n_row, n_col = cmat_.shape
    assert len(aset) >= cmat_.shape[0]

    assert [elm[0] for elm in aset if elm[0] != ""].__len__() == n_row
    assert [elm[1] for elm in aset if elm[1] != ""].__len__() == n_col

    # number of anchors
    assert [elm[2] for elm in aset if elm[2] != ""].__len__() > 280  # 295

    # picked up the max
    max0 = max([elm[2] for elm in aset if elm[2] != ""])  # 0.58
    max1 = cmat_.max()  # 0.5756367233340672

    assert np.isclose(max0, max1, atol=0.01)
