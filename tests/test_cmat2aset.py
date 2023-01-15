"""Test cmat2aset."""
import joblib
import numpy as np
from gen_trace.cmat2aset import cmat2aset


def test_cmat2aset_hero():
    """Test cmat2aset hero.npy."""
    cmat_ = np.load("tests/hero.npy")

    aset = cmat2aset(cmat_)

    n_row, n_col = cmat_.shape  # (1833, 596)
    assert len(aset) >= cmat_.shape[0]

    assert [elm[0] for elm in aset if elm[0] != ""].__len__() == n_row
    assert [elm[1] for elm in aset if elm[1] != ""].__len__() == n_col

    # number of anchors
    assert [elm[2] for elm in aset if elm[2] != ""].__len__() > 280  # 295

    # picked up the max?
    max0 = max([elm[2] for elm in aset if elm[2] != ""])  # 0.58
    max1 = cmat_.max()  # 0.5756367233340672

    assert np.isclose(max0, max1, atol=0.01)

    assert [elm[2] for elm in aset if elm[2] != ""].__len__() / min(cmat_.shape) > .45  # 0.495798


def test_cmat2aset():
    """Test cmat2aset.lzma."""
    cmat_ = joblib.load("tests/cmat.lzma")

    aset = cmat2aset(cmat_)

    n_row, n_col = cmat_.shape
    assert len(aset) >= cmat_.shape[0]

    assert [elm[0] for elm in aset if elm[0] != ""].__len__() == n_row
    assert [elm[1] for elm in aset if elm[1] != ""].__len__() == n_col

    # number of anchors
    n_anchor = [elm[2] for elm in aset if elm[2] != ""].__len__()
    assert n_anchor > 40  # 44

    # picked up the max?
    max0 = max([elm[2] for elm in aset if elm[2] != ""])  # 0.65
    max1 = cmat_.max()  # 0.6478468452684447

    assert np.isclose(max0, max1, atol=0.01)

    assert n_anchor / min(cmat_.shape) > 0.7  # 0.74576
