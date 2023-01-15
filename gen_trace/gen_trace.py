"""Generate trace funtion given cset."""
# pylint: disable=invalid-name
from typing import Any, Callable, Optional, Tuple, Union

from nptyping import Float, NDArray, Shape
from scipy import interpolate


def gen_trace(
    cset: NDArray[Shape["Any, 3"], Float],
    endpoint: Optional[Tuple[int, int]] = None,
) -> Callable:
    """Generate trace funtion given cset.

    Args:
        cset, clustered triple set, two-tuple set
        endpoint: along with (0, 0) for interp1d
            e.g. endpoint = cmat.T.shape  (cmat_.shape): ncol0, ncol1

    Returns:
        trace functions
    """
    x = cset.T[0]
    y = cset.T[1]
    if endpoint is None:
        trace = interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")
    else:
        [x_b, y_b], [x_e, y_e] = [0, 0], endpoint
        trace = interpolate.interp1d(
            [x_b] + x.tolist() + [x_e],
            [y_b] + y.tolist() + [y_e],
            kind="linear",
            fill_value="extrapolate",
        )

    return trace
