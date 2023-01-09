"""Generate trace funtion given cset."""
# pylint: disable=invalid-name
from typing import Any, Callable, Tuple, Union
from nptyping import NDArray, Shape, Float

from scipy import interpolate


def gen_trace(
    cset: NDArray[Shape["Any, 3"], Float],
    fill_value: Union[str, Tuple[int, int]] = "extrapolate",
) -> Callable:
    """Generate trace funtion given cset.

    Args:
        cset, clustered triple set, two-tuple set
        fill_value: endpoint (along with (0, 0) for interp1d or using 'extrapolate'

    Returns:
        trace functions

    """
    x = cset.T[0]
    y = cset.T[1]
    if fill_value == "extrapolate":
        trace = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
    else:
        [x_b, y_b], [x_e, y_e] = [0, 0], fill_value
        trace = interpolate.interp1d([x_b] + x + [x_e], [y_b] + y + [y_e], kind='linear')

    return trace
