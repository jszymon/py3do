import numpy as np

import pytest

from py3do.geom_2d import two_segment_fillet

def _test_single_fillet(xy, c, a, r):
    xc, yc = c
    if a != 0:
        assert np.allclose(np.hypot(xy[:,0] - xc, xy[:,1] - yc), r)
    else:
        # degenerate cases
        assert (np.allclose(np.hypot(xy[:,0] - xc, xy[:,1] - yc), r) or 
                np.allclose(np.hypot(xy[:,0] - xc, xy[:,1] - yc), 0))

def test_fillet():
    for x0, x1, x2, y0, y1, y2 in [
            [0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0, -1.0, 0.0], # CCW
            [0.0, 1.0, 2.0, 0.0, 3.0, 0.0],
            [0.0, 1.0, 2.0, 0.0, 3.0, 1.0],
            [0.0, 1.0, 2.0, 0.0, -3.0, 1.0], # CCW
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0], # collinear
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], # overlapping
            [0.0, 1.0, 0.0, 0.0, -1.0, 0.0], # overlapping
    ]:
        r = 0.5
        xy, (xc, yc), a = two_segment_fillet(x0, y0, x1, y1, x2, y2, r)
        _test_single_fillet(xy, (xc, yc), a, r)
