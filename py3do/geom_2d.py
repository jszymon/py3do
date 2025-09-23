import numpy as np


def two_segment_fillet(x0, y0, x1, y1, x2, y2, r, n=10):
    # fillet arc center
    dx1 = x1 - x0
    dy1 = y1 - y0
    dx2 = x2 - x1
    dy2 = y2 - y1

    # is clockwise?
    CW = (dx1 * (y2 - y0) - (x2 - y0) * dy1 <= 0)
    if not CW:
        r = -r

    # normalize dx, dy vectors
    l1 = np.hypot(dx1, dy1)
    l2 = np.hypot(dx2, dy2)
    if not ((l1 > 0) and (l2 > 0)):
        raise ValueError("two_segment_fillet: one of the segments has zero length")
    dx1 /= l1; dy1 /= l1
    dx2 /= l2; dy2 /= l2
    
    # find point at distance r from both lines
    A = np.array([[-dx1, dy1], [-dx2, dy2]])
    b = np.full(2, r)
    # projection of the center on both (normalized) segments should have length r
    # compute deltas from P1 such that least squares handles collinearity
    dyc, dxc = np.linalg.lstsq(A, b, rcond=None)[0]
    xc = x1 + dxc
    yc = y1 + dyc

    # fillet arc endpoints
    dc1 = dx1*dxc + dy1*dyc
    xa = x1 + dx1*dc1
    ya = y1 + dy1*dc1
    dc2 = dx2*dxc + dy2*dyc
    xb = x1 + dx2*dc2
    yb = y1 + dy2*dc2
    # is fillet possible?
    if abs(dc1/l1) > 1:
        raise ValueError("two_segment_fillet: radius too large for segment 1")
    if abs(dc2/l2) > 1:
        raise ValueError("two_segment_fillet: radius too large for segment 1")

    # arc's angular length
    a = np.pi - 2*np.arctan2(np.abs(r), np.abs(dc1))
    if not CW:
        a = -a

    # points on arc
    rx = xb - xc
    ry = yb - yc
    t = np.linspace(0, a, n)
    x = xc + rx * np.cos(t) - ry * np.sin(t)
    y = yc + rx * np.sin(t) + ry * np.cos(t)
    xy = np.column_stack([x, y])
    xy[0] = (xb, yb)
    xy[-1] = (xa, ya)
    return xy, (xc, yc), a
