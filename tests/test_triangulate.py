import numpy as np

import pytest

from py3do.triangulate import triangulate


def _signed_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def _tri_areas(V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                  (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

def _inside_evenodd(pt, V, cycles):
    x, y = pt
    inside = False
    for c in cycles:
        xy = V[c]
        x1, y1 = xy[:, 0], xy[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        cross = (y1 > y) != (y2 > y)
        xi = x1[cross] + (y - y1[cross]) * (x2[cross] - x1[cross]) / (y2[cross] - y1[cross])
        inside ^= bool(np.sum(xi > x) % 2)
    return inside

def _check(V, cycles, expected_area):
    T = triangulate(V, cycles)
    areas = _tri_areas(V, T)
    assert np.all(areas > 0)
    assert np.isclose(areas.sum(), expected_area)
    for t in T:
        assert _inside_evenodd(V[t].mean(axis=0), V, cycles)
    # every interior edge is shared by two triangles with opposite
    # orientation, boundary edges appear once
    edges = {}
    for t in T:
        for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
            assert (a, b) not in edges
            edges[(a, b)] = True
    boundary = [e for e in edges if e[::-1] not in edges]
    bnd_len = sum(np.hypot(*(V[a] - V[b])) for a, b in boundary)
    cyc_len = sum(np.hypot(*(V[c] - V[np.roll(c, -1)]).T).sum()
                  for c in cycles)
    assert np.isclose(bnd_len, cyc_len)
    return T

def test_triangle():
    V = np.array([[0.0, 0], [1, 0], [0, 1]])
    T = triangulate(V)
    assert T.shape == (1, 3)
    T = triangulate(V[::-1])
    assert T.shape == (1, 3)

def test_square_with_hole():
    V = np.array([[0.0, 0], [4, 0], [4, 4], [0, 4],
                  [1, 1], [1, 3], [3, 3], [3, 1]])
    cycles = [np.arange(4), np.arange(4, 8)]
    T = _check(V, cycles, 12.0)
    assert len(T) == 8
    # orientation of cycles does not matter
    _check(V, [np.arange(4), np.arange(4, 8)[::-1]], 12.0)

def test_nested_and_disjoint():
    V = []
    cycles = []
    def square(x0, y0, s):
        cycles.append(np.arange(len(V), len(V) + 4))
        V.extend([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]])
    square(0, 0, 10)
    square(1, 1, 8)  # hole
    square(2, 2, 6)  # island
    square(3, 3, 4)  # hole in island
    square(20, 0, 1) # separate polygon
    V = np.array(V, dtype=float)
    _check(V, cycles, 100 - 64 + 36 - 16 + 1)

def test_collinear():
    # comb with many collinear and vertically aligned vertices
    V = np.array([[0.0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2],
                  [3, 2], [3, 1], [2, 1], [2, 2], [1, 2], [1, 1], [0, 1]])
    _check(V, [np.arange(len(V))], 6.0)

def test_shared_edges_cancel():
    # 2x2 squares sharing edges, union is a square
    V = np.array([[x, y] for y in range(3) for x in range(3)], dtype=float)
    cycles = [np.array([0, 1, 4, 3]), np.array([1, 2, 5, 4]),
              np.array([3, 4, 7, 6]), np.array([4, 5, 8, 7])]
    T = triangulate(V, cycles)
    assert np.isclose(_tri_areas(V, T).sum(), 4.0)

def test_touching_vertices():
    # two squares touching at a corner
    V = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1],
                  [1, 1], [2, 1], [2, 2], [1, 2]])
    T = triangulate(V, [np.arange(4), np.arange(4, 8)])
    assert np.isclose(_tri_areas(V, T).sum(), 2.0)
    # holes touching the outer boundary at a vertex (left and right)
    V = np.array([[0.0, 0], [4, 0], [4, 4], [0, 4], [0, 2], [2, 1], [2, 3]])
    _check(V, [np.arange(5), np.array([4, 5, 6])], 14.0)
    V = np.array([[0.0, 0], [4, 0], [4, 2], [4, 4], [0, 4], [2, 1], [2, 3]])
    _check(V, [np.arange(5), np.array([2, 5, 6])], 14.0)

def test_duplicate_vertices():
    V = np.array([[0.0, 0], [1, 0], [1, 1], [1, 1], [0, 1], [0, 0]])
    T = triangulate(V)
    assert np.isclose(_tri_areas(V, T).sum(), 1.0)

def test_fill_rules():
    # two nested CCW squares
    V = np.array([[0.0, 0], [3, 0], [3, 3], [0, 3],
                  [1, 1], [2, 1], [2, 2], [1, 2]])
    cycles = [np.arange(4), np.arange(4, 8)]
    for rule, area in [("evenodd", 8), ("nonzero", 9), ("positive", 9)]:
        T = triangulate(V, cycles, fill_rule=rule)
        assert np.isclose(_tri_areas(V, T).sum(), area)
    T = triangulate(V[::-1], [np.arange(4)], fill_rule="positive")
    assert len(T) == 0  # CW square has negative winding

def _star(rng, cx, cy, r, n):
    a = np.sort(rng.uniform(0, 2 * np.pi, n))
    rr = r * rng.uniform(0.3, 1.0, n)
    return np.column_stack([cx + rr * np.cos(a), cy + rr * np.sin(a)])

@pytest.mark.parametrize("seed", range(20))
def test_random(seed):
    rng = np.random.default_rng(seed)
    outer = _star(rng, 0, 0, 10, int(rng.integers(3, 60)))
    pieces = [outer]
    area = _signed_area(outer)
    # holes: small stars near the center, each fits in the outer kernel
    for k, (hx, hy) in enumerate([(-1, -1), (1, 1), (1, -1)][:int(rng.integers(0, 4))]):
        h = _star(rng, hx, hy, 0.9, int(rng.integers(3, 20)))
        pieces.append(h)
        area -= _signed_area(h)
    if seed % 2:
        # snap to grid to create many collinear / aligned vertices
        pieces = [np.round(p * 4) / 4 for p in pieces]
        area = _signed_area(pieces[0]) - sum(_signed_area(p) for p in pieces[1:])
        # rounding might make stars non-simple; skip such cases
        for p in pieces:
            if len(np.unique(p, axis=0)) != len(p):
                pytest.skip("degenerate after rounding")
    V = np.vstack(pieces)
    cycles, start = [], 0
    for p in pieces:
        cycles.append(np.arange(start, start + len(p)))
        start += len(p)
    _check(V, cycles, area)

def test_large():
    rng = np.random.default_rng(0)
    outer = _star(rng, 0, 0, 10, 2000)
    T = triangulate(outer)
    assert len(T) == len(outer) - 2
    assert np.isclose(_tri_areas(outer, T).sum(), _signed_area(outer))
