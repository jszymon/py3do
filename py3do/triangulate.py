"""Triangulation of general polygonal regions (Seidel's algorithm).

The region is given by any number of closed cycles of vertices.  Cycles
may be nested (holes, islands in holes, ...), may be disjoint, and may
share vertices or edges.  Edges must not cross each other and a vertex
must not lie in the interior of an edge.

The algorithm follows Seidel (1991):

1. a trapezoidal decomposition of the plane is built by inserting edges
   in random order, using a DAG as point location structure,
2. trapezoids are classified as inside/outside based on the winding
   number,
3. each inside trapezoid whose two defining vertices are not joined by
   a polygon edge gets a diagonal between them; this splits the region
   into monotone polygons,
4. each monotone polygon is triangulated in linear time.

Degeneracies (equal x coordinates) are handled by a symbolic shear,
i.e. points are ordered lexicographically by (x, y).  Collinear
vertices are allowed.
"""

import random
from functools import cmp_to_key

import numpy as np


_LEAF, _XNODE, _YNODE = 0, 1, 2


class _Trap:
    """Trapezoid.  top/bottom are segment indices, leftp/rightp point
    indices; -1 stands for infinity."""
    __slots__ = ("top", "bottom", "leftp", "rightp", "node")

    def __init__(self, top, bottom, leftp, rightp):
        self.top = top
        self.bottom = bottom
        self.leftp = leftp
        self.rightp = rightp
        self.node = _Node(_LEAF, self)


class _Node:
    """DAG node.  Leaf: a=trapezoid.  X-node: a=point, left/right
    subtrees.  Y-node: a=segment, left=above, right=below."""
    __slots__ = ("kind", "a", "left", "right")

    def __init__(self, kind, a, left=None, right=None):
        self.kind = kind
        self.a = a
        self.left = left
        self.right = right


class _TrapezoidMap:
    def __init__(self, P, segp, segq):
        self.P = P
        self.segp = segp
        self.segq = segq
        t = _Trap(-1, -1, -1, -1)
        self.root = t.node
        self.traps = {t}

    def orient(self, a, b, c):
        P = self.P
        ax, ay = P[a]
        bx, by = P[b]
        cx, cy = P[c]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _side(self, a, b, c):
        o = self.orient(a, b, c)
        if o == 0:
            raise ValueError("triangulate: overlapping edges or vertex"
                             " lying on an edge")
        return o > 0

    def seg_above(self, s, u):
        """Is segment s above segment u?  The segments must not cross
        and their x ranges must overlap."""
        sp, sq = self.segp[s], self.segq[s]
        up, uq = self.segp[u], self.segq[u]
        if sp == up:
            return self._side(up, uq, sq)
        if self.P[sp] > self.P[up]:
            return self._side(up, uq, sp)
        return not self._side(sp, sq, up)

    def locate(self, s, r):
        """Find the trapezoid containing the part of segment s just
        right of point r."""
        P = self.P
        node = self.root
        while node.kind != _LEAF:
            if node.kind == _XNODE:
                node = node.right if P[node.a] <= P[r] else node.left
            else:
                node = node.left if self.seg_above(s, node.a) else node.right
        return node.a

    def insert(self, s):
        P = self.P
        p, q = self.segp[s], self.segq[s]
        crossed = [self.locate(s, p)]
        while True:
            r = crossed[-1].rightp
            if r == -1 or P[r] >= P[q]:
                break
            crossed.append(self.locate(s, r))

        # pieces above and below s
        first, last = crossed[0], crossed[-1]
        cur_up = _Trap(first.top, s, p, -1)
        cur_lo = _Trap(s, first.bottom, p, -1)
        new_traps = [cur_up, cur_lo]
        up_for, lo_for = [], []
        for j, t in enumerate(crossed):
            up_for.append(cur_up)
            lo_for.append(cur_lo)
            if j < len(crossed) - 1:
                r = t.rightp
                if self._side(p, q, r):
                    # r above s: the wall through r splits upper pieces
                    cur_up.rightp = r
                    cur_up = _Trap(crossed[j+1].top, s, r, -1)
                    new_traps.append(cur_up)
                else:
                    cur_lo.rightp = r
                    cur_lo = _Trap(s, crossed[j+1].bottom, r, -1)
                    new_traps.append(cur_lo)
        cur_up.rightp = q
        cur_lo.rightp = q
        A = B = None
        if first.leftp != p:
            A = _Trap(first.top, first.bottom, first.leftp, p)
            new_traps.append(A)
        if last.rightp != q:
            B = _Trap(last.top, last.bottom, q, last.rightp)
            new_traps.append(B)

        # update the DAG: replace leaves of crossed trapezoids in place
        for j, t in enumerate(crossed):
            sub = _Node(_YNODE, s, up_for[j].node, lo_for[j].node)
            if j == len(crossed) - 1 and B is not None:
                sub = _Node(_XNODE, q, sub, B.node)
            if j == 0 and A is not None:
                sub = _Node(_XNODE, p, A.node, sub)
            node = t.node
            node.kind, node.a = sub.kind, sub.a
            node.left, node.right = sub.left, sub.right
            self.traps.discard(t)
        self.traps.update(new_traps)


def _fill(w, fill_rule):
    if fill_rule == "evenodd":
        return w % 2 == 1
    if fill_rule == "nonzero":
        return w != 0
    if fill_rule == "positive":
        return w > 0
    raise ValueError(f"triangulate: unknown fill_rule {fill_rule!r}")


def _winding_numbers(tmap, segd):
    """Compute winding number just above each segment."""
    nseg = len(segd)
    wa = [None] * nseg
    by_bottom = [[] for _ in range(nseg)]
    by_top = [[] for _ in range(nseg)]
    queue = []

    def assign(s, w):
        if wa[s] is None:
            wa[s] = w
            queue.append(s)
        elif wa[s] != w:
            raise ValueError("triangulate: inconsistent winding numbers,"
                             " do edges intersect?")

    for t in tmap.traps:
        if t.bottom != -1:
            by_bottom[t.bottom].append(t.top)
        if t.top != -1:
            by_top[t.top].append(t.bottom)
    for t in tmap.traps:
        if t.top == -1 and t.bottom != -1:
            assign(t.bottom, 0)
        elif t.bottom == -1 and t.top != -1:
            assign(t.top, segd[t.top])
    while queue:
        s = queue.pop()
        for u in by_bottom[s]:
            if u != -1:
                assign(u, wa[s] + segd[u])
        for b in by_top[s]:
            if b != -1:
                assign(b, wa[s] - segd[s])
    return wa


def _triangulate_monotone(face, P, orient, out):
    """Triangulate a CCW polygon monotone w.r.t. lexicographic order."""
    n = len(face)
    if len(set(face)) != n:
        raise ValueError("triangulate: non-simple face (pinched vertex)")

    def add(a, b, c):
        o = orient(a, b, c)
        if o > 0:
            out.append((a, b, c))
        elif o < 0:
            out.append((a, c, b))

    if n == 3:
        add(*face)
        return
    imin = min(range(n), key=lambda i: P[face[i]])
    imax = max(range(n), key=lambda i: P[face[i]])
    lower = set()
    i = imin
    while i != imax:
        lower.add(face[i])
        i = (i + 1) % n
    order = sorted(face, key=lambda v: P[v])

    stack = [order[0], order[1]]
    for j in range(2, n - 1):
        u = order[j]
        u_low = u in lower
        if u_low != (stack[-1] in lower):
            while len(stack) > 1:
                v = stack.pop()
                add(u, v, stack[-1])
            stack = [order[j-1], u]
        else:
            last = stack.pop()
            while stack:
                o = orient(stack[-1], last, u)
                if (o > 0) if u_low else (o < 0):
                    add(stack[-1], last, u)
                    last = stack.pop()
                else:
                    break
            stack.append(last)
            stack.append(u)
    u = order[-1]
    while len(stack) > 1:
        v = stack.pop()
        add(u, v, stack[-1])


def triangulate(vertices, cycles=None, fill_rule="evenodd", seed=None):
    """Triangulate a polygonal region.

    vertices: array of shape (N, 2).
    cycles: list of integer arrays, each a closed cycle of indices into
        vertices.  If None, all vertices form a single cycle.
    fill_rule: which areas are inside: "evenodd" (default, orientation
        of cycles is irrelevant), "nonzero" or "positive" (based on
        winding number, CCW cycles count +1).
    seed: seed for random insertion order (affects only running time).

    Returns an (M, 3) array of indices into vertices.  All triangles are
    oriented counterclockwise.  Coincident vertices are merged; the
    lowest index of a group of coincident vertices is used in the
    output.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("triangulate: vertices must have shape (N, 2)")
    if cycles is None:
        cycles = [np.arange(vertices.shape[0])]

    # canonical (merged) points
    P = []
    orig_idx = []
    canon = {}
    def get_point(i):
        xy = (float(vertices[i, 0]), float(vertices[i, 1]))
        c = canon.get(xy)
        if c is None:
            c = canon[xy] = len(P)
            P.append(xy)
            orig_idx.append(i)
        return c

    # segments with net direction, zero length segments dropped
    seg_dir = {}
    for cycle in cycles:
        cycle = [get_point(int(i)) for i in cycle]
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            if a == b:
                continue
            if P[a] < P[b]:
                seg_dir[(a, b)] = seg_dir.get((a, b), 0) + 1
            else:
                seg_dir[(b, a)] = seg_dir.get((b, a), 0) - 1
    segs = [(k, d) for k, d in seg_dir.items() if d != 0]
    if len(segs) == 0:
        return np.zeros((0, 3), dtype=int)
    segp = [k[0] for k, d in segs]
    segq = [k[1] for k, d in segs]
    segd = [d for k, d in segs]

    # trapezoidal decomposition
    tmap = _TrapezoidMap(P, segp, segq)
    order = list(range(len(segs)))
    random.Random(seed).shuffle(order)
    for s in order:
        tmap.insert(s)
    wa = _winding_numbers(tmap, segd)

    # diagonals splitting the region into monotone polygons
    diagonals = []
    for t in tmap.traps:
        if t.bottom == -1 or not _fill(wa[t.bottom], fill_rule):
            continue
        a, b = t.leftp, t.rightp
        if a == segp[t.top] and b == segq[t.top]:
            continue
        if a == segp[t.bottom] and b == segq[t.bottom]:
            continue
        diagonals.append((a, b))

    # planar graph; half edge -> is the face on its left inside?
    left_inside = {}
    for s in range(len(segs)):
        p, q = segp[s], segq[s]
        left_inside[(p, q)] = _fill(wa[s], fill_rule)
        left_inside[(q, p)] = _fill(wa[s] - segd[s], fill_rule)
    for a, b in diagonals:
        left_inside[(a, b)] = left_inside[(b, a)] = True
    nbrs = {}
    for a, b in left_inside:
        nbrs.setdefault(a, []).append(b)

    def angle_cmp(c):
        cx, cy = P[c]
        def half(v):
            dx, dy = P[v][0] - cx, P[v][1] - cy
            return 0 if (dy > 0 or (dy == 0 and dx > 0)) else 1
        def cmp(u, v):
            hu, hv = half(u), half(v)
            if hu != hv:
                return hu - hv
            o = tmap.orient(c, u, v)
            return -1 if o > 0 else (1 if o < 0 else 0)
        return cmp_to_key(cmp)
    pos = {}
    for c, nb in nbrs.items():
        nb.sort(key=angle_cmp(c))
        for i, v in enumerate(nb):
            pos[(c, v)] = i

    # walk inside faces and triangulate them
    tris = []
    visited = set()
    for he, inside in left_inside.items():
        if not inside or he in visited:
            continue
        face = []
        u, v = he
        while (u, v) not in visited:
            visited.add((u, v))
            face.append(u)
            nb = nbrs[v]
            u, v = v, nb[pos[(v, u)] - 1]  # next clockwise around v
        _triangulate_monotone(face, P, tmap.orient, tris)

    if len(tris) == 0:
        return np.zeros((0, 3), dtype=int)
    return np.asarray(orig_idx)[np.array(tris, dtype=int)]
