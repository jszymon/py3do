"""Make shells of a flat ring (a circle with a hole) using all three
edge types: flat, inset and round.

The shells are placed side by side and shown.  Optionally they are
written to an STL file given as the first argument."""

import sys

import numpy as np

from py3do import Mesh, circle, shell
from py3do.triangulate import triangulate
from py3do.io import write_binary_stl
from py3do.vis import view_pyglet

n = 64
outer = circle(n) * 10
inner = circle(n) * 4
vs = np.vstack([outer, inner])
faces = triangulate(vs, [np.arange(n), np.arange(n, 2*n)])
ring = Mesh(np.column_stack([vs, np.zeros(2*n)]), faces)

m = None
for i, edge in enumerate(["flat", "inset", "round"]):
    s = shell(ring, 2, edge=edge, n_round=8)
    s.vertices[:,0] += 25 * i
    if m is None:
        m = s
    else:
        m.merge(s)

if len(sys.argv) > 1:
    write_binary_stl(m, sys.argv[1])
view_pyglet(m)
