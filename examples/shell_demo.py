"""Make shells of a flat ring (a circle with a hole) and of a dome
using both edge types, flat and inset, with and without rounding.

For the flat ring flat and inset edges are identical, for the dome
inset edges form a horizontal rim.  The shells are placed side by
side and shown.  Optionally they are written to an STL file given as
the first argument."""

import sys

import numpy as np

from py3do import Mesh, circle, uv_sphere, shell
from py3do.triangulate import triangulate
from py3do.io import write_binary_stl
from py3do.vis import view_pyglet

n = 64
outer = circle(n) * 10
inner = circle(n) * 4
vs = np.vstack([outer, inner])
faces = triangulate(vs, [np.arange(n), np.arange(n, 2*n)])
ring = Mesh(np.column_stack([vs, np.zeros(2*n)]), faces)

# a sphere of radius 10 cut at z=3, turned upside down
sphere = uv_sphere(n_u=n, n_v=n//2)
dome, _ = sphere.get_submesh(sphere.vertices[:,2] > 0.3)
dome.vertices *= 10
dome.rot90("XX")
dome = Mesh(dome.vertices, dome.faces)  # recompute normals

m = None
for j, sm in enumerate([ring, dome]):
    variants = [(e, r) for r in (False, True) for e in ("flat", "inset")]
    for i, (edge, rnd) in enumerate(variants):
        s = shell(sm, 2, edge=edge, round=rnd, n_round=8)
        s.vertices[:,2] -= s.vertices[:,2].min()  # rest on z=0
        s.vertices[:,0] += 25 * i
        s.vertices[:,1] += 25 * j
        if m is None:
            m = s
        else:
            m.merge(s)

if len(sys.argv) > 1:
    write_binary_stl(m, sys.argv[1])
view_pyglet(m)
