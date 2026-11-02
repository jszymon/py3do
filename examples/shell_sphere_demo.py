"""Make shells of a sphere with caps cut away at six sides using both
edge types, flat and inset, with and without rounding.  The original
cut sphere is shown first, together with its offset.

Thickness is added on the inside.  Outside, the inset rims would have
to be trimmed back, which shell only does for boundary vertices, so
the surface would fold over.  The shells are placed side by side and
shown.  Optionally they are written to an STL file given as the first
argument."""

import sys

from py3do import Mesh, uv_sphere, shell, offset_mesh, slice_horiz_0
from py3do.io import write_binary_stl
from py3do.vis import view_pyglet

r = 10  # sphere radius
c = 8   # distance of cutting planes from the center, > r/sqrt(2)

m = uv_sphere(n_u=96)
m.vertices *= r
# rotations bringing each of six sides to the top
for rot in ["", "XX", "X", "XXX", "Y", "YYY"]:
    m.rot90(rot)
    m.vertices[:,2] -= c
    m = slice_horiz_0(m, keep="negative", fill=False)
    m.vertices[:,2] += c
    m.rot90(rot * 3)  # inverse rotation
m = Mesh(m.vertices, m.faces)  # recompute normals after rotations

# the original cut sphere together with its offset first, then the
# shells
out = m.clone()
out.merge(offset_mesh(m, -1.5))
out.vertices[:,2] -= out.vertices[:,2].min()  # rest on z=0
variants = [(e, rnd) for rnd in (False, True) for e in ("flat", "inset")]
for i, (edge, rnd) in enumerate(variants, 1):
    s = shell(m, -1.5, edge=edge, round=rnd, n_round=8)
    s.vertices[:,2] -= s.vertices[:,2].min()  # rest on z=0
    s.vertices[:,0] += 30 * i
    out.merge(s)

if len(sys.argv) > 1:
    write_binary_stl(out, sys.argv[1])
view_pyglet(out)
