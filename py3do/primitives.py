"""Primitive solids."""

import numpy as np

from .mesh import Mesh
from .geom import normals_cross

_cube_vertices = np.array(list(np.ndindex(2,2,2)))
_cube_faces = np.array([[0, 1, 3], [0, 3, 2],
                        [7, 5, 4], [6, 7, 4],
                        [5, 1, 0], [4, 5, 0],
                        [2, 3, 7], [2, 7, 6],
                        [0, 2, 6], [0, 6, 4],
                        [7, 3, 1], [5, 7, 1],
                       ])
_cube_normals = np.array([[-1.0, 0, 0], [-1.0, 0, 0],
                          [+1.0, 0, 0], [+1.0, 0, 0],
                          [0, -1.0, 0], [0, -1.0, 0],
                          [0, +1.0, 0], [0, +1.0, 0],
                          [0, 0, -1.0], [0, 0, -1.0],
                          [0, 0, +1.0], [0, 0, +1.0],
                         ])

def cube():
    """A unit cube."""
    c = Mesh(_cube_vertices.copy(),
                _cube_faces.copy(),
                normals=_cube_normals.copy())
    return c

def circle(n):
    """Return coordinates of points on a unit circle."""
    i = np.arange(n)
    angles = i * 2 * (np.pi / n)
    c = np.column_stack([np.cos(angles), np.sin(angles)])
    # make special values exact
    c[i == 0] = [1.0, 0.0]
    c[4*i == n] = [0.0, 1.0]  # 1/2 pi
    c[2*i == n] = [-1.0, 0.0]  # pi
    c[4*i == 3*n] = [0.0, -1.0]  # 3/2 pi
    return c

def cylinder_faces(bottom_idxs, top_idxs):
    """Heleper function to create faces between two rings of vertices
    given by their indices.

    Rings are assumed to be ordered.  Connections are between
    consecutive pairs."""
    nb = len(bottom_idxs)
    nt = len(top_idxs)
    if nt == nb == 1:  # no edges, degenerate cylinder
        return np.empty((0, 3), dtype=int)
    if nb == 1 and nt > 1:
        bi = bottom_idxs[0]
        f = [(bi, top_idxs[(j+1) % nt], top_idxs[j]) for j in range(nt)]
    elif nb > 1 and nt == 1:
        ti = top_idxs[0]
        f = [(ti, bottom_idxs[j], bottom_idxs[(j+1) % nb]) for j in range(nb)]
    elif nb == nt:
        f1 = [(top_idxs[(j+1) % nt], bottom_idxs[j], bottom_idxs[(j+1) % nb])
                                  for j in range(nt)]
        f2 = [(top_idxs[j], bottom_idxs[j], top_idxs[(j+1) % nt])
                                  for j in range(nt)]
        f = f1 + f2
    return np.array(f, dtype=int)

def cone_pipe(*args, n=128, close_bottom=False, close_top=False,
                  connect_top_bottom=False):
    """Make a circular shape based on r_1, h_1, r_2, h_2, .... sequence.

    h_1 is assumed to be 0."""
    def make_cone_segment(r, h):
        nonlocal v_idx, vs, fcs, c
        if r == 0:
            new_v = np.array([[0, 0, h]])
        else:
            new_v = c * r
            new_v[:,2] = h
        if len(vs) > 0:
            new_fs = cylinder_faces(range(v_idx - len(vs[-1]), v_idx),
                                        range(v_idx, v_idx + len(new_v)))
            fcs.append(new_fs)
        vs.append(new_v)
        v_idx = v_idx + len(new_v)

    args = list(args)
    if close_bottom and args[0] != 0:
        args = [0, 0] + args
    c = np.hstack([circle(n), np.zeros((n, 1))])
    vs = []
    v_idx = 0
    fcs = []
    for i, rh in enumerate(args):
        if i % 2 == 0:
            ri = rh
        else:
            hi = rh
            make_cone_segment(ri, hi)
    if len(args) > 0 and i % 2 == 0:
        raise RuntimeError("odd number of cone pipe coordinates")
    if len(args) > 0 and close_top and ri != 0:
        make_cone_segment(0, hi)
    if connect_top_bottom:
        new_fs = cylinder_faces(range(v_idx - len(vs[-1]), v_idx),
                                    range(len(vs[0])))
        fcs.append(new_fs)
        
    v = np.vstack([np.empty((0, 3), dtype=np.double)] + vs)
    f = np.vstack([np.empty((0, 3), dtype=int)] + fcs)
    m = Mesh(v, f)
    m.normals, _ = normals_cross(m)
    return m

def uv_sphere(n_u=128, n_v=None):
    """An uv-sphere.

    u is the horizontal coordinate."""
    if n_v is None:
        n_v = n_u
    angles = np.linspace(-np.pi/2, np.pi/2, n_v+2)[1:-1]
    c = np.column_stack([np.cos(angles), np.sin(angles)]).ravel()
    c = np.concatenate([(0,-1), c, (0,1)])
    return cone_pipe(*c, close_bottom=True, close_top=True, n=n_u)
