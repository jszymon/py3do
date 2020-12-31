"""Primitive solids."""

import numpy as np

from .mesh import Mesh
from .geom import normals_cross

_cube_vertices = np.array(list(np.ndindex(2,2,2)))
_cube_faces = np.array([[0, 1, 3], [0, 2, 3],
                        [4, 5, 7], [4, 6, 7],
                        [0, 1, 5], [0, 4, 5],
                        [2, 3, 7], [2, 6, 7],
                        [0, 2, 6], [0, 4, 6],
                        [1, 3, 7], [1, 5, 7],
                       ])
_cube_normals = np.array([[-0.5, 0, 0], [-0.5, 0, 0],
                          [+0.5, 0, 0], [+0.5, 0, 0],
                          [0, -0.5, 0], [0, -0.5, 0],
                          [0, +0.5, 0], [0, +0.5, 0],
                          [0, 0, -0.5], [0, 0, -0.5],
                          [0, 0, +0.5], [0, 0, +0.5],
                         ])

def cube():
    """A unit cube."""
    c = Mesh(_cube_vertices, _cube_faces, normals=_cube_normals)
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

def cone_pipe(*args, n=10):
    """r_1, h_1, r2, .... sequence."""
    def make_cone_section(r, h):
        nonlocal v_idx, vs, fcs, c
        if r == 0:
            new_v = np.array([[0, 0, h]])
        else:
            new_v = c * r
            new_v[:,2] = h
        if len(vs) > 0:
            prev_v = vs[-1]
            nnv = len(new_v)
            npv = len(prev_v)
            if npv == nnv == 1:
                pass # no edges, degenerate segment
            else:
                if len(prev_v) == 1 and len(new_v) > 1:
                    new_fs = [(v_idx - 1, v_idx + (j + 1) % nnv, v_idx + j)
                                  for j in range(nnv)]
                elif len(prev_v) > 1 and len(new_v) == 1:
                    new_fs = [(v_idx, v_idx - npv + j, v_idx - npv + (j + 1) % npv)
                                  for j in range(npv)]
                else:
                    assert nnv == npv
                    new_fs1 = [(v_idx + (j + 1) % nnv, v_idx - npv + j, v_idx - npv + (j + 1) % npv)
                                  for j in range(nnv)]
                    new_fs2 = [(v_idx + j, v_idx - npv + j, v_idx + (j + 1) % nnv)
                                  for j in range(nnv)]
                    new_fs = new_fs1 + new_fs2
                fcs.append(np.array(new_fs))
        vs.append(new_v)
        v_idx = v_idx + len(new_v)
            

    c = np.hstack([circle(n), np.zeros((n, 1))])
    vs = []
    v_idx = 0
    fcs = []
    hi = 0.0
    for i, rh in enumerate(args):
        if i % 2 == 0:
            make_cone_section(rh, hi)
            prev_r = rh
        else:
            hi = rh
    if len(args) > 0 and i % 2 == 1:
        print(prev_r, hi)
        make_cone_section(prev_r, hi)
    v = np.vstack([np.empty((0, 3), dtype=np.float)] + vs)
    f = np.vstack([np.empty((0, 3), dtype=np.uint)] + fcs)
    m = Mesh(v, f)
    m.normals, _ = normals_cross(m)
    return m
