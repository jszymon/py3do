import numpy as np

from py3do.primitives import cube, uv_sphere
from py3do.slice import slice_horiz_0
from py3do.vis import view_pyglet
from py3do import is_isomorphic

if __name__ == "__main__":
    m=cube()
    ms = slice_horiz_0(m)
    assert is_isomorphic(m, ms) # full model on positive side

    m2 = m.clone()
    m2.vertices[:,2] += 1
    m2s = slice_horiz_0(m2, keep="negative")
    assert len(m2s.vertices) == 0
    assert len(m2s.faces) == 0

    m3 = m.clone()
    m3.vertices[:,2] -= 0.25
    m3s = slice_horiz_0(m3, keep="both")
    view_pyglet(m3s, marked_vertices = np.arange(len(m3s.vertices)).tolist())

    m4 = uv_sphere(n_u=10)
    m4.vertices[:,2] -= 0.33
    m4s = slice_horiz_0(m4, keep="negative")
    view_pyglet(m4s, marked_vertices = np.arange(len(m4s.vertices)).tolist())
