import numpy as np

from py3do.primitives import cube, uv_sphere
from py3do.slice import slice_horiz_0
from py3do import is_isomorphic

def test_horiz_0_full_positive():
    m=cube()
    ms = slice_horiz_0(m)
    assert is_isomorphic(m, ms) # full model on positive side
    
def test_horiz_0_full_negative():
    # full model on negative side
    # return empty model
    m=cube()
    m.vertices[:,2] += 1
    ms = slice_horiz_0(m, keep="negative")
    assert len(ms.vertices) == 0
    assert len(ms.faces) == 0
    
def test_horiz_0_cut_cube():
    m = cube()
    m.vertices[:,2] -= 0.25
    ms = slice_horiz_0(m, keep="both")
    assert len(ms.vertices) == 8 + 4 + 4 # top/bottom + new
    assert len(ms.faces) == 4 + 4*6 # top/bottom + 4 sides * 6 triangles
    

if __name__ == "__main__":
    from py3do.vis import view_pyglet
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

    m5 = uv_sphere(n_u=10)
    m5.vertices[:,2] *= -1
    m5.vertices[:,2] -= 0.33
    m5s = slice_horiz_0(m5, keep="negative")
    view_pyglet(m5s, marked_vertices = np.arange(len(m5s.vertices)).tolist())
