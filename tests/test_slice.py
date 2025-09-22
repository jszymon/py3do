import numpy as np

from py3do.primitives import cube, uv_sphere
from py3do.slice import slice_horiz_0
from py3do import is_isomorphic
from py3do.topo import EdgeToFaceMap

def _assert_verts_unique(ms):
    assert np.unique(ms.vertices, axis=0).shape == ms.vertices.shape
def _assert_faces_on_one_side(ms):
    vert_sgn = np.sign(ms.vertices[:,2])
    # assert all faces have all nonnegative and/or nonpositive vertices
    assert ((vert_sgn[ms.faces] > -0.5).all(axis=1) |
            (vert_sgn[ms.faces] < 0.5).all(axis=1)).all()
def _assert_correct_topo(ms, closed=True):
    efm = EdgeToFaceMap(ms)
    assert efm.oriented, "Edges not correctly oriented"
    assert efm.manifold, "Model is not manifold"
    if closed:
        assert efm.watertight, "Model is not watertight"

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
    _assert_verts_unique(ms)
    _assert_faces_on_one_side(ms)
    _assert_correct_topo(ms, closed=True)
def test_horiz_0_cut_cube_positive_open():
    m = cube()
    m.vertices[:,2] -= 0.25
    ms = slice_horiz_0(m, keep="positive", fill=False)
    assert len(ms.vertices) == 8 + 4 # top/bottom + new on bottom
    assert len(ms.faces) == 2 + 4*3  # top + 4 sides*3 triangles + bot 6 triangles
    _assert_verts_unique(ms)
    _assert_faces_on_one_side(ms)
    _assert_correct_topo(ms, closed=False)
def test_horiz_0_cut_cube_positive():
    m = cube()
    m.vertices[:,2] -= 0.25
    ms = slice_horiz_0(m, keep="positive")
    assert len(ms.vertices) == 8 + 4 # top/bottom + new on bottom
    assert len(ms.faces) == 2 + 4*3 + 6 # top + 4 sides*3 triangles + bot 6 triangles
    _assert_verts_unique(ms)
    _assert_faces_on_one_side(ms)
    _assert_correct_topo(ms, closed=True)
def test_horiz_0_cut_cube_negative():
    m = cube()
    m.vertices[:,2] -= 0.25
    ms = slice_horiz_0(m, keep="negative")
    assert len(ms.vertices) == 8 + 4 # top/bottom + new on bottom
    assert len(ms.faces) == 2 + 4*3 + 6 # top + 4 sides*3 triangles + bot 6 triangles
    _assert_verts_unique(ms)
    _assert_faces_on_one_side(ms)
    _assert_correct_topo(ms, closed=True)
def test_horiz_0_cut_cube_pos():
    m = cube()
    m.vertices[:,2] -= 0.25
    ms = slice_horiz_0(m, keep="positive")
    #assert len(ms.vertices) == 8 + 4 + 4 # top/bottom + new
    #assert len(ms.faces) == 4 + 4*6 # top/bottom + 4 sides * 6 triangles
    _assert_verts_unique(ms)
    _assert_faces_on_one_side(ms)

def test_horiz_0_cut_sphere():
    for dz in np.linspace(-1,1,20):
        m = uv_sphere(n_u=10)
        m.vertices[:,2] -= dz
        ms = slice_horiz_0(m, keep="positive")
        _assert_verts_unique(ms)
        _assert_faces_on_one_side(ms)
        _assert_correct_topo(ms, closed=True)
        ms = slice_horiz_0(m, keep="negative")
        _assert_verts_unique(ms)
        _assert_faces_on_one_side(ms)
        _assert_correct_topo(ms, closed=True)
        ms = slice_horiz_0(m, keep="both")
        _assert_verts_unique(ms)
        _assert_faces_on_one_side(ms)
        _assert_correct_topo(ms, closed=False)
