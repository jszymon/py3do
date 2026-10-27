import numpy as np

from py3do import cube
from py3do import Mesh
from py3do import EdgeToFaceMap
from py3do import volume
from py3do import connected_components
from py3do import is_isomorphic

def test_merge_two_cubes():
    m = cube()
    m2 = cube()
    m2.vertices += [2, 0, 0]
    n_v, n_f = m.vertices.shape[0], m.faces.shape[0]
    offset = m.merge(m2)
    assert offset == n_v
    assert m.vertices.shape[0] == 2 * n_v
    assert m.faces.shape[0] == 2 * n_f
    assert m.normals.shape[0] == 2 * n_f
    m.check_faces_and_vertices()
    assert np.array_equal(m.vertices[offset:], m2.vertices)
    assert np.array_equal(m.faces[n_f:], m2.faces + offset)
    assert np.array_equal(m.normals[n_f:], m2.normals)
    etf = EdgeToFaceMap(m)
    assert etf.watertight
    assert etf.oriented
    assert np.isclose(volume(m), 2)
    assert connected_components(m)[0] == 2

def test_merge_overlapping_no_union():
    m = cube()
    m.merge(cube())
    assert m.vertices.shape[0] == 2 * cube().vertices.shape[0]
    assert connected_components(m)[0] == 2
    assert np.isclose(volume(m), 2)

def test_merge_empty():
    m = cube()
    m_orig = m.clone()
    assert m.merge(Mesh([], [])) == m.vertices.shape[0]
    assert is_isomorphic(m, m_orig)
    e = Mesh([], [])
    assert e.merge(cube()) == 0
    e.check_faces_and_vertices()
    assert is_isomorphic(e, cube())
