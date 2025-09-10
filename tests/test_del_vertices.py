import numpy as np

from py3do import cube
from py3do import is_isomorphic

def test_del_no_vertices():
    m = cube()
    m_orig = m.clone()
    m.delete_vertices(np.array([], dtype=int))
    assert is_isomorphic(m, m.clone())

def test_del_all_vertices():
    m = cube()
    m.delete_vertices(np.arange(m.vertices.shape[0]))
    assert m.vertices.shape[0] == 0
    assert m.normals.shape[0] == 0
    assert m.faces.shape[0] == 0

def test_del_vertices():
    m = cube()
    n = m.vertices.shape[0]
    m.delete_vertices([0])
    assert m.vertices.shape[0] == n-1
    assert np.isin(m.faces, np.arange(n-1)).all()
    m.delete_vertices([3,5])
    assert m.vertices.shape[0] == n-3
    assert np.isin(m.faces, np.arange(n-3)).all()
