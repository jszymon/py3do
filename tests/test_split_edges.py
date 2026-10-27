import numpy as np

from py3do import cube
from py3do import Mesh
from py3do import EdgeToFaceMap
from py3do import volume

def check_split(m, n_faces, n_vertices):
    assert m.faces.shape[0] == n_faces
    assert m.normals.shape[0] == n_faces
    assert m.vertices.shape[0] == n_vertices
    m.check_faces_and_vertices()
    etf = EdgeToFaceMap(m)
    assert etf.watertight
    assert etf.oriented
    assert np.isclose(volume(m), 1)

def test_split_no_edges():
    m = cube()
    new_v = m.split_edges([])
    assert new_v.shape[0] == 0
    check_split(m, 12, 8)

def test_split_one_edge():
    m = cube()
    new_v = m.split_edges([[0, 3]])  # face diagonal
    assert np.array_equal(new_v, [8])
    assert np.allclose(m.vertices[8], [0, 0.5, 0.5])
    check_split(m, 14, 9)

def test_split_repeated_edges():
    m = cube()
    new_v = m.split_edges(np.array([[0, 3], [3, 0], [0, 1]]))
    assert new_v[0] == new_v[1]
    assert new_v[0] != new_v[2]
    # edge 0-1 has two faces, so does 0-3, face 0,1,3 is split into 3
    check_split(m, 16, 10)

def test_split_all_edges():
    m = cube()
    edges = m.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    m.split_edges(edges)
    check_split(m, 48, 8 + 18)
    # all subtriangles are coplanar with their parents
    tri = m.vertices[m.faces]
    assert np.allclose(np.einsum("ij,ikj->ik", m.normals,
                                 tri - tri[:, [0]]), 0)

def test_split_triangle():
    m = Mesh([[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    m.split_edges([[1, 2], [2, 0]])
    assert m.faces.shape[0] == 3
    assert np.allclose(m.normals, [0, 0, 1])
