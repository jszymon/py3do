import numpy as np
import itertools
import pytest

from py3do import cube
from py3do import Mesh
from py3do import EdgeToFaceMap
from py3do import sorted_edges
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

def face_areas(m):
    tri = m.vertices[m.faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                   tri[:, 2] - tri[:, 0]), axis=1) / 2

def test_split_k1():
    m = cube()
    m.split_edges([[0, 3]], k=1)
    check_split(m, 12, 8)

def test_split_k_positions():
    m = cube()
    new_v = m.split_edges([[3, 0]], k=4)
    assert np.array_equal(new_v, [8])
    # ordered from lower to higher vertex index
    assert np.allclose(m.vertices[8:], [[0, 0.25, 0.25],
                                        [0, 0.5, 0.5],
                                        [0, 0.75, 0.75]])
    check_split(m, 12 + 2 * 3, 11)

def test_split_k_list():
    m = cube()
    new_v = m.split_edges([[0, 3], [0, 1], [3, 0]], k=[3, 5, 3])
    assert new_v[0] == new_v[2]
    # face 0,1,3 is a 9-gon -> 7 triangles, faces 0,3,2 and 5,1,0
    # are 5- and 7-gons -> 3 and 5 triangles
    check_split(m, 12 + 6 + 2 + 4, 8 + 2 + 4)

def test_split_k_conflict():
    m = cube()
    with pytest.raises(ValueError):
        m.split_edges([[0, 3], [3, 0]], k=[2, 3])
    with pytest.raises(ValueError):
        m.split_edges([[0, 3]], k=0)

def test_split_k_all_edges():
    m = cube()
    edges = sorted_edges(m)
    m.split_edges(edges, k=np.arange(len(edges)) % 4 + 1)
    m.check_faces_and_vertices()
    etf = EdgeToFaceMap(m)
    assert etf.watertight and etf.oriented
    assert np.isclose(volume(m), 1)
    assert (face_areas(m) > 1e-9).all()
    tri = m.vertices[m.faces]
    assert np.allclose(np.einsum("ij,ikj->ik", m.normals,
                                 tri - tri[:, [0]]), 0)

def test_split_triangle_patterns():
    for ks in itertools.product(range(1, 7), repeat=3):
        m = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        m.split_edges([[0, 1], [1, 2], [2, 0]], k=ks)
        assert m.faces.shape[0] == sum(ks) - 2
        assert (face_areas(m) > 1e-9).all()
        assert np.isclose(face_areas(m).sum(), 0.5)

def test_split_grid_triangle():
    for k in range(1, 8):
        m = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        m.split_edges([[0, 1], [1, 2], [2, 0]], k=k, method="grid")
        n_vert = 3 + 3 * (k - 1) + (k - 1) * (k - 2) // 2
        assert m.vertices.shape[0] == n_vert
        assert m.faces.shape[0] == k * k
        # all triangles are congruent
        assert np.allclose(face_areas(m), 0.5 / (k * k))
        assert np.allclose(m.normals, [0, 0, 1])
        # every vertex is used
        assert len(np.unique(m.faces)) == n_vert

def test_split_grid_cube():
    m = cube()
    m.split_edges(sorted_edges(m), k=4, method="grid")
    check_split(m, 12 * 16, 8 + 18 * 3 + 12 * 3)
    assert np.allclose(face_areas(m), 0.5 / 16)

def test_split_grid_mixed_k():
    # faces with unequal k fall back to ear clipping
    m = cube()
    edges = sorted_edges(m)
    m.split_edges(edges, k=np.arange(len(edges)) % 4 + 1, method="grid")
    m.check_faces_and_vertices()
    etf = EdgeToFaceMap(m)
    assert etf.watertight and etf.oriented
    assert np.isclose(volume(m), 1)
    assert (face_areas(m) > 1e-9).all()

def test_split_bad_method():
    with pytest.raises(ValueError):
        cube().split_edges([[0, 3]], method="foo")
