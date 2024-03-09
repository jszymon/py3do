import pytest

import numpy as np

from py3do.primitives import cube, cone_pipe

from py3do.topo import repeated_face_vertices
from py3do.topo import unused_vertices
from py3do.topo import EdgeToFaceMap

def test_repeated_vertices_good():
    c = cube()
    rvs = repeated_face_vertices(c)
    assert len(rvs) == 0
def test_repeated_vertices_bad():
    c = cube()
    c.faces[0][1] = c.faces[0][2]
    rvs = repeated_face_vertices(c)
    assert len(rvs) == 1
    # make sure cube is not changed
    c = cube()
    rvs = repeated_face_vertices(c)
    assert len(rvs) == 0

def test_unused_vertices_good():
    c = cube()
    uvs = unused_vertices(c)
    assert len(uvs) == 0
def test_unused_vertices_bad():
    c = cube()
    c.vertices = np.vstack([c.vertices, [[1, 2, 3]]])
    uvs = unused_vertices(c)
    assert len(uvs) == 1

def test_edge_face_map():
    c = cube()
    efm = EdgeToFaceMap(c)
    assert efm.manifold
    assert efm.watertight
    assert efm.oriented
    assert len(efm.get_multiface_edges()) == 0
    assert len(efm.get_misoriented_edges()) == 0
    assert len(efm.get_boundary_edges()) == 0
def test_edge_face_map2():
    cyl = cone_pipe(0, 0, 1, 0, 1, 1, 0, 1, n=10)
    efm = EdgeToFaceMap(cyl)
    assert efm.manifold
    assert efm.watertight
    assert efm.oriented
    assert len(efm.get_multiface_edges()) == 0
    assert len(efm.get_misoriented_edges()) == 0
    assert len(efm.get_boundary_edges()) == 0
def test_edge_face_map3():
    open_cyl = cone_pipe(1, 0, 1, 1, n=10)
    efm = EdgeToFaceMap(open_cyl)
    assert efm.manifold
    assert not efm.watertight
    assert efm.oriented
    assert len(efm.get_multiface_edges()) == 0
    assert len(efm.get_misoriented_edges()) == 0
    assert len(efm.get_boundary_edges()) == 20
def test_edge_face_map4():
    c = cube()
    # add spurious face to make edge (1,3) touch 3 faces
    c.faces = np.vstack([c.faces, [[1,3,6]]])
    efm = EdgeToFaceMap(c)
    assert not efm.manifold
    assert not efm.watertight
    assert not efm.oriented
    assert len(efm.get_misoriented_edges()) > 0
    assert len(efm.get_multiface_edges()) == 1
    assert len(efm.get_boundary_edges()) > 0
    # test returned faces
    for ij in [[0,3], [(0,3)], [[0,3]]]:  # various arg types
        f = efm.find_faces(*ij)
        assert len(f) == 1
        f = f.popitem()[1]
        assert len(f) == 2
        assert 0 in f
        assert 1 in f
    f = efm.find_faces(1,3)
    assert len(f) == 1
    f = f.popitem()[1]
    assert len(f) == 3
    assert 0 in f
    assert 10 in f
    assert 12 in f
    f = efm.find_faces([[0,3], [1,3]])
    assert len(f) == 2
