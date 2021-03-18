import pytest

import numpy as np

from py3do.primitives import cube, cone_pipe

from py3do.topo import repeated_face_vertices
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

def test_edge_face_map():
    c = cube()
    efm = EdgeToFaceMap(c)
    assert efm.manifold
    assert efm.watertight
    assert efm.oriented
def test_edge_face_map2():
    cyl = cone_pipe(0, 0, 1, 1, 1, 1, 0, n=10)
    efm = EdgeToFaceMap(cyl)
    assert efm.manifold
    assert efm.watertight
    assert efm.oriented
def test_edge_face_map2():
    open_cyl = cone_pipe(1, 1, n=10)
    efm = EdgeToFaceMap(open_cyl)
    assert efm.manifold
    assert efm.oriented
    