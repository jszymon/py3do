import pytest

from py3do.primitives import cube

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
    
