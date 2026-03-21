import pytest

import numpy as np

from py3do.primitives import cube, cone_pipe, Mesh

from py3do.topo import repeated_face_vertices
from py3do.topo import unused_vertices
from py3do.topo import EdgeToFaceMap
from py3do.topo import sorted_edges

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

def test_sorted_edges_basic():
    """Test sorted_edges with a simple cube."""
    m = cube()
    edges = sorted_edges(m)
    
    # Should return edges with sorted vertex indices
    assert edges.shape[1] == 2
    assert edges.shape[0] > 0
    
    # All edges should have i < j
    assert np.all(edges[:, 0] < edges[:, 1])
    
    # Check that edges are valid vertex indices
    assert np.all(edges >= 0)
    assert np.all(edges < m.vertices.shape[0])


def test_sorted_edges_unique():
    """Test sorted_edges with unique=True (default)."""
    m = cube()
    edges = sorted_edges(m, unique=True)
    
    # Should return unique edges only
    unique_edges = np.unique(edges, axis=0)
    assert unique_edges.shape[0] == edges.shape[0]  # All edges should be unique
    
    # For a cube, we should have 18 unique edges (including face diagonals)
    assert edges.shape[0] == 18


def test_sorted_edges_non_unique():
    """Test sorted_edges with unique=False."""
    m = cube()
    edges = sorted_edges(m, unique=False)
    
    # Should return all edges (including duplicates from different faces)
    # A cube has 12 faces * 3 edges per face = 36 edges, but many are shared
    # Actually, each face has 3 edges, and each edge is shared by 2 faces
    # So we should have 12 faces * 3 edges = 36 edge references
    assert edges.shape[0] == m.faces.shape[0] * 3  # 3 edges per face
    
    # All edges should still be sorted (i < j)
    assert np.all(edges[:, 0] <= edges[:, 1])


def test_sorted_edges_return_order():
    """Test sorted_edges with return_order=True."""
    m = cube()
    edges, order = sorted_edges(m, return_order=True, unique=False)
    
    # Should return edges and order array
    assert edges.shape[1] == 2
    assert edges.shape[0] == order.shape[0]  # Order array should match number of edges
    assert order.dtype == bool
    
    # Check that order indicates which edges were originally unsorted
    # For a cube, some edges will be in order, some will need swapping
    assert np.any(order)  # Some edges should be True
    assert np.any(~order)  # Some edges should be False


def test_sorted_edges_cylinder():
    """Test sorted_edges with a cylinder."""
    m = cone_pipe(1, 0, 1, 1, n=10, close_bottom=True, close_top=True)
    edges = sorted_edges(m, unique=True)
    
    # Should return sorted unique edges
    assert edges.shape[1] == 2
    assert np.all(edges[:, 0] < edges[:, 1])
    
    # Check edge validity
    assert np.all(edges >= 0)
    assert np.all(edges < m.vertices.shape[0])


def test_sorted_edges_custom_mesh():
    """Test sorted_edges with a custom mesh."""
    # Create a simple tetrahedron
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    
    m = Mesh(vertices, faces)
    edges = sorted_edges(m, unique=True)
    
    # Should have 6 unique edges for a tetrahedron
    assert edges.shape[0] == 6
    assert np.all(edges[:, 0] < edges[:, 1])
    
    # Check that all expected edges are present
    expected_edges = np.array([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
    ])
    # Sort both for comparison
    edges_sorted = np.sort(edges, axis=0)
    expected_sorted = np.sort(expected_edges, axis=0)
    
    # They should be the same (allowing for different ordering)
    assert np.array_equal(np.sort(edges_sorted, axis=0), 
                          np.sort(expected_sorted, axis=0))


def test_sorted_edges_invalid_combination():
    """Test that return_order=True and unique=True cannot be used together."""
    m = cube()
    with pytest.raises(AssertionError):  # The function uses assert, not ValueError
        sorted_edges(m, return_order=True, unique=True)

def test_sorted_edges_edge_cases():
    """Test sorted_edges with edge cases."""
    # Test with a single triangle
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    m = Mesh(vertices, faces)
    
    edges = sorted_edges(m, unique=True)
    assert edges.shape[0] == 3  # 3 edges in a triangle
    assert np.all(edges[:, 0] < edges[:, 1])
    
    # Test with a mesh that has some edges already sorted
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 1, 3]])  # Some edges already sorted
    m = Mesh(vertices, faces)
    
    edges, order = sorted_edges(m, return_order=True, unique=False)
    assert edges.shape[0] == 6  # 2 faces * 3 edges each
    assert order.shape[0] == 6
