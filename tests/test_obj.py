import io

import pytest

from py3do.io import read_obj
from py3do import is_isomorphic


def test_simple_triangle():
    f = io.StringIO("""# Simple triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
""")
    mesh = read_obj(f)
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
    assert mesh.normals.shape == (1, 3)


def test_two_triangles():
    f = io.StringIO("""# Two triangles sharing an edge
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
f 1 2 3
f 1 2 4
""")
    mesh = read_obj(f)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)
    assert mesh.normals.shape == (2, 3)


def test_with_comments():
    f = io.StringIO("""# This is a comment
# Another comment
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
# Face comment
f 1 2 3
""")
    mesh = read_obj(f)
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)


def test_face_with_texture_coords():
    f = io.StringIO("""# Face with texture coordinates (should be ignored)
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1/1/1 2/2/2 3/3/3
""")
    mesh = read_obj(f)
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
    # Should extract only vertex indices
    assert (mesh.faces == [[0, 1, 2]]).all()


def test_empty_file():
    f = io.StringIO("")
    mesh = read_obj(f)
    assert mesh.vertices.shape == (0, 3)
    assert mesh.faces.shape == (0, 3)


def test_no_vertices_no_faces():
    f = io.StringIO("# Just comments\n# No vertices\n# No faces")
    # Should create empty mesh
    mesh = read_obj(f)
    assert mesh.vertices.shape == (0, 3)
    assert mesh.faces.shape == (0, 3)


def test_no_faces():
    f = io.StringIO("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0")
    # Should create mesh with vertices but no faces
    mesh = read_obj(f)
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (0, 3)


def test_3d_coordinates():
    f = io.StringIO("""# 3D coordinates
v 1.0 2.0 3.0
v 4.0 5.0 6.0
v 7.0 2.0 9.0
f 1 2 3
""")
    mesh = read_obj(f)
    expected_vertices = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 2.0, 9.0]]
    assert (mesh.vertices == expected_vertices).all()


def test_negative_coordinates():
    f = io.StringIO("""# Negative coordinates
v -1.0 -2.0 -3.0
v -4.0 -5.0 -6.0
v -7.0 -2.0 -9.0
f 1 2 3
""")
    mesh = read_obj(f)
    expected_vertices = [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -2.0, -9.0]]
    assert (mesh.vertices == expected_vertices).all()