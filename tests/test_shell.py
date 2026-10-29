"""Tests for missing shell functions."""

import numpy as np
import pytest
from pytest import approx

from py3do import cube, uv_sphere
from py3do.shell import check_offset


def test_check_offset_basic():
    """Test check_offset with basic displacement."""
    m = cube()
    
    # Create a simple displacement (move all vertices by [0.1, 0.1, 0.1])
    v_disp = np.full_like(m.vertices, 0.1)
    
    true_offsets = check_offset(m, v_disp)
    
    # Should return a matrix with one offset per vertex per face
    assert true_offsets.shape == (m.faces.shape[0], 3)
    
    # For a cube with uniform displacement, offsets depend on face normals
    # Faces with normals in same direction as displacement will have positive offsets
    # Faces with normals opposite to displacement will have negative offsets
    # Since all vertices have the same displacement, all vertices on a face have the same offset
    for i in range(m.faces.shape[0]):
        expected_offset = np.dot(v_disp[0], m.normals[i])
        assert np.allclose(true_offsets[i], expected_offset, atol=1e-6)


def test_check_offset_zero():
    """Test check_offset with zero displacement."""
    m = cube()
    
    # Zero displacement
    v_disp = np.zeros_like(m.vertices)
    
    true_offsets = check_offset(m, v_disp)
    
    # All offsets should be zero
    assert true_offsets.shape == (m.faces.shape[0], 3)
    assert np.allclose(true_offsets, 0.0, atol=1e-6)


def test_check_offset_normal_aligned():
    """Test check_offset with displacement aligned to normals."""
    m = cube()
    
    # Displace each vertex by its normal direction
    v_disp = m.normals * 0.1  # Small offset in normal direction
    
    true_offsets = check_offset(m, v_disp)
    
    # Should return proper offset matrix
    assert true_offsets.shape == (m.faces.shape[0], 3)
    
    # For displacement along normals, the offset for each vertex on a face
    # is the dot product of that vertex's displacement with the face normal
    # This can be positive or negative depending on the angle between
    # the vertex normal and face normal
    assert np.all(np.isfinite(true_offsets))


def test_check_offset_sphere():
    """Test check_offset with a sphere."""
    m = uv_sphere(n_u=10, n_v=10)
    
    # Create radial displacement
    v_disp = m.vertices * 0.1  # Move each vertex radially outward
    
    true_offsets = check_offset(m, v_disp)
    
    # Should return proper offset matrix
    assert true_offsets.shape == (m.faces.shape[0], 3)
    assert np.all(np.isfinite(true_offsets))


def test_check_offset_negative():
    """Test check_offset with negative displacement."""
    m = cube()
    
    # Negative displacement (move inward)
    v_disp = np.full_like(m.vertices, -0.1)
    
    true_offsets = check_offset(m, v_disp)
    
    # Should return proper offset matrix
    assert true_offsets.shape == (m.faces.shape[0], 3)
    
    # For negative displacement, offsets depend on face normals
    # Some will be positive, some negative depending on normal direction
    # Since all vertices have the same displacement, all vertices on a face have the same offset
    for i in range(m.faces.shape[0]):
        expected_offset = np.dot(v_disp[0], m.normals[i])
        assert np.allclose(true_offsets[i], expected_offset, atol=1e-6)


def test_check_offset_varying():
    """Test check_offset with varying displacement."""
    m = cube()
    
    # Create displacement that varies by vertex
    v_disp = np.zeros_like(m.vertices)
    v_disp[:, 0] = np.linspace(0, 0.2, m.vertices.shape[0])  # Vary x displacement
    
    true_offsets = check_offset(m, v_disp)
    
    # Should return proper offset matrix
    assert true_offsets.shape == (m.faces.shape[0], 3)
    
    # Check that offsets vary appropriately
    assert not np.allclose(true_offsets, true_offsets[0, 0])  # Should not all be the same


def test_check_offset_consistency():
    """Test that check_offset is consistent with manual calculation."""
    m = cube()
    
    # Create a simple displacement
    v_disp = np.array([[0.1, 0.0, 0.0]] * m.vertices.shape[0])
    
    true_offsets = check_offset(m, v_disp)
    
    # Manually calculate expected offsets
    # Offset = dot product of displacement with face normal
    expected_offsets = np.zeros((m.faces.shape[0], 3))
    for i in range(m.faces.shape[0]):
        for j in range(3):
            vertex_idx = m.faces[i, j]
            displacement = v_disp[vertex_idx]
            face_normal = m.normals[i]
            expected_offsets[i, j] = np.dot(displacement, face_normal)
    
    # Should match
    assert np.allclose(true_offsets, expected_offsets, atol=1e-6)

from py3do import Mesh, EdgeToFaceMap, volume
from py3do.shell import shell, _boundary_edges


def _square():
    return Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0, 1, 2], [0, 2, 3]])


def _cap():
    """Upper part of a sphere, an open curved surface."""
    s = uv_sphere(n_u=16, n_v=8)
    m, _ = s.get_submesh(s.vertices[:, 2] > 0.3)
    return m


def _check_closed(m):
    em = EdgeToFaceMap(m)
    assert em.watertight
    assert em.oriented


@pytest.mark.parametrize("edge", ["flat", "inset", "round"])
@pytest.mark.parametrize("d", [0.1, -0.1])
def test_shell_square(edge, d):
    s = shell(_square(), d, edge=edge, n_round=6)
    _check_closed(s)
    v = volume(s)
    if edge == "round":
        r = 0.05
        assert 0.1 < v < 0.1 + 4 * np.pi * r**2 / 2
        # corners bulge along the bisector by the profile radius
        assert s.vertices[:, 0].min() == approx(-r / np.sqrt(2))
    else:
        assert v == approx(0.1)
        assert s.vertices[:, 0].min() == approx(0)
    assert np.sort(s.vertices[:, 2])[[0, -1]] == approx(sorted([0, d]))


@pytest.mark.parametrize("edge", ["flat", "inset", "round"])
@pytest.mark.parametrize("d", [0.1, -0.1])
def test_shell_cap(edge, d):
    m = _cap()
    s = shell(m, d, edge=edge, n_round=4)
    _check_closed(s)
    assert volume(s) > 0
    # displacement of boundary vertices along outward directions
    # perpendicular to boundary edges in adjacent faces
    n = m.vertices.shape[0]
    a, b, f = _boundary_edges(m)
    eo = np.cross(m.vertices[b] - m.vertices[a], m.normals[f])
    eo /= np.linalg.norm(eo, axis=1, keepdims=True)
    protrusion = np.concatenate(
        [np.einsum("ij,ij->i", s.vertices[v + n] - m.vertices[v], eo)
         for v in (a, b)])
    if edge == "flat":
        # vertex normals at the boundary of a cap tilt inwards
        assert (protrusion > 1e-3).any() == (d < 0)
    else:
        assert (protrusion <= 1e-9).all()


def test_shell_bad_args():
    with pytest.raises(ValueError):
        shell(_square(), 0.1, edge="bad")
    with pytest.raises(ValueError):
        shell(_square(), 0)
