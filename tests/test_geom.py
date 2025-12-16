"""Tests for geometric algorithms."""

def test_normals_cross():
    """Test normals_cross function with various mesh types."""
    from py3do import cube, normals_cross
    import numpy as np
    from pytest import approx

    m = cube()
    normals, areas = normals_cross(m)

    # Test that we get the right number of normals (should match number of faces)
    assert normals.shape == (m.faces.shape[0], 3)

    # Test that areas are positive
    assert np.all(areas > 0)

    # Test that normals are unit vectors (length ≈ 1)
    norm_lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(norm_lengths, 1.0, atol=1e-6)

    # Test that all normals are axis-aligned (should be ±1 in one component, 0 in others)
    for normal in normals:
        # Count non-zero components (should be exactly 1)
        non_zero_count = np.count_nonzero(np.abs(normal) > 1e-6)
        assert non_zero_count == 1
        
        # The non-zero component should be ±1
        max_component = np.max(np.abs(normal))
        assert np.isclose(max_component, 1.0, atol=1e-6)


def test_normals_cross_triangle():
    """Test normals_cross with a simple triangle mesh."""
    from py3do import Mesh, normals_cross
    import numpy as np
    from pytest import approx

    # Create a simple triangle mesh
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0], 
        [0, 1, 0]
    ])
    faces = np.array([[0, 1, 2]])
    
    m = Mesh(vertices, faces)
    normals, areas = normals_cross(m)

    # Should have one normal for one face
    assert normals.shape == (1, 3)
    assert areas.shape == (1,)

    # The normal should be [0, 0, 1] (pointing in z-direction)
    expected_normal = np.array([0, 0, 1])
    assert np.allclose(normals[0], expected_normal, atol=1e-6)

    # Area should be 0.5 (area of right triangle with legs of length 1)
    assert areas[0] == approx(0.5, abs=1e-6)


def test_normals_cross_orientation():
    """Test that normals_cross produces consistent orientation."""
    from py3do import cube, normals_cross
    import numpy as np

    m = cube()
    normals, areas = normals_cross(m)

    # For a cube, opposite faces should have opposite normals
    # Let's check some pairs of opposite faces based on actual cube structure
    # Face 0 (x-negative) is opposite to Face 2 (x-positive)
    dot_product = np.dot(normals[0], normals[2])
    assert np.isclose(dot_product, -1.0, atol=1e-6)

    # Face 4 (y-negative) is opposite to Face 6 (y-positive)
    dot_product = np.dot(normals[4], normals[6])
    assert np.isclose(dot_product, -1.0, atol=1e-6)

    # Face 8 (z-negative) is opposite to Face 10 (z-positive)
    dot_product = np.dot(normals[8], normals[10])
    assert np.isclose(dot_product, -1.0, atol=1e-6)


def test_normals_cross_area_calculation():
    """Test that area calculation is correct."""
    from py3do import cube, normals_cross
    import numpy as np
    from pytest import approx

    m = cube()
    normals, areas = normals_cross(m)

    # All faces of a unit cube should have area = 0.5 (triangular faces)
    assert np.allclose(areas, 0.5, atol=1e-6)

    # Total surface area should be 6.0 (12 triangular faces * 0.5 area each = 6.0)
    total_area = np.sum(areas)
    assert total_area == approx(6.0, abs=1e-6)


def test_normals_cross_complex_mesh():
    """Test normals_cross with a more complex mesh."""
    from py3do import Mesh, normals_cross
    import numpy as np
    from pytest import approx

    # Create a tetrahedron mesh
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0], 
        [0, 1, 0],
        [0, 0, 1]
    ])
    faces = np.array([
        [0, 1, 2],  # bottom face
        [0, 1, 3],  # front face
        [0, 2, 3],  # left face  
        [1, 2, 3]   # back face
    ])
    
    m = Mesh(vertices, faces)
    normals, areas = normals_cross(m)

    # Should have 4 normals for 4 faces
    assert normals.shape == (4, 3)
    assert areas.shape == (4,)

    # All normals should be unit vectors
    norm_lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(norm_lengths, 1.0, atol=1e-6)

    # All areas should be positive
    assert np.all(areas > 0)

    # Test specific face normals
    # Bottom face (0,1,2) should have normal pointing in positive z-direction
    expected_bottom_normal = np.array([0, 0, 1])
    dot_product = np.dot(normals[0], expected_bottom_normal)
    assert np.abs(dot_product) == approx(1.0, abs=1e-6)

    # Front face (0,1,3) should have normal pointing in positive y-direction
    expected_front_normal = np.array([0, 1, 0])
    dot_product = np.dot(normals[1], expected_front_normal)
    assert np.abs(dot_product) == approx(1.0, abs=1e-6)