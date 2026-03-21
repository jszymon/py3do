import numpy as np
from pytest import approx

from py3do import circle, cube, cone_pipe

def test_cone_pipe():
    """Test cone pipes.  Currently just checks for exceptions."""
    # cylinder empty top/bottom
    cone_pipe(1, 0, 1, 1, n=10)
    # cylinder with top/bottom
    cone_pipe(1, 0, 1, 1, n=10, close_top=True, close_bottom=True)
    # need to flip normals:
    cone_pipe(1, 0, 1, 1, 2, 1, n=10, connect_top_bottom=True)
    
# test circles

def test_circle_basic():
    """Test circle function with basic parameters."""
    # Test with n=4 (square-like circle)
    points = circle(4)
    
    # Should return 4 points
    assert points.shape == (4, 2)
    
    # All points should be on unit circle (distance from origin ≈ 1)
    distances = np.linalg.norm(points, axis=1)
    assert np.allclose(distances, 1.0, atol=1e-6)
    
    # Check specific points for n=4
    expected_points = np.array([
        [1.0, 0.0],   # 0 degrees
        [0.0, 1.0],   # 90 degrees
        [-1.0, 0.0],  # 180 degrees
        [0.0, -1.0]   # 270 degrees
    ])
    assert np.allclose(points, expected_points, atol=1e-6)

def test_circle_various_n():
    """Test circle function with various n values."""
    for n in [3, 4, 5, 10, 20, 50]:
        points = circle(n)
        
        # Should return n points
        assert points.shape == (n, 2)
        
        # All points should be on unit circle
        distances = np.linalg.norm(points, axis=1)
        assert np.allclose(distances, 1.0, atol=1e-6)
        
        # First point should always be [1, 0]
        assert np.allclose(points[0], [1.0, 0.0], atol=1e-6)

def test_circle_special_values():
    """Test that circle produces exact values for special angles."""
    # Test n=4 (should have exact values at 0, 90, 180, 270 degrees)
    points = circle(4)
    assert np.allclose(points[0], [1.0, 0.0], atol=1e-10)  # Exact
    assert np.allclose(points[1], [0.0, 1.0], atol=1e-10)  # Exact
    assert np.allclose(points[2], [-1.0, 0.0], atol=1e-10)  # Exact
    assert np.allclose(points[3], [0.0, -1.0], atol=1e-10)  # Exact
    
    # Test n=8 (should have exact values at multiples of 45 degrees)
    points = circle(8)
    assert np.allclose(points[0], [1.0, 0.0], atol=1e-10)  # 0 degrees
    assert np.allclose(points[2], [0.0, 1.0], atol=1e-10)  # 90 degrees
    assert np.allclose(points[4], [-1.0, 0.0], atol=1e-10)  # 180 degrees
    assert np.allclose(points[6], [0.0, -1.0], atol=1e-10)  # 270 degrees

def test_circle_angles():
    """Test that circle points are at correct angles."""
    n = 12
    points = circle(n)
    
    # Calculate angles from points
    angles = np.arctan2(points[:, 1], points[:, 0])
    
    # Expected angles (in radians)
    expected_angles = np.arange(n) * (2 * np.pi / n)
    
    # Angles should match expected (modulo 2π)
    angle_diffs = np.abs(angles - expected_angles)
    angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)  # Handle wrap-around
    assert np.all(angle_diffs < 1e-6)

def test_circle_continuity():
    """Test that circle points form a continuous loop."""
    n = 100
    points = circle(n)
    
    # Calculate distances between consecutive points
    distances = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    
    # All distances should be approximately equal (uniform spacing)
    assert np.allclose(distances, distances[0], rtol=1e-2)
    
    # Distance between first and last point should also be similar
    # (since it's a closed loop)
    first_last_distance = np.linalg.norm(points[-1] - points[0])
    assert np.allclose(first_last_distance, distances[0], rtol=1e-2)

def test_circle_centered():
    """Test that circle is centered at origin."""
    n = 20
    points = circle(n)
    
    # Mean should be very close to [0, 0]
    mean_point = np.mean(points, axis=0)
    assert np.allclose(mean_point, [0.0, 0.0], atol=1e-6)

def test_circle_symmetry():
    """Test that circle has proper symmetry."""
    n = 8
    points = circle(n)
    
    # For even n, points should be symmetric
    # Point i should be opposite to point i + n/2
    for i in range(n // 2):
        opposite_i = i + n // 2
        assert np.allclose(points[i], -points[opposite_i], atol=1e-6)

def test_circle_edge_cases():
    """Test circle with edge case n values."""
    # Test with minimum n=3
    points = circle(3)
    assert points.shape == (3, 2)
    distances = np.linalg.norm(points, axis=1)
    assert np.allclose(distances, 1.0, atol=1e-6)
    
    # Test with larger n
    points = circle(1000)
    assert points.shape == (1000, 2)
    distances = np.linalg.norm(points, axis=1)
    assert np.allclose(distances, 1.0, atol=1e-6)
