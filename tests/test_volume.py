from pytest import approx

from py3do import volume
from py3do import cube, uv_sphere

def _check_volume_invariance(m):
    """Make sure volume does not change under transformations."""
    v0 = volume(m)
    m1 = m.clone()
    m1.vertices += [[10,10,10]]
    v1 = volume(m1)
    assert v1 == approx(v0)
    m2 = m.clone()
    m2.vertices -= [[10,10,10]]
    v2 = volume(m2)
    assert v2 == approx(v0)
    m3 = m.clone()
    m3.rot90("XYZ")
    v3 = volume(m3)
    assert v3 == approx(v0)

def test_cube_volume():
    m = cube()
    v = volume(m)
    assert v == approx(1)
    _check_volume_invariance(m)

def test_uv_sphere_volume():
    m = uv_sphere()
    _check_volume_invariance(m)
