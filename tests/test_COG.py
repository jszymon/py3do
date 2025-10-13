import numpy as np

from py3do import COG
from py3do import cube, uv_sphere

def _check_COG_invariance(m):
    """Make sure COG changes correctly under transformations."""
    v0 = COG(m)
    m1 = m.clone()
    m1.vertices += [[10,10,10]]
    v1 = COG(m1)
    assert np.allclose(v1, v0+10)
    m2 = m.clone()
    m2.vertices -= [[10,10,10]]
    v2 = COG(m2)
    assert np.allclose(v2, v0-10)
    m3 = m.clone()
    _v = COG(m3)
    m3.vertices -= _v
    v3_0 = COG(m3)
    m3.rot90("XYZ")
    v3 = COG(m3)
    assert np.allclose(v3, v3_0)

def test_cube_COG():
    m = cube()
    c = COG(m)
    assert np.allclose(c, 0.5)
    _check_COG_invariance(m)

def test_uv_sphere_COG():
    m = uv_sphere()
    c = COG(m)
    assert np.allclose(c, 0)
    _check_COG_invariance(m)
