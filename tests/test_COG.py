import numpy as np

from py3do import COG
from py3do import cube, uv_sphere

def test_cube_COG():
    m = cube()
    c = COG(m)
    assert np.allclose(c, 0.5)

def test_uv_sphere_COG():
    m = uv_sphere()
    c = COG(m)
    assert np.allclose(c, 0)
