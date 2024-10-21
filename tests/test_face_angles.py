import numpy as np

from pytest import approx

from py3do import cube
from py3do import face_angles

def test_cube_angles():
    m = cube()
    a = face_angles(m)
    assert np.testing.assert_allclose(a, np.pi/2)

