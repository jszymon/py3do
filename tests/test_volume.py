from pytest import approx

from py3do import volume
from py3do import cube

def test_cube_volume():
    m = cube()
    v = volume(m)
    assert v == approx(1)
