from pytest import approx

from py3do import cube
from py3do import offset_mesh

def test_offset_cube():
    m = cube()
    mo, true_offsets = offset_mesh(m, 0.1, return_true_offsets=True)
    assert true_offsets == approx(0.1)

    mo, true_offsets = offset_mesh(m, -0.1, return_true_offsets=True)
    assert true_offsets == approx(-0.1)
