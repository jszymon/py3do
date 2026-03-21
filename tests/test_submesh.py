from pytest import approx

from py3do import cube
from py3do import volume


def test_submesh_cube():
    m = cube()
    mask = m.vertices[:,2] < 0.5
    subm, map_submesh_mesh = m.get_submesh(mask)
    subm.vertices[:,2] -= 1
    m.set_submesh_vertices(subm, map_submesh_mesh)
    assert volume(m) == approx(2)
