import numpy as np

from py3do import Mesh
from py3do import cube
from py3do import connected_components
from py3do import split_mesh
from py3do import is_isomorphic

def _make_repeated_cube(n):
    cubes = [cube() for i in range(n)]
    nf = cubes[0].vertices.shape[0]
    for i in range(n):
        cubes[i].vertices += 2*i # shift cube
        cubes[i].faces += i*nf
    return Mesh(np.vstack([c.vertices for c in cubes]),
                np.vstack([c.faces for c in cubes]))
def test_cube():
    m = cube()
    nc, component_vertices, component_faces = connected_components(m)
    assert nc == 1
def _test_n_cubes(n):
    repc = _make_repeated_cube(n)
    nc, component_vertices, component_faces = connected_components(repc)
    assert nc == n
    for i in range(nc):
        v = component_vertices[i]
        v.sort()
        ci = v[0] // 8
        assert np.array_equal(v, np.arange(ci*8, ci*8+8))
        f = component_faces[i]
        f.sort()
        ci = f[0] // 12
        assert np.array_equal(f, np.arange(ci*12, ci*12+12))
def test_2_cubes():
    _test_n_cubes(2)
def test_3_cubes():
    _test_n_cubes(3)
def test_10_cubes():
    _test_n_cubes(10)

def _test_split_n_cubes(n):
    repc = _make_repeated_cube(n)
    ms = split_mesh(repc)
    assert len(ms) == n
    c = cube()
    for m in ms:
        assert is_isomorphic(m, c)
        c.vertices += 2
def test_split_1_cube():
    _test_split_n_cubes(1)
def test_split_2_cubes():
    _test_split_n_cubes(2)
def test_split_3_cubes():
    _test_split_n_cubes(3)
def test_split_10_cubes():
    _test_split_n_cubes(10)
