import io

import pytest

from py3do.io import read_ascii_stl
from py3do.io import read_binary_stl
from py3do.io import write_ascii_stl


def test_empty():
    f = io.StringIO("""solid empty
endsolid empty""")
    read_ascii_stl(f)

def test_triange():
    f = io.StringIO("""solid triangle
facet normal 0.0 0.0 -1.0
outer loop
vertex 1.0 0.0 0.0
vertex 0.0 -1.0 0.0
vertex 0.0 0.0 0.0
endloop
endfacet
endsolid triangle""")
    read_ascii_stl(f)

cube_stl = """solid cube
facet normal 0.0 0.0 -1.0
outer loop
vertex 10.0 0.0 0.0
vertex 0.0 -10.0 0.0
vertex 0.0 0.0 0.0
endloop
endfacet
facet normal 0.0 0.0 -1.0
outer loop
vertex 0.0 -10.0 0.0
vertex 10.0 0.0 0.0
vertex 10.0 -10.0 0.0
endloop
endfacet
facet normal -0.0 -1.0 -0.0
outer loop
vertex 10.0 -10.0 10.0
vertex 0.0 -10.0 0.0
vertex 10.0 -10.0 0.0
endloop
endfacet
facet normal -0.0 -1.0 -0.0
outer loop
vertex 0.0 -10.0 0.0
vertex 10.0 -10.0 10.0
vertex 0.0 -10.0 10.0
endloop
endfacet
facet normal 1.0 0.0 0.0
outer loop
vertex 10.0 0.0 0.0
vertex 10.0 -10.0 10.0
vertex 10.0 -10.0 0.0
endloop
endfacet
facet normal 1.0 0.0 0.0
outer loop
vertex 10.0 -10.0 10.0
vertex 10.0 0.0 0.0
vertex 10.0 0.0 10.0
endloop
endfacet
facet normal -0.0 -0.0 1.0
outer loop
vertex 10.0 -10.0 10.0
vertex 0.0 0.0 10.0
vertex 0.0 -10.0 10.0
endloop
endfacet
facet normal -0.0 -0.0 1.0
outer loop
vertex 0.0 0.0 10.0
vertex 10.0 -10.0 10.0
vertex 10.0 0.0 10.0
endloop
endfacet
facet normal -1.0 0.0 0.0
outer loop
vertex 0.0 0.0 10.0
vertex 0.0 -10.0 0.0
vertex 0.0 -10.0 10.0
endloop
endfacet
facet normal -1.0 0.0 0.0
outer loop
vertex 0.0 -10.0 0.0
vertex 0.0 0.0 10.0
vertex 0.0 0.0 0.0
endloop
endfacet
facet normal -0.0 1.0 0.0
outer loop
vertex 0.0 0.0 10.0
vertex 10.0 0.0 0.0
vertex 0.0 0.0 0.0
endloop
endfacet
facet normal -0.0 1.0 0.0
outer loop
vertex 10.0 0.0 0.0
vertex 0.0 0.0 10.0
vertex 10.0 0.0 10.0
endloop
endfacet
endsolid cube
"""
def test_cube():
    f = io.StringIO(cube_stl)
    read_ascii_stl(f)

def test_wrong_header():
    f = io.StringIO("""soli triangle
facet normal 0.0 0.0 -1.0
outer loop
vertex 1.0 0.0 0.0
vertex 0.0 -1.0 0.0
vertex 0.0 0.0 0.0
endloop
endfacet
endsolid triangle""")
    with pytest.raises(RuntimeError, match="Wrong ASCII STL header"):
        read_ascii_stl(f)

def test_missing_endsolid():
    f = io.StringIO("""solid empty
""")
    with pytest.raises(RuntimeError, match="Missing 'endsolid'"):
        read_ascii_stl(f)

def test_two_vertices():
    f = io.StringIO("""solid triangle
facet normal 0.0 0.0 -1.0
outer loop
vertex 1.0 0.0 0.0
vertex 0.0 -1.0 0.0
endloop
endfacet
endsolid triangle""")
    with pytest.raises(RuntimeError, match="Expected 'vertex'"):
        read_ascii_stl(f)

def test_four_vertices():
    f = io.StringIO("""solid triangle
facet normal 0.0 0.0 -1.0
outer loop
vertex 1.0 0.0 0.0
vertex 0.0 -1.0 0.0
vertex 0.0 0.0 1.0
vertex 0.0 0.0 0.0
endloop
endfacet
endsolid triangle""")
    with pytest.raises(RuntimeError, match="Expected 'endloop'"):
        read_ascii_stl(f)

def test_cube_write():
    f = io.StringIO(cube_stl)
    cube = read_ascii_stl(f)
    import sys
    write_ascii_stl(cube, sys.stdout)
