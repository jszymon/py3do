import io

from py3do.io import read_ascii_stl

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
