from contextlib import nullcontext
from itertools import count
from collections import defaultdict
from struct import unpack, Struct

from .. import Mesh

def _numbered_line_reader(f):
    for li, l in enumerate(f):
        yield li+1, l.strip()

def _match_line(f, match):
    for li, l in f:
        if l != "": break
    else:
        raise RuntimeError("Unexpected end of file, expected " + str(match))
    if l != match:
        raise RuntimeError("Expected '" + str(match) + "' on line " + str(li))
def _parse_vector(f, match, raise_on_nonmatch=True):
    for li, l in f:
        if l != "": break
    else:
        if raise_on_nonmatch:
            raise RuntimeError("Unexpected end of file, expected " + str(match))
        if "li" not in locals():
            li = None
            l = ""
        return None, li, l
    if not l.startswith(match):
        if raise_on_nonmatch:
            raise RuntimeError("Expected '" + str(match) + "' on line " + str(li))
        return None, li, l
    toks = l[len(match):].strip().split()
    if len(toks) != 3:
        raise RuntimeError("A vector must have exactly 3 coordinates"
                               " (line " + str(li) + ")")
    vec = tuple(float(x) for x in toks)
    return vec, li, l
def _vertex_list_from_map(vertex_map):
    """Extract ordered list of vertices from vertex map."""
    items = list(vertex_map.items())
    items.sort(key = lambda x: x[1])
    vertices = [i[0] for i in items]
    return vertices

def read_ascii_stl(fname):
    vertex_map = defaultdict(count().__next__)
    faces = []
    normals = []
    if hasattr(fname, 'read'):
        f_ctx = nullcontext(fname)
    else:
        f_ctx = open(fname, 'r')
    with f_ctx as fl:
        f = _numbered_line_reader(fl)
        li, header = next(f)
        if header[0:6].lower() != "solid ":
            raise RuntimeError("Wrong ASCII STL header")
        name = header[6:].strip()
        #print("read stl", name)
        while True:
            normal, li, l = _parse_vector(f, "facet normal",
                                          raise_on_nonmatch=False)
            #print(normal, li, l)
            if normal is None:
                if li is None:  # end of file
                    raise RuntimeError("Missing 'endsolid'")
                if l.startswith("endsolid"):
                    if name != l[9:]:
                        print("Warning: different names in 'solid'"
                                  " and 'endsolid'")
                    break
                raise RuntimeError("Expected 'facet'")
            _match_line(f, "outer loop")
            v1, li, l = _parse_vector(f, "vertex")
            v2, li, l = _parse_vector(f, "vertex")
            v3, li, l = _parse_vector(f, "vertex")
            _match_line(f, "endloop")
            _match_line(f, "endfacet")
            #print("facet")
            #print("  normal", normal)
            #print("  v1", v1)
            #print("  v2", v2)
            #print("  v3", v3)
            i1 = vertex_map[v1]
            i2 = vertex_map[v2]
            i3 = vertex_map[v3]
            faces.append((i1, i2, i3))
            normals.append(normal)
        else:
            raise RuntimeError("Missing 'endsolid'")
        for li, l in f:
            if l != "":
                raise RuntimeError("Content after 'endsolid'")
    vertices = _vertex_list_from_map(vertex_map)
    print(vertices)
    #print(faces)
    #print(normals)
    m = Mesh(vertices, faces, normals)
    return m

def _read_n_bytes(f, n):
    b = f.read(n)
    if len(b) != n:
        raise RuntimeError("Unexpected end of file")
    return b
def read_binary_stl(fname):
    vertex_map = defaultdict(count().__next__)
    faces = []
    normals = []
    facet_str = Struct("<" + "f" * 12 + "H")
    if hasattr(fname, 'read'):
        f_ctx = nullcontext(fname)
    else:
        f_ctx = open(fname, 'rb')
    with f_ctx as f:
        header = _read_n_bytes(f, 80)
        n_factes_b = _read_n_bytes(f, 4)
        n_factes = unpack("<L", n_factes_b)[0]
        print("header", header)
        print(n_factes, "facets")
        for i in range(n_factes):
            facet = _read_n_bytes(f, facet_str.size)
            normal_x, normal_y, normal_z, \
            v1x, v1y, v1z, \
            v2x, v2y, v2z, \
            v3x, v3y, v3z, \
            attr = facet_str.unpack(facet)
            normal = (normal_x, normal_y, normal_z)
            v1 = (v1x, v1y, v1z)
            v2 = (v2x, v2y, v2z)
            v3 = (v3x, v3y, v3z)
            #print("facet")
            #print("  normal", normal)
            #print("  v1", v1)
            #print("  v2", v2)
            #print("  v3", v3)
            i1 = vertex_map[v1]
            i2 = vertex_map[v2]
            i3 = vertex_map[v3]
            faces.append((i1, i2, i3))
            normals.append(normal)
        if len(f.read(1)) != 0:
            raise RuntimeError("Expected end of file")
    #print(vertex_map)
    #print(faces)
    #print(normals)
    vertices = _vertex_list_from_map(vertex_map)
    m = Mesh(vertices, faces, normals)
    return m
