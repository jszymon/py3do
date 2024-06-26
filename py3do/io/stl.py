import io
from contextlib import nullcontext
from itertools import count
from collections import defaultdict
from struct import unpack, pack, Struct

import numpy as np

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
    # uncomment for unsorted defaultdict
    #items = list(vertex_map.items())
    #items.sort(key = lambda x: x[1])
    #vertices = [i[0] for i in items]
    # assume defaultdict is sorted
    vertices = list(vertex_map.keys())
    return vertices

def read_ascii_stl(fname, *, fix_nan_normals=False):
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
        while True:
            normal, li, l = _parse_vector(f, "facet normal",
                                          raise_on_nonmatch=False)
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
    m = Mesh(vertices, faces, normals, fix_nan_normals=fix_nan_normals)
    return m

def _vector_to_str(v):
    return " ".join(str(x) for x in v)
def write_ascii_stl(m, fname, model_name="exported from py3do", indent=2):
    if m.normals is None:
        raise RuntimeError("Face normal not available in model")
    if hasattr(fname, 'write'):
        f_ctx = nullcontext(fname)
    else:
        f_ctx = open(fname, 'w')
    fvs = m.vertices[m.faces]  # vertices of faces
    ol_str = "\nouter loop\n" + " "*indent + "vertex "
    v_str = "\n" + " "*indent + "vertex "
    with f_ctx as f:
        f.write("solid ")
        f.write("model_name\n")
        for i, fv in enumerate(fvs.tolist()):
            f.write("facet normal ")
            f.write(_vector_to_str(m.normals[i]))
            f.write(ol_str)
            f.write(_vector_to_str(fv[0]))
            f.write(v_str)
            f.write(_vector_to_str(fv[1]))
            f.write(v_str)
            f.write(_vector_to_str(fv[2]))
            f.write("\nendloop\n")
            f.write("endfacet\n")
        f.write("endsolid ")
        f.write("model_name")

def _read_n_bytes(f, n):
    b = f.read(n)
    if len(b) != n:
        raise RuntimeError("Unexpected end of file")
    return b
def read_binary_stl_numpy(fname, *, fix_nan_normals=False):
    """Use numpy operations to read stl.  Unfortunately np.unique is
    slower that python dicts."""
    facet_dtype = [("normal", "<f4", (3,)),
                   ("v", "<f4", (3,3)),
                   ("attr", np.uint16)
                   ]
    if hasattr(fname, 'read'):
        f_ctx = nullcontext(fname)
    else:
        f_ctx = open(fname, 'rb')
    with f_ctx as f:
        header = _read_n_bytes(f, 80)
        n_faces_b = _read_n_bytes(f, 4)
        n_faces = unpack("<L", n_faces_b)[0]
        try:
            d = np.fromfile(f, dtype=facet_dtype, count=n_faces)
        except io.UnsupportedOperation: # probably StringIO
            d = np.frombuffer(f.read(), dtype=facet_dtype, count=n_faces)            
        if len(f.read(1)) != 0:
            raise RuntimeError("Expected end of file")
    vertices, faces = np.unique(d["v"].reshape(-1,3), axis=0, return_inverse=True)
    faces = faces.reshape(-1,3)
    normals = d["normal"]
    m = Mesh(vertices, faces, normals, fix_nan_normals=fix_nan_normals)
    return m
def read_binary_stl(fname, *, fix_nan_normals=False):
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
        n_faces_b = _read_n_bytes(f, 4)
        n_faces = unpack("<L", n_faces_b)[0]
        buf = f.read(facet_str.size * n_faces)
        if len(f.read(1)) != 0:
            raise RuntimeError("Expected end of file")
    for normal_x, normal_y, normal_z, \
        v1x, v1y, v1z, \
        v2x, v2y, v2z, \
        v3x, v3y, v3z, \
        attr in facet_str.iter_unpack(buf):
        normal = (normal_x, normal_y, normal_z)
        v1 = (v1x, v1y, v1z)
        v2 = (v2x, v2y, v2z)
        v3 = (v3x, v3y, v3z)
        i1 = vertex_map[v1]
        i2 = vertex_map[v2]
        i3 = vertex_map[v3]
        faces.append((i1, i2, i3))
        normals.append(normal)
    vertices = _vertex_list_from_map(vertex_map)
    m = Mesh(vertices, faces, normals, fix_nan_normals=fix_nan_normals)
    return m

def write_binary_stl(m, fname, header=b"exported from py3do"):
    if m.normals is None:
        raise RuntimeError("Face normal not available in model")
    if hasattr(fname, 'write'):
        f_ctx = nullcontext(fname)
    else:
        f_ctx = open(fname, 'wb')
    fvs = m.vertices[m.faces]  # vertices of faces
    if header.startswith(b"solid"):
        raise RuntimeError("Binary STL header cannot start with 'solid'")
    if len(header) > 80:
        raise RuntimeError("Binary STL header too long")
    header = (header + b" " * 80)[:80]
    assert len(header) == 80
    n_faces = m.faces.shape[0]
    n_faces_b = pack("<L", n_faces)
    attr_b = pack("<H", 0)
    facet_str = Struct("<" + "f" * 12)
    with f_ctx as f:
        f.write(header)
        f.write(n_faces_b)
        for i, fv in enumerate(fvs.tolist()):
            face_b = facet_str.pack(*m.normals[i], *fv[0], *fv[1], *fv[2])
            f.write(face_b)
            f.write(attr_b)

def read_stl(fname, *, fix_nan_normals=False):
    """Read binary or ascii STL.

    Type is automatically determined.  Setting fix_nan_normals=True
    allows for reading files with e.g. collinear triangles (sets their
    normals to 0).
    """
    # determine file type
    header = None
    if hasattr(fname, 'read'):
        f_ctx = nullcontext(fname)
    else:
        f_ctx = open(fname, 'rb')
    with f_ctx as f:
        header = _read_n_bytes(f, 6)
    if header.lower() == b"solid ":
        m = read_ascii_stl(fname, fix_nan_normals=fix_nan_normals)
    else:
        m = read_binary_stl(fname, fix_nan_normals=fix_nan_normals)
    return m
