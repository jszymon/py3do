"""Functions for generic mesh reading"""

from .stl import read_stl
from .obj import read_obj

def read_mesh(fname, *, fix_nan_normals=False):
    """Detect mesh type and read it.

    Currently STL and OBJ files are handled.  OBJ vs STL are
    recognized based on file extension, binary vs ASCII STLs based on
    content.

    """
    if hasattr(fname, 'read'):
        # this is a file object
        name = fname.name
    else:
        name = fname
    name = name.lower()
    if name.endswith(".stl"):
        return read_stl(fname, fix_nan_normals=fix_nan_normals)
    elif name.endswith(".obj"):
        return read_obj(fname, fix_nan_normals=fix_nan_normals)
    raise RuntimeError("read_mesh: unrecognized file type")
