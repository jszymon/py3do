"""OBJ file reader."""

from contextlib import nullcontext

import numpy as np

from .. import Mesh

def read_obj(fname, *, fix_nan_normals=False):
    """Read a simple OBJ file with vertices and triangular faces.
    
    This function only reads vertex coordinates and assumes all faces
    are triangular. It does not read normals, texture coordinates,
    or other OBJ features.
    
    Args:
        fname: File path or file-like object
    
        fix_nan_normals: if True allow incorrect normals an fix them.
            Useful for models with degenerate faces.
        
    Returns:
        Mesh object with vertices and faces

    """
    vertices = []
    faces = []
    
    # Handle both file paths and file-like objects
    if hasattr(fname, 'read'):
        f_ctx = nullcontext(fname)
    else:
        f_ctx = open(fname, 'r')
    with f_ctx as fl:
        for line in fl:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':
                # Vertex line: v x y z
                if len(parts) >= 4:
                    vertex = tuple(float(parts[i]) for i in range(1, 4))
                    vertices.append(vertex)
            elif parts[0] == 'f':
                # Face line: f v1 v2 v3 (assuming triangular)
                if len(parts) >= 4:
                    # OBJ indices are 1-based, convert to 0-based
                    face = tuple(int(parts[i].split('/')[0]) - 1 for i in range(1, 4))
                    faces.append(face)
    m = Mesh(vertices, faces, fix_nan_normals=fix_nan_normals)
    return m
