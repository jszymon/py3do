"""Triangular mesh."""

import numpy as np

def _as_3col(x, fl=True):
    """Convert x to 3 column array.

    Works for empty lists"""
    if len(x) == 0:
        return np.empty((0,3), dtype=np.float if fl else np.uint)
    if fl:
        x = np.asfarray(x)
    else:
        x = np.asarray(x)
    return np.asarray(x)
def _check_points_array(x, name):
    """Make sure x is a 3 column matrix.

    name is used for error reporting"""
    if len(x.shape) != 2:
        raise RuntimeError(name + " must be a 2d array")
    if x.shape[1] != 3:
        raise RuntimeError(name + " must have 3 coordinates")
    if not np.isfinite(x).all():
        raise RuntimeError("Inf or Nan in " + name)

class Mesh:
    def __init__(self, vertices, faces, normals=None):
        self.vertices = _as_3col(vertices, fl=True)
        self.faces = _as_3col(faces, fl=False)
        if normals is not None:
            self.normals = _as_3col(normals, fl=True)
        else:
            self.normals = None
        self.check_faces_and_vertices()
    def check_faces_and_vertices(self):
        """Basic checks of consistency of faces and vertices."""
        _check_points_array(self.vertices, "vertices")
        if not np.issubdtype(self.faces.dtype, np.integer):
            raise RuntimeError("faces must be an integer typed array, got "\
                               + str(self.faces.dtype))
        if len(self.faces.shape) != 2:
            raise RuntimeError("faces must be a 2d array")
        if self.faces.shape[1] != 3:
            raise RuntimeError("faces must have 3 coordinates")
        if (self.faces < 0).any():
            raise RuntimeError("Negative indices in faces")
        n_vert = self.vertices.shape[0]
        if (self.faces >= n_vert).any():
            raise RuntimeError("faces indices out of bounds")
        if self.normals is not None:
            _check_points_array(self.normals, "normals")
        else:
            pass
            #self.normals, _ = normals_Newell(self)
