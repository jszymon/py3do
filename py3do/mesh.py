"""Triangular mesh."""

import copy

import numpy as np

from .geom import normals_cross

def _as_3col(x, *, fl=True):
    """Convert x to 3 column array.

    Works for empty lists"""
    if len(x) == 0:
        return np.empty((0,3), dtype=np.double if fl else np.uint)
    if fl:
        x = np.asarray(x, dtype=float)
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
    def __init__(self, vertices, faces, /, normals=None, *,
                 fix_nan_normals=False):
        """Create a mesh.

        Setting fix_nan_normals=True replaces Nan's in normals with
        zeros.  Useful for meshes with bad normals, e.g. collinear
        triangles.
        """
        self.vertices = _as_3col(vertices, fl=True)
        self.faces = _as_3col(faces, fl=False)
        if normals is not None:
            self.normals = _as_3col(normals, fl=True)
        else:
            self.normals, _ = normals_cross(self)
        if fix_nan_normals:
            self.normals[np.isnan(self.normals)] = 0
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
            raise RuntimeError("faces must have 3 vertices")
        if (self.faces < 0).any():
            raise RuntimeError("Negative indices in faces")
        n_vert = self.vertices.shape[0]
        if (self.faces >= n_vert).any():
            raise RuntimeError("faces indices out of bounds")
        _check_points_array(self.normals, "normals")

    def clone(self):
        return copy.deepcopy(self)
    
    def round_coords(self, decimals, *, inplace=False, merge=False):
        """Round coordinates of all points.

        merge identical points if requested."""
        if merge:
            raise NotImplementedError("round_coords: Merging is not implemented")
        if inplace:
            np.around(self.vertices, decimals, out=self.vertices)
            if self.normals is not None:
                self.normals, _ = normals_cross(self)
            return None
        return Mesh(np.around(self.vertices, decimals),
                    self.faces, self.normals)

    def extents(self):
        """Size of the bounding box."""
        M = self.vertices.max(axis=0)
        m = self.vertices.min(axis=0)
        return M - m

    def delete_edge(self, i, j):
        """Delete an edge i--j.

        vertex i is removed, all edges pointing to it now point at j.
        Results in an usused vertex.

        """
        i_mask = (self.faces == i).any(axis=1)
        j_mask = (self.faces == j).any(axis=1)
        f_mask = ~(i_mask & j_mask)
        self.faces = self.faces[f_mask]
        self.normals = self.normals[f_mask]
        self.faces[self.faces == i] = j
