"""Triangular mesh."""

import numpy as np

class Mesh:
    def __init__(self, vertices, faces, normals=None):
        self.vertices = np.asfarray(vertices)
        self.faces = np.asarray(faces)
        self.check_faces_and_vertices()
    def check_faces_and_vertices(self):
        """Basic checks of consistency of faces and vertices."""
        if len(self.vertices.shape) != 2:
            raise RuntimeError("vertices must be a 2d array")
        if self.vertices.shape[1] != 3:
            raise RuntimeError("vertices must have 3 coordinates")
        if np.issubdtype(self.faces.dtype, np.integer):
            raise RuntimeError("faces must be an integral type array")
        if len(self.faces.shape) != 2:
            raise RuntimeError("faces must be a 2d array")
        if self.faces.shape[1] != 3:
            raise RuntimeError("faces must have 3 coordinates")
        if not np.isfinite(self.vertices).all():
            raise RuntimeError("Inf or Nan in vertices")
        if (self.faces < 0).any():
            raise RuntimeError("Negative indices in faces")
        n_vert = self.vertices.shape[0]
        if (self.faces >= n_vert).any():
            raise RuntimeError("faces indices out of bounds")
