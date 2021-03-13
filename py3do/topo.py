"""Topological properties."""

import numpy as np

def repeated_face_vertices(m):
    """Check if any triangle has repeated vertices."""
    mask = (m.faces[:,0] == m.faces[:,1]) |\
      (m.faces[:,0] == m.faces[:,2]) |\
      (m.faces[:,1] == m.faces[:,2])
    return np.nonzero(mask)[0]

class EdgeToFaceMap:
    """Stores faces adjacent to each edge.

    Edges are sorted pairs (i,j) of vertex numbers.  Original edge
    orientation is stored separately: 0 - ascending i<j, 1 - descending
    i>j.
    """
    def __init__(self, m):
        edges1 = m.faces[:,0:2]
        edges2 = m.faces[:,1:3]
        edges3 = m.faces[:,[2,0]]
        edges = np.concatenate([edges1, edges2, edges3])
        orientations = (edges[:,0] > edges[:,1])
        edges[orientations] = edges[orientations][:, [1,0]]  # sort edge endpoints
        self.edges = np.column_stack([edges,
                            np.tile(np.arange(m.faces.shape[0]), 3)])
        rec_dtype = edges.dtype
        self.view_dt = np.dtype([("i", rec_dtype), ("j", rec_dtype),
                                     ("face", np.uint)])
        # create a view for searchsorted
        assert self.edges.flags.c_contiguous
        n = self.edges.shape[0]
        edges_rec = self.edges.view(dtype=self.view_dt).reshape((n,))
        idx = edges_rec.argsort()
        self.edges_rec = edges_rec[idx]
        self.orientations = orientations[idx] * 1
    def find_faces(self, i, j):
        """Find faces for given edges."""
        if np.isscalar(i):
            if not np.isscalar(j):
                raise RuntimeError("Edge indices must have matching shapes")
            ij = np.array([i, j], dtype=self.view_dt)
