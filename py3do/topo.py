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
                        np.tile(np.arange(m.faces.shape[0],
                                              dtype=edges.dtype), 3)])
        rec_dtype = self.edges.dtype
        self.view_dt = np.dtype([("i", rec_dtype), ("j", rec_dtype),
                                     ("face", rec_dtype)])
        # create a view to emulate lexicographic searchsorted
        assert self.edges.flags.c_contiguous
        n = self.edges.shape[0]
        edges_rec = self.edges.view(dtype=self.view_dt).reshape((n,))
        idx = edges_rec.argsort()
        self.edges_rec = edges_rec[idx]
        self.orientations = orientations[idx] * 1
        self.unique_edges, self.unique_edges_ptr, self.face_counts = \
                        np.unique(self.edges_rec[["i", "j"]],
                        return_index=True, return_counts=True)
        if np.all(self.face_counts <= 2):
            self.manifold = True
        if np.all(self.face_counts == 2):
            self.watertight = True
        # check edge orientation
        ptr = self.unique_edges_ptr[self.face_counts == 2]
        orients = self.orientations[ptr] + self.orientations[ptr + 1]
        self.oriented_edge = np.full(self.unique_edges.shape[0], False)
        self.oriented_edge[self.face_counts == 2][orients == 1] = True
        self.oriented_edge[self.face_counts == 1] = True
        print(orients)
        self.oriented = np.all(self.oriented_edge)
    def get_boundary_edges(self):
        pass
    def find_faces(self, i, j=None):
        """Find faces for given edges."""
        if np.isscalar(i):
            if not np.isscalar(j):
                raise RuntimeError("Edge indices must have matching shapes")
            ij = np.array([i, j, 0], dtype=self.view_dt)
        elif j is not None:
            ij = np.column_stack([i, j, np.zeros(i.shape[0])])
            ij = np.array(ij, dtype=self.view_dt)
        else:
            ij = np.vstack([i, j, np.zeros(i.shape[0])])
            ij = np.array(ij, dtype=self.view_dt)
        
