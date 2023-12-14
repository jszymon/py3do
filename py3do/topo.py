"""Topological properties."""

import numpy as np

from .utils import UnionFind, arg_split

def repeated_face_vertices(m):
    """Check if any triangle has repeated vertices."""
    mask = (m.faces[:,0] == m.faces[:,1]) |\
      (m.faces[:,0] == m.faces[:,2]) |\
      (m.faces[:,1] == m.faces[:,2])
    return np.nonzero(mask)[0]

def unused_vertices(m):
    """Find vertices not in any faces."""
    used_vertices = np.unique(m.faces.ravel())
    unused_vertices = np.setdiff1d(np.arange(m.vertices.shape[0]), used_vertices)
    return unused_vertices

def sorted_edges(m, return_order=False, unique=True):
    """Return edges with sorted vertices.

    If return_order is True a Boolean array is returned indicated
    original edge had i > j.  If unique is True, repeated edges are
    removed.  return_order and unique cannot both be True.

    """
    edges1 = m.faces[:,0:2]
    edges2 = m.faces[:,1:3]
    edges3 = m.faces[:,[2,0]]
    edges = np.concatenate([edges1, edges2, edges3])
    orientations = (edges[:,0] > edges[:,1])
    edges[orientations] = edges[orientations][:, [1,0]]  # sort edge endpoints
    if return_order:
        assert not unique
        return edges, orientations
    if unique:
        assert not return_order
        edges = np.unique(edges, axis=0)
    return edges

class EdgeToFaceMap:
    """Stores faces adjacent to each edge.

    Edges are sorted pairs (i,j) of vertex numbers.  Original edge
    orientation is stored separately: 0 - ascending i<j, 1 - descending
    i>j.
    """
    def __init__(self, m):
        edges, orientations = sorted_edges(m, return_order=True, unique=False)
        edges = np.column_stack([edges,
                        np.tile(np.arange(m.faces.shape[0],
                                              dtype=edges.dtype), 3)])
        rec_dtype = edges.dtype
        self.view_dt = np.dtype([("i", rec_dtype), ("j", rec_dtype),
                                     ("face", rec_dtype)])
        self.query_dt = np.dtype([("i", rec_dtype), ("j", rec_dtype)])
        # create a view to emulate lexicographic searchsorted
        assert edges.flags.c_contiguous
        n = edges.shape[0]
        edges_rec = edges.view(dtype=self.view_dt).reshape((n,))
        idx = edges_rec.argsort()
        self.edges_rec = edges_rec[idx]
        self.orientations = orientations[idx] * 1
        self.unique_edges, self.unique_edges_ptr, self.face_counts = \
                        np.unique(self.edges_rec[["i", "j"]],
                        return_index=True, return_counts=True)
        self.manifold = np.all(self.face_counts <= 2)
        self.watertight = np.all(self.face_counts == 2)
        # check edge orientation
        fc2_idx = np.nonzero(self.face_counts == 2)[0]
        ptr = self.unique_edges_ptr[fc2_idx]
        orients = self.orientations[ptr] + self.orientations[ptr + 1]
        self.oriented_edge = np.full(self.unique_edges.shape[0], False)
        self.oriented_edge[fc2_idx[orients == 1]] = True
        self.oriented_edge[self.face_counts == 1] = True
        self.oriented = np.all(self.oriented_edge)
    def get_boundary_edges(self):
        return self.unique_edges[self.face_counts == 1]
    def get_multiface_edges(self):
        """Get edges adjacent to more than two faces."""
        return self.unique_edges[self.face_counts > 2]
    def get_misoriented_edges(self):
        return self.unique_edges[~self.oriented_edge]
    def find_faces(self, i, j=None, return_type="dict"):
        """Find faces adjacent to given edges."""
        dt = self.query_dt
        if np.isscalar(i):
            if not np.isscalar(j):
                raise RuntimeError("Edge indices must have matching shapes")
            ij = np.array([(i, j)], dtype=dt)
        elif j is not None:
            ij = np.squeeze(np.atleast_2d(np.column_stack([i, j])).view(dt))
        else:
            ij = np.atleast_2d(i)
            ij = np.atleast_2d(i).view(dt).reshape((len(ij),))
        indices = np.searchsorted(self.unique_edges, ij)
        assert return_type == "dict"
        ret = dict()
        for k in indices:
            e = tuple(self.unique_edges[k])
            nf = self.face_counts[k]
            idx = self.unique_edges_ptr[k]
            ret[e] = list(self.edges_rec[idx:idx+nf]["face"])
        return ret

def connected_components(m):
    """Detect connected components.  Only face information is used,
    overlapping is not taken into account.

    returns the number of connected components and two lists of
    integer arrays of respectively vertex and face indices of each
    component.

    """
    n = m.vertices.shape[0]
    uf = UnionFind(n)
    for (x,y,z) in m.faces:
        uf.union(x, y)
        uf.union(x, z)
        uf.union(y, z)
        if uf.n == 1:
            break
    nc = uf.n
    vert_components = uf.sets()
    face_components = vert_components[m.faces[:,0]]
    return nc, arg_split(vert_components, nc), arg_split(face_components, nc)
