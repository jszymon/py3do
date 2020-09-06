"""Topological properties."""

import numpy as np

def repeated_face_vertices(m):
    """Check if any triangle has repeated vertices."""
    mask = (m.faces[:,0] == m.faces[:,1]) |\
      (m.faces[:,0] == m.faces[:,2]) |\
      (m.faces[:,1] == m.faces[:,2])
    return np.nonzero(mask)

def edge_face_map(m):
    """Compute faces adjacent to each edge.

    Edges are sorted pairs of vertex numbers.  Edge orientation in a
    face is also computed: 0 - ascending, 1 - descending."""
    edges1 = n.faces[:,0:2]
    edges2 = n.faces[:,1:3]
    edges3 = np.hstack([faces[:,2], faces[:,0]])
    edges = np.concatenate([edges1, edges2, edges3])
    orientations = (edges[:,0] < edges[:,1])
