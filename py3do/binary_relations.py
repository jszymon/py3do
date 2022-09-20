"""Binary relations between meshes."""

import numpy as np

def is_isomorphic(m1, m2, tol=0):
    """A simple test if two meshes are isomorphic.

    tol is the tolerance in vertex positions: vertices closer than tol
    are assumed equal."""
    if m1.vertices.shape[0] != m2.vertices.shape[0]:
        return False
    if m1.faces.shape[0] != m2.faces.shape[0]:
        return False
    # build a map of m1 to m2 vertices
    if tol == 0:
        vertex_map_1 = dict()
        for i, v in enumerate(m1.vertices):
            vertex_map_1[tuple(v)] = i
        vmap = [-1] * m1.vertices.shape[0]
        for i, v in enumerate(m2.vertices):
            vt = tuple(v)
            if vt not in vertex_map_1:
                return False
            vmap[i] = vertex_map_1[vt]
        vmap = np.array(vmap)
    else:
        raise NotImplemented("is_isomorphic: nonzero tolerance not implemented")
    mapped_faces = vmap[m2.faces]
    # sort faces lexicographically
    i1 = np.lexsort(m1.faces.T)
    i2 = np.lexsort(mapped_faces.T)
    return np.array_equal(m1.faces[i1], mapped_faces[i2])
