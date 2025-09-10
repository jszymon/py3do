"""Slice models with plane."""

import numpy as np

from .mesh import Mesh
from .topo import unused_vertices

def slice_horiz_0(m, keep="both"):
    """Slices the model m with a horizontal place z=0.

    keep='both' (default) means the whole model is kept cut points/edges are
    simply added to the model.
    keep='positive'/'negative': keep only part of model lying on the
    positive/negative side

    """
    assert keep in ['both', 'positive', 'negative']
    d = m.vertices[:,2] # distances from the plane
    s = np.sign(d) # side of the plane each vertex is on
    fs = s[m.faces]
    pos_faces = (fs >= 0).all(axis=1)
    neg_faces = (fs <= 0).all(axis=1)
    if keep == "both":
        keep_faces = pos_faces | neg_faces
    elif keep == "positive":
        keep_faces = pos_faces
    elif keep == "negative":
        keep_faces = neg_faces
    # full model on desired side
    if keep_faces.all():
        return m
    # full model on opposite side
    if not keep_faces.any():
        return Mesh([], [])
    cut_face_idx = np.flatnonzero(~(pos_faces | neg_faces))
    # for each face find single vertes on opposite side
    fs_cut = fs[cut_face_idx]
    num_pos = (fs_cut >= 0).sum(axis=1)
    vertex_opposite = np.argmax(fs_cut * (2*(num_pos == 1) - 1).reshape(-1,1),
                                axis=1)
    two_on_side = np.array([[1,2], [0,2], [0,1]])[vertex_opposite]
    #two_on_side = np.column_stack([(vertex_opposite==0),
    #                               (vertex_opposite<2)+1])

    # new vertex coordinates
    
    cut_faces = m.faces[cut_face_idx]
    # indices of opposite vertices in vertex array
    opposite_idx = np.take_along_axis(cut_faces, vertex_opposite.reshape(-1,1), axis=1)
    v1 = m.vertices[opposite_idx].squeeze()
    d1 = np.abs(d[opposite_idx])
    two_on_side_idx_1 = np.take_along_axis(cut_faces, two_on_side[:,0].reshape(-1,1), axis=1)
    v2 = m.vertices[two_on_side_idx_1].squeeze()
    d2 = np.abs(d[two_on_side_idx_1])
    two_on_side_idx_2 = np.take_along_axis(cut_faces, two_on_side[:,1].reshape(-1,1), axis=1)
    v3 = m.vertices[two_on_side_idx_2].squeeze()
    d3 = np.abs(d[two_on_side_idx_2])
    # proportions
    p1 = d2 / (d1+d2)
    p2 = d3 / (d1+d3)
    new_v1 = p1*v1 + (1-p1)*v2
    new_v2 = p2*v1 + (1-p2)*v3
    new_v1[:,2] = 0
    new_v2[:,2] = 0

    # create the new model
    m_sliced = m.clone()
    m_sliced.faces = m_sliced.faces[keep_faces]
    m_sliced.normals = m_sliced.normals[keep_faces]
    m_sliced.vertices = np.vstack([m.vertices, new_v1, new_v2])

    # new faces
    n_cut = opposite_idx.shape[0]
    n1 = m.vertices.shape[0]
    n2 = n1 + new_v1.shape[0]
    new_faces_1 = np.column_stack([opposite_idx, n1 + np.arange(n_cut),
                                   n2 + np.arange(n_cut)])
    normals_1 = m.normals[cut_face_idx]
    new_faces_2 = np.column_stack([two_on_side_idx_1, n1 + np.arange(n_cut),
                                   n2 + np.arange(n_cut)])
    normals_2 = m.normals[cut_face_idx]
    new_faces_3 = np.column_stack([two_on_side_idx_1, two_on_side_idx_2,
                                   n2 + np.arange(n_cut)])
    # remove faces which are on wrong side of the plane
    if keep != "both":
        if keep == "positive":
            mask = (s[opposite_idx.ravel()] >= 0)
        else:
            mask = (s[opposite_idx.ravel()] <= 0)
        new_faces_1 = new_faces_1[mask]
        normals_1 = normals_1[mask]
        new_faces_2 = new_faces_2[~mask]
        normals_2 = normals_2[~mask]
        new_faces_3 = new_faces_3[~mask]
    m_sliced.faces = np.vstack([m_sliced.faces,
                                new_faces_1,
                                new_faces_2,
                                new_faces_3,])
    m_sliced.normals = np.vstack([m_sliced.normals,
                                  normals_1,
                                  normals_2,
                                  normals_2, # same as normals_2
                                  ])
    # remove unused vertices
    m_sliced.delete_vertices(unused_vertices(m_sliced))
    return m_sliced
    
