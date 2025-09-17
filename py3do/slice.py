"""Slice models with plane."""

import numpy as np

import mapbox_earcut

from .mesh import Mesh
from .topo import unused_vertices
from .geom import normals_cross

def slice_horiz_0(m, keep="both", fill=None):
    """Slices the model m with a horizontal place z=0.

    keep='both' (default) means the whole model is kept cut points/edges are
    simply added to the model.
    keep='positive'/'negative': keep only part of model lying on the
    positive/negative side

    if fill is True the resulting hole will be filled.  Default: fill
    when keep is either positive or negative.

    """
    assert keep in ['both', 'positive', 'negative']
    if fill is None:
        fill = (keep != "both")
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
    # for each face find single vertex on opposite side
    fs_cut = fs[cut_face_idx]
    num_pos = (fs_cut >= 0).sum(axis=1)
    vertex_opposite = np.argmax(fs_cut * (2*(num_pos == 1) - 1).reshape(-1,1),
                                axis=1)
    # find two vertices on the same side
    two_on_side = np.array([[1,2], [2,0], [0,1]])[vertex_opposite]
    #two_on_side = np.column_stack([(vertex_opposite==0),
    #                               (vertex_opposite<2)+1])
    
    # new vertex coordinates
    
    cut_faces = m.faces[cut_face_idx]
    # indices of opposite vertices in vertex array
    opposite_idx = np.take_along_axis(cut_faces, vertex_opposite.reshape(-1,1), axis=1)
    two_on_side_idx_1 = np.take_along_axis(cut_faces, two_on_side[:,0].reshape(-1,1), axis=1)
    two_on_side_idx_2 = np.take_along_axis(cut_faces, two_on_side[:,1].reshape(-1,1), axis=1)
    
    # eliminate repeated new vertices (each edge belongs to two faces)
    cut_edges = np.vstack([np.column_stack([opposite_idx, two_on_side_idx_1]),
                           np.column_stack([opposite_idx, two_on_side_idx_2])])
    # eliminate repeated edges, get a map from original to unique edges
    cut_edges.sort(axis=1)
    unique_edges, cut_point_index = np.unique(cut_edges, axis=0, return_inverse=True)
    # proportions
    d = np.abs(d[unique_edges])
    p = d / d.sum(axis=1, keepdims=True)
    v12 = m.vertices[unique_edges]
    new_v = (v12 * p[:,::-1,np.newaxis]).sum(axis=1)
    new_v[:,2] = 0

    # create the new model
    m_sliced = m.clone()
    m_sliced.faces = m_sliced.faces[keep_faces]
    m_sliced.normals = m_sliced.normals[keep_faces]
    m_sliced.vertices = np.vstack([m.vertices, new_v])

    # new faces
    n_cut = opposite_idx.shape[0]
    n1 = m.vertices.shape[0]
    new_faces_1 = np.column_stack([opposite_idx,
                                   n1 + cut_point_index[:n_cut],
                                   n1 + cut_point_index[n_cut:]])
    new_faces_2 = np.column_stack([two_on_side_idx_1,
                                   n1 + cut_point_index[n_cut:],
                                   n1 + cut_point_index[:n_cut]])
    new_faces_3 = np.column_stack([two_on_side_idx_1,
                                   two_on_side_idx_2,
                                   n1 + cut_point_index[n_cut:]])
    new_normals = m.normals[cut_face_idx]
    #m_sliced.faces = np.vstack([m_sliced.faces, new_faces_1, new_faces_2, new_faces_3])
    #m_sliced.normals = np.vstack([m_sliced.normals, new_normals, new_normals, new_normals])
    #return m_sliced

    # remove faces which are on wrong side of the plane
    if keep != "both":
        if keep == "positive":
            mask = (s[opposite_idx.ravel()] >= 0)
        else:
            mask = (s[opposite_idx.ravel()] <= 0)
        new_faces_1 = new_faces_1[mask]
        new_normals_1 = new_normals[mask]
        new_faces_2 = new_faces_2[~mask]
        new_normals_2 = new_normals[~mask]
        new_faces_3 = new_faces_3[~mask]
    else:
        new_normals_1 = new_normals_2 = new_normals
    m_sliced.faces = np.vstack([m_sliced.faces,
                                new_faces_1,
                                new_faces_2,
                                new_faces_3,])
    m_sliced.normals = np.vstack([m_sliced.normals,
                                  new_normals_1,
                                  new_normals_2,
                                  new_normals_2, # same as normals_2
                                  ])

    # fill the hole
    if fill:
        # find new edges
        new_edges = np.vstack([new_faces_1[:,[2,1]], new_faces_2[:,[2,1]]])
        new_edge_order = np.argsort(new_edges[:,0])
        new_edges = new_edges[new_edge_order]
        print(new_edges)
        
        edge_dict = {int(i):int(j) for i, j in new_edges}
        assert len(new_edges) == len(edge_dict)

        # find cycles of vertices
        remaining = set(range(two_on_side.shape[0]))
        cycles = []
        while len(edge_dict) > 0:
            cur_cycle = []
            cur_idx = next(iter(edge_dict.keys()))
            while cur_idx in edge_dict:
                cur_cycle.append(cur_idx)
                prev_idx = cur_idx
                cur_idx = edge_dict[cur_idx]
                del edge_dict[prev_idx]
            print(cur_cycle)
            cycles.append(np.array(cur_cycle))

        # prepare data for mapbox_earcut
        # TODO: holes are treated as separate vertices!  fix this!
        for cycle in cycles:
            print(cycle)
            cycle_verts = m_sliced.vertices[cycle][:,:2]
            cycle_faces = mapbox_earcut.triangulate_float64(cycle_verts,
                                                            [len(cycle)])
            cycle_faces = cycle[cycle_faces] # return to original vertex numbering
            m_sliced.faces = np.vstack([m_sliced.faces,
                                        cycle_faces.reshape(-1,3)])
        # TODO: only update normal of new faces
        m_sliced.normals, _ = normals_cross(m_sliced)

    # remove unused vertices
    m_sliced.delete_vertices(unused_vertices(m_sliced))

    from py3do.topo import EdgeToFaceMap
    efm = EdgeToFaceMap(m_sliced)
    print(f"Model edges are {'' if efm.oriented else 'NOT '}correctly oriented")
    print(f"Model is {'' if efm.manifold else 'NOT '}manifold")
    print(f"Model is {'' if efm.watertight else 'NOT '}watertight")

    return m_sliced
    
