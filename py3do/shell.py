"""Offsets and shells."""

import numpy as np

from .mesh import Mesh
from .geom import vertex_normals

def check_offset(m, v_disp):
    """Computes accuracy of offset based on diplacing estisting mesh
    points.

    Returns a matrix of true offsets at every vertex (axis 1) of every face (axis 0).
    m is the original model and v_disp an matrix of displacement
    vectors for vertices of m.

    """
    fns = m.normals.reshape(-1,1,3)#.repeat(3, axis=1)
    true_offsets = np.einsum("...k,...k", v_disp[m.faces], fns)
    return true_offsets

def offset_mesh(m, d, *, method="vertex normals", return_true_offsets=False):
    """Offset mesh faces by d (possibly approximately)

    If exact offset is not possible ensures each face to be
    offset at least by d, but may distort mesh shape.

    if return_true_offsets is True return additionally the matrix of
    true offsets for each vertex of each face (see check_offset).
    maximum of this matrix is the largest true offset.  This value can
    be used to assess offset mesh distortion.

    Currently only one method is implemented: * 'vertex normals'
    offset using vertex normals.  Works well for some shapes, worse
    for others.

    """
    if method == "vertex normals":
        vn = vertex_normals(m, "angle weighted", normalize=False)
        # ensure every face is offset by at least d
        true_offsets = check_offset(m, vn)
        # different offset scaling for each vertex
        min_face_offsets = np.full(m.vertices.shape[0], true_offsets.max() + 1)
        np.minimum.at(min_face_offsets, m.faces[:,0], true_offsets[:,0])
        np.minimum.at(min_face_offsets, m.faces[:,1], true_offsets[:,1])
        np.minimum.at(min_face_offsets, m.faces[:,2], true_offsets[:,2])
        min_face_offsets = min_face_offsets.reshape(-1,1)
        s = d / min_face_offsets
        v_disp = vn * s
    else:
        raise RuntimeError("Wrong mesh offset method: " + method)

    om = Mesh(m.vertices + v_disp, m.faces)
    if return_true_offsets:
        true_offsets = check_offset(m, v_disp)
        return om, true_offsets
    return om
