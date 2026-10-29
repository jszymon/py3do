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

def offset_mesh(m, d, *, method="vertex normals", return_true_offsets=False,
                offset_vecs=None):
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

    offset_vecs can be used to provide precomputed offset direction
    vectors, otherwise vertex normals are used.

    """
    if method == "vertex normals":
        if offset_vecs is None:
            vn = vertex_normals(m, "angle weighted", normalize=False)
        else:
            vn = offset_vecs
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

def _boundary_edges(m):
    """Directed boundary edges of m.

    Returns arrays a, b, f: edge a[k] -> b[k] is a boundary edge of
    face f[k], oriented as in the face."""
    f = m.faces
    a = f.ravel()
    b = np.roll(f, -1, axis=1).ravel()
    fi = np.repeat(np.arange(f.shape[0]), 3)
    pairs = np.sort(np.column_stack([a, b]), axis=1)
    _, inv, cnt = np.unique(pairs, axis=0, return_inverse=True,
                            return_counts=True)
    bmask = (cnt[inv.reshape(-1)] == 1)
    return a[bmask], b[bmask], fi[bmask]

def shell(m, d, *, edge="flat", n_round=8, offset_vecs=None):
    """Add thickness d to a flat (open surface) mesh m.

    The mesh is offset by d using offset_mesh (in the direction of
    face normals for d > 0, opposite for d < 0) and boundary edges of
    m are connected with their offset copies, which results in a
    closed solid.  Returns a new mesh.

    edge determines how the boundary of the shell is formed:

    "flat": each boundary edge is connected to its offset copy by two
    triangles.

    "inset": as "flat", but boundary vertices of the offset surface
    are moved back so that they do not protrude outside the original
    edge, i.e. their displacement has no component pointing outward
    from the surface along its boundary.  Identical to "flat" for
    planar meshes.

    "round": as "inset", but the edge is a half round of diameter
    equal to the length of the (inset) side wall, made of n_round
    segments.  The edge bulges outward from the original boundary.

    offset_vecs is passed on to offset_mesh.

    """
    if edge not in ("flat", "inset", "round"):
        raise ValueError("shell: unknown edge type " + repr(edge))
    if d == 0:
        raise ValueError("shell: thickness must be nonzero")
    if edge == "round" and n_round < 1:
        raise ValueError("shell: n_round must be at least 1")
    n = m.vertices.shape[0]
    v0 = m.vertices
    disp = offset_mesh(m, d, offset_vecs=offset_vecs).vertices - v0
    ba, bb, bf = _boundary_edges(m)
    bv = np.unique(np.concatenate([ba, bb]))
    # outward direction at boundary vertices, perpendicular to
    # boundary edges and lying in adjacent faces
    eo = np.cross(v0[bb] - v0[ba], m.normals[bf])
    outward = np.zeros_like(v0)
    np.add.at(outward, ba, eo)
    np.add.at(outward, bb, eo)
    ob = outward[bv]
    ob /= np.linalg.norm(ob, axis=1, keepdims=True)
    if edge != "flat":
        p = np.einsum("ij,ij->i", disp[bv], ob)
        disp[bv] -= np.maximum(p, 0).reshape(-1, 1) * ob
    v1 = v0 + disp
    vertices = [v0, v1]
    # rings[k][i] is the index of k-th edge profile vertex of
    # boundary vertex i
    rings = [np.arange(n)]
    if edge == "round":
        c = (v0[bv] + v1[bv]) / 2
        u = v0[bv] - c
        r = np.linalg.norm(u, axis=1, keepdims=True)
        o = ob - np.einsum("ij,ij->i", ob, u).reshape(-1, 1) * u / r**2
        o /= np.linalg.norm(o, axis=1, keepdims=True)
        n_vert = 2 * n
        for k in range(1, n_round):
            t = np.pi * k / n_round
            vertices.append(c + np.cos(t) * u + np.sin(t) * r * o)
            ring = np.full(n, -1)
            ring[bv] = n_vert + np.arange(len(bv))
            rings.append(ring)
            n_vert += len(bv)
    rings.append(np.arange(n) + n)
    walls = []
    for lo, hi in zip(rings[:-1], rings[1:]):
        walls.append(np.column_stack([lo[ba], lo[bb], hi[bb]]))
        walls.append(np.column_stack([lo[ba], hi[bb], hi[ba]]))
    faces = np.concatenate([m.faces[:, ::-1], m.faces + n] + walls)
    if d < 0:
        faces = faces[:, ::-1]
    return Mesh(np.concatenate(vertices), faces)
