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
                offset_vecs=None, fix_boundary=True):
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
    vectors, otherwise vertex normals are used.  fix_boundary is
    passed to vertex_normals, it improves offset directions at
    boundaries of open meshes.

    """
    if method == "vertex normals":
        if offset_vecs is None:
            vn = vertex_normals(m, "angle weighted", normalize=False,
                                fix_boundary=fix_boundary)
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

def _boundary_loops(a, b):
    """Order directed boundary edges a -> b into cycles.

    Returns a list of vertex index arrays."""
    nxt = dict(zip(a.tolist(), b.tolist()))
    if len(nxt) != len(a):
        raise ValueError("shell: boundary is not manifold")
    loops = []
    seen = set()
    for v in a.tolist():
        if v in seen:
            continue
        cycle = [v]
        seen.add(v)
        x = nxt[v]
        while x != v:
            if x in seen:
                raise ValueError("shell: boundary is not manifold")
            cycle.append(x)
            seen.add(x)
            x = nxt[x]
        loops.append(np.array(cycle))
    return loops

def _clip_beyond_plane(v, faces, start, c, p, eps, tag, tags):
    """Cut away the part of surface (v, faces) beyond plane through c
    with normal p which is connected to start vertices.

    Cut points get tag in tags.  Returns new v, faces, tags.
    Removed vertices are left unused.

    """
    dist = (v - c) @ p
    beyond = dist > eps
    marked = np.zeros(len(v), dtype=bool)
    marked[start[beyond[start]]] = True
    if not marked.any():
        return v, faces, tags
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    e = e[beyond[e[:, 0]] & beyond[e[:, 1]]]
    while True:
        grow = e[marked[e[:, 0]] & ~marked[e[:, 1]], 1]
        if len(grow) == 0:
            break
        marked[grow] = True
    fm = marked[faces]
    new_faces = [faces[~fm.any(axis=1)]]
    new_v = []
    cuts = {}
    def cut(x, y):
        """Index of the point where edge x--y crosses the plane."""
        kept = y if marked[x] else x
        if abs(dist[kept]) <= eps:
            return kept
        key = (min(x, y), max(x, y))
        if key not in cuts:
            t = dist[x] / (dist[x] - dist[y])
            new_v.append(v[x] + t * (v[y] - v[x]))
            cuts[key] = len(v) + len(new_v) - 1
        return cuts[key]
    for f in faces[fm.any(axis=1) & ~fm.all(axis=1)]:
        poly = []
        for k in range(3):
            x, y = f[k], f[(k + 1) % 3]
            if not marked[x]:
                poly.append(x)
            if marked[x] != marked[y]:
                poly.append(cut(x, y))
        poly = [q for i, q in enumerate(poly) if q != poly[i - 1]]
        for i in range(1, len(poly) - 1):
            new_faces.append(np.array([[poly[0], poly[i], poly[i + 1]]]))
    if new_v:
        v = np.concatenate([v, new_v])
        tags = np.concatenate([tags, np.full(len(new_v), tag)])
    return v, np.concatenate(new_faces), tags

def _closest_on_polyline(x, poly):
    """Points of closed polyline poly closest to each point in x."""
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    ax = x[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", ax, ab)
                / np.maximum(np.einsum("jk,jk->j", ab, ab), 1e-300), 0, 1)
    q = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    j = np.argmin(np.linalg.norm(q - x[:, None, :], axis=2), axis=1)
    return q[np.arange(len(x)), j]

def _zip_cycles(pts, lo, hi):
    """Triangulate the strip between closed cycles lo and hi.

    Both cycles must run in the same direction.  Diagonals are chosen
    greedily, always taking the shorter one."""
    nl, nh = len(lo), len(hi)
    j0 = np.argmin(np.linalg.norm(pts[hi] - pts[lo[0]], axis=1))
    hi = np.roll(hi, -j0)
    tris = []
    i = j = 0
    while i < nl or j < nh:
        li, li1 = lo[i % nl], lo[(i + 1) % nl]
        hj, hj1 = hi[j % nh], hi[(j + 1) % nh]
        if j >= nh or (i < nl and np.linalg.norm(pts[li1] - pts[hj])
                       <= np.linalg.norm(pts[li] - pts[hj1])):
            tris.append((li, li1, hj))
            i += 1
        else:
            tris.append((li, hj1, hj))
            j += 1
    return np.array(tris)

def _half_round(v0, v1, ob, n_round, n_vert):
    """Vertices of half rounds between v0 and v1 bulging towards ob.

    Returns a list of n_round-1 vertex arrays (rings) and a list of
    their starting indices, beginning with n_vert."""
    c = (v0 + v1) / 2
    u = v0 - c
    r = np.linalg.norm(u, axis=1, keepdims=True)
    o = ob - np.einsum("ij,ij->i", ob, u).reshape(-1, 1) * u / r**2
    o /= np.linalg.norm(o, axis=1, keepdims=True)
    rings, starts = [], []
    for k in range(1, n_round):
        t = np.pi * k / n_round
        rings.append(c + np.cos(t) * u + np.sin(t) * r * o)
        starts.append(n_vert)
        n_vert += len(v0)
    return rings, starts

def _inset_shell(m, v1, disp, ba, bb, bv, ob, round, n_round):
    """Faces and vertices of the shell for edge="inset".

    The offset surface v1 is extended or clipped at the plane of each
    boundary loop, the side walls connect the loops with the rims of
    the offset surface."""
    n = m.vertices.shape[0]
    v0 = m.vertices
    eps = 1e-9 * max(np.ptp(v0, axis=0).max(), 1)
    # sliding direction: outward direction made perpendicular to the
    # offset displacement
    dn = disp / np.linalg.norm(disp, axis=1, keepdims=True)
    t = np.zeros_like(v0)
    t[bv] = ob - np.einsum("ij,ij->i", ob, dn[bv]).reshape(-1, 1) * dn[bv]
    t[bv] /= np.linalg.norm(t[bv], axis=1, keepdims=True)
    loops = _boundary_loops(ba, bb)
    v = v1.copy()
    faces = m.faces.copy()
    tags = np.full(n, -1)
    target = v1.copy()  # estimated ends of the side walls
    for l, loop in enumerate(loops):
        tags[loop] = l
        c = v0[loop].mean(axis=0)
        p = np.linalg.svd(v0[loop] - c)[2][-1]  # plane normal
        tp = t[loop] @ p
        if np.abs(tp).mean() < 1e-3:
            continue  # offset surface parallel to the plane
        if tp.mean() < 0:
            p, tp = -p, -tp
        dist = (v[loop] - c) @ p
        ok = np.abs(tp) > 1e-6
        s = np.where(ok, -dist / np.where(ok, tp, 1), 0)
        target[loop] += s.reshape(-1, 1) * t[loop]
        # extend offset surface if it ends before the plane
        short = loop[s > 0]
        v[short] = target[short]
        # clip it if it extends beyond
        v, faces, tags = _clip_beyond_plane(v, faces, loop, c, p, eps,
                                            l, tags)
    # rims of the offset surface
    ra, rb, _ = _boundary_edges(Mesh(v, faces, fix_nan_normals=True))
    rims = [None] * len(loops)
    for rim in _boundary_loops(ra, rb):
        rt = np.unique(tags[rim])
        rt = rt[rt >= 0]
        if len(rt) != 1 or rims[rt[0]] is not None:
            raise ValueError("shell: inset clips the offset surface between "
                             "boundary loops away, reduce thickness")
        rims[rt[0]] = rim
    if any(r is None for r in rims):
        raise ValueError("shell: inset clipped away the offset surface "
                         "at a boundary loop, reduce thickness")
    # drop unused offset vertices
    used = np.zeros(len(v), dtype=bool)
    used[faces] = True
    new_idx = np.full(len(v), -1)
    new_idx[used] = n + np.arange(used.sum())
    vertices = [v0, v[used]]
    faces = new_idx[faces]
    rims = [new_idx[r] for r in rims]
    n_vert = n + used.sum()
    # profile rings, rings[k][i] is the index of k-th edge profile
    # vertex of boundary vertex i
    rings = [np.arange(n)]
    if round:
        pts = np.concatenate(vertices)
        ends = np.zeros_like(v0)
        for loop, rim in zip(loops, rims):
            ends[loop] = _closest_on_polyline(target[loop], pts[rim])
        rv, starts = _half_round(v0[bv], ends[bv], ob, n_round, n_vert)
        for x, st in zip(rv, starts):
            vertices.append(x)
            ring = np.full(n, -1)
            ring[bv] = st + np.arange(len(bv))
            rings.append(ring)
    pts = np.concatenate(vertices)
    walls = []
    for lo, hi in zip(rings[:-1], rings[1:]):
        walls.append(np.column_stack([lo[ba], lo[bb], hi[bb]]))
        walls.append(np.column_stack([lo[ba], hi[bb], hi[ba]]))
    for loop, rim in zip(loops, rims):
        walls.append(_zip_cycles(pts, rings[-1][loop], rim))
    return pts, faces, walls

def shell(m, d, *, edge="flat", round=False, n_round=8,
          offset_vecs=None, fix_boundary=True):
    """Add thickness d to a flat (open surface) mesh m.

    The mesh is offset by d using offset_mesh (in the direction of
    face normals for d > 0, opposite for d < 0) and boundary edges of
    m are connected with their offset copies, which results in a
    closed solid.  Returns a new mesh.

    edge determines how the boundary of the shell is formed:

    "flat": each boundary edge is connected to its offset copy by two
    triangles.

    "inset": the offset surface ends at the plane of each boundary
    loop, so the side walls lie in that plane.  E.g. a dome cut at
    z=c gets a horizontal rim at z=c.  If the offset surface ends
    before the plane, its boundary vertices slide along it
    (perpendicular to the boundary) into the plane.  If it extends
    beyond the plane, the part beyond connected to the boundary is cut
    away.  For non planar loops the best fitting plane is used.
    Loops whose offset surface is parallel to the plane are treated
    as in "flat", so this is identical to "flat" for planar meshes.

    If round is True, the side wall (of either edge type) is replaced
    by a half round of diameter equal to the wall's length, made of
    n_round segments.  It bulges outward from the original boundary.

    offset_vecs and fix_boundary are passed on to offset_mesh.

    """
    if edge not in ("flat", "inset"):
        raise ValueError("shell: unknown edge type " + repr(edge))
    if d == 0:
        raise ValueError("shell: thickness must be nonzero")
    if round and n_round < 1:
        raise ValueError("shell: n_round must be at least 1")
    n = m.vertices.shape[0]
    v0 = m.vertices
    disp = offset_mesh(m, d, offset_vecs=offset_vecs,
                       fix_boundary=fix_boundary).vertices - v0
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
    v1 = v0 + disp
    if edge == "inset":
        vertices, offset_faces, walls = _inset_shell(
            m, v1, disp, ba, bb, bv, ob, round, n_round)
        faces = np.concatenate([m.faces[:, ::-1], offset_faces] + walls)
        if d < 0:
            faces = faces[:, ::-1]
        return Mesh(vertices, faces)
    vertices = [v0, v1]
    # rings[k][i] is the index of k-th edge profile vertex of
    # boundary vertex i
    rings = [np.arange(n)]
    if round:
        rv, starts = _half_round(v0[bv], v1[bv], ob, n_round, 2 * n)
        for x, st in zip(rv, starts):
            vertices.append(x)
            ring = np.full(n, -1)
            ring[bv] = st + np.arange(len(bv))
            rings.append(ring)
    rings.append(np.arange(n) + n)
    walls = []
    for lo, hi in zip(rings[:-1], rings[1:]):
        walls.append(np.column_stack([lo[ba], lo[bb], hi[bb]]))
        walls.append(np.column_stack([lo[ba], hi[bb], hi[ba]]))
    faces = np.concatenate([m.faces[:, ::-1], m.faces + n] + walls)
    if d < 0:
        faces = faces[:, ::-1]
    return Mesh(np.concatenate(vertices), faces)
