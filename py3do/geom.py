"""Low level geometric algorithms."""

import numpy as np

from .topo import sorted_edges

def normals_cross(m):
    """Caclulate normals using cross products."""
    fvs = m.vertices[m.faces]  # vertices of faces
    d1 = (fvs[:,1,:] - fvs[:,0,:])
    d2 = (fvs[:,2,:] - fvs[:,0,:])
    normals = np.cross(d1, d2)
    areas = np.linalg.norm(normals, axis=1)
    normals /= areas.reshape(-1,1)
    return normals, areas / 2
    

def normals_Newell(m):
    """Caclulate normals for mesh m using Newell's method."""
    fvs = m.vertices[m.faces]  # vertices of faces
    normals = np.empty((m.faces.shape[0], 3))
    normals[:,0] =  (fvs[:,0,1] - fvs[:,1,1]) * (fvs[:,0,2] + fvs[:,1,2])
    normals[:,0] += (fvs[:,1,1] - fvs[:,2,1]) * (fvs[:,1,2] + fvs[:,2,2])
    normals[:,0] += (fvs[:,2,1] - fvs[:,0,1]) * (fvs[:,2,2] + fvs[:,0,2])
    normals[:,1] =  (fvs[:,0,2] - fvs[:,1,2]) * (fvs[:,0,0] + fvs[:,1,0])
    normals[:,1] += (fvs[:,1,2] - fvs[:,2,2]) * (fvs[:,1,0] + fvs[:,2,0])
    normals[:,1] += (fvs[:,2,2] - fvs[:,0,2]) * (fvs[:,2,0] + fvs[:,0,0])
    normals[:,2] =  (fvs[:,0,0] - fvs[:,1,0]) * (fvs[:,0,1] + fvs[:,1,1])
    normals[:,2] += (fvs[:,1,0] - fvs[:,2,0]) * (fvs[:,1,1] + fvs[:,2,1])
    normals[:,2] += (fvs[:,2,0] - fvs[:,0,0]) * (fvs[:,2,1] + fvs[:,0,1])
    # simpler computations using np.roll (2x slower):
    #rolled_fvs = np.roll(fvs, -1, axis=1)
    #normals = (np.roll(fvs - rolled_fvs, -1, axis=2) *\
    #               np.roll(fvs + rolled_fvs, 1, axis=2)).sum(axis=1)
    areas = np.linalg.norm(normals, axis=1)
    normals /= areas.reshape(-1,1)
    return normals, areas / 2

def vec_angle(a, b):
    """Angle between vectors a and b.

    If dim a = dim b > 1 compute angles along last axis.  Use method
    from https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf

    """
    la = np.linalg.norm(a, axis=-1, keepdims=True)
    lb = np.linalg.norm(b, axis=-1, keepdims=True)
    y = np.linalg.norm(a*lb - b*la, axis=-1)
    x = np.linalg.norm(a*lb + b*la, axis=-1)
    return 2*np.arctan2(y, x)

def face_angles(m):
    """Angles in faces adjacent to each vertex.

    use method from https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf"""
    fvs = m.vertices[m.faces]  # vertices of faces
    d1 = (fvs[:,1,:] - fvs[:,0,:])
    l1 = np.linalg.norm(d1, axis=1).reshape(-1,1)
    d2 = (fvs[:,2,:] - fvs[:,1,:])
    l2 = np.linalg.norm(d2, axis=1).reshape(-1,1)
    d3 = (fvs[:,0,:] - fvs[:,2,:])
    l3 = np.linalg.norm(d3, axis=1).reshape(-1,1)
    #2arctan2( ||x·||y|| – ||x||·y||/||x·||y|| + ||x||·y|| )
    angles0 = 2 * np.arctan2(np.linalg.norm(d1*l3 + d3*l1, axis=1),
                             np.linalg.norm(d1*l3 - d3*l1, axis=1))
    angles1 = 2 * np.arctan2(np.linalg.norm(d1*l2 + d2*l1, axis=1),
                             np.linalg.norm(d1*l2 - d2*l1, axis=1))
    angles2 = 2 * np.arctan2(np.linalg.norm(d2*l3 + d3*l2, axis=1),
                             np.linalg.norm(d2*l3 - d3*l2, axis=1))
    return np.column_stack([angles0, angles1, angles2])

def _fix_boundary_normals(m, v_normals):
    """Recompute normals of boundary vertices of m in place.

    Normals of boundary vertices computed from adjacent faces are
    biased towards the interior, since all faces lie on one side.
    Instead, normals of interior vertices w within two edges of a
    boundary vertex v are reflected across the plane perpendicular to
    v--w (exact if v and w lie on a circular arc) and averaged with
    weights |v-w|^2 to reduce the influence of short edges.  Lengths
    of v_normals are preserved.  Boundary vertices with no interior
    vertices nearby are left unchanged.

    """
    n = m.vertices.shape[0]
    edges, counts = np.unique(sorted_edges(m, unique=False), axis=0,
                              return_counts=True)
    is_b = np.zeros(n, dtype=bool)
    is_b[edges[counts == 1]] = True
    if not is_b.any():
        return v_normals
    # directed edges sorted by start vertex
    e = np.concatenate([edges, edges[:, ::-1]])
    e = e[np.argsort(e[:, 0], kind="stable")]
    starts = np.searchsorted(e[:, 0], np.arange(n + 1))
    # pairs (boundary vertex, vertex within two edges)
    p1 = e[is_b[e[:, 0]]]
    cnt = starts[p1[:, 1] + 1] - starts[p1[:, 1]]
    rows = np.repeat(starts[p1[:, 1]], cnt) + (np.arange(cnt.sum())
                         - np.repeat(np.cumsum(cnt) - cnt, cnt))
    p2 = np.column_stack([np.repeat(p1[:, 0], cnt), e[rows, 1]])
    pairs = np.unique(np.concatenate([p1, p2]), axis=0)
    pairs = pairs[~is_b[pairs[:, 1]]]
    v, w = pairs[:, 0], pairs[:, 1]
    u = m.vertices[w] - m.vertices[v]
    l = np.linalg.norm(u, axis=1, keepdims=True)
    u /= l
    nw = v_normals[w] / np.linalg.norm(v_normals[w], axis=1, keepdims=True)
    r = nw - 2 * np.einsum("ij,ij->i", nw, u).reshape(-1, 1) * u
    acc = np.zeros_like(v_normals)
    np.add.at(acc, v, r * l**2)
    fix = np.unique(v)
    acc = acc[fix] / np.linalg.norm(acc[fix], axis=1, keepdims=True)
    v_normals[fix] = acc * np.linalg.norm(v_normals[fix], axis=1,
                                          keepdims=True)
    return v_normals

def vertex_normals(m, method="angle weighted", normalize=True,
                   fix_boundary=False):
    """Compute vertex normals based on face normals.

    Available methods are
    * 'average': average of normals of all adjacent faces
    * 'area weighted': as above but weighted by face area
    * 'angle weighted': as above but weighted by face angle adjacent to vertex

    If fix_boundary is True, normals of boundary vertices of open
    meshes are estimated from normals of nearby interior vertices,
    see _fix_boundary_normals.  This gives more accurate normals at
    boundaries of curved surfaces.
    """
    if method in ["average", "area weighted", "angle weighted"]:
        v_normals = np.zeros_like(m.vertices)
        if method == "average":
            f_normals0 = f_normals1 = f_normals2 = m.normals
            den0 = den1 = den2 = 1
        elif  method == "area weighted":
            f_normals, areas = normals_cross(m)
            den0 = den1 = den2 = areas
            f_normals *= den0.reshape(-1,1)
            f_normals0 = f_normals1 = f_normals2 = f_normals
        else: # angle weighted
            f_normals = m.normals
            angles = face_angles(m)
            den0 = angles[:,0]
            f_normals0 = f_normals * den0.reshape(-1,1)
            den1 = angles[:,1]
            f_normals1 = f_normals * den1.reshape(-1,1)
            den2 = angles[:,2]
            f_normals2 = f_normals * den2.reshape(-1,1)
        np.add.at(v_normals, m.faces[:,0], f_normals0)
        np.add.at(v_normals, m.faces[:,1], f_normals1)
        np.add.at(v_normals, m.faces[:,2], f_normals2)
        if normalize:
            v_normals /= np.linalg.norm(v_normals, axis=1).reshape(-1,1)
        else:
            denoms = np.zeros(m.vertices.shape[0])
            np.add.at(denoms, m.faces[:,0], den0)
            np.add.at(denoms, m.faces[:,1], den1)
            np.add.at(denoms, m.faces[:,2], den2)
            v_normals /= denoms.reshape(-1,1)
    else:
        raise NotImplemented("vertex normals method '" + method + "' not implemented")
    if fix_boundary:
        _fix_boundary_normals(m, v_normals)
    return v_normals

def edge_lengths(m):
    """Return a list of unique edges and an array of their
    corresponding lengths."""
    edges = sorted_edges(m, unique=True)
    evs = m.vertices[edges]
    d = np.linalg.norm(evs[:,1,:] - evs[:,0,:], axis=-1)
    return edges, d

def volume(m):
    """Volume of mesh m.

    For unbounded polytopes (with flipped normals) volume will be
    negative."""
    n, a = normals_cross(m)
    n *= a.reshape(-1,1)
    vs = m.vertices[m.faces[:,0]] # coords of first vertex of every face
    return np.vdot(vs, n) / 3
def COG(m):
    """Center of gravity of mesh m.

    """
    n, a = normals_cross(m)
    n *= a.reshape(-1,1)
    vs = m.vertices[m.faces[:,0]] # coords of first vertex of every face
    V = np.vecdot(vs, n) / 3 # volumes of face tetrahedrons
    c = m.vertices[m.faces].sum(axis=1) / 4 # centers of tetrahedra based on faces
    return (V @ c) / V.sum() # devide by volume

def cart2sph(v):
    """Cartesian to spherical coordinates.

    Input should be an n x 3 array."""
    r = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = vec_angle(v, [0,0,1])
    # below works for vectors and scalars:
    phi = np.arctan2(np.take(v, 1, axis=-1),
                     np.take(v, 0, axis=-1))
    return r, theta, phi

def sph2cart(r, theta, phi):
    """Spherical to cartesian coordinates.

    Input should be an n x 3 array."""
    v = np.empty((len(r), 3), dtype=float)
    st = np.sin(theta)
    v[:,0] = st * np.cos(phi)
    v[:,1] = st * np.sin(phi)
    v[:,2] = np.cos(theta)
    return v * r.reshape(-1, 1)
