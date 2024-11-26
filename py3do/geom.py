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

def vertex_normals(m, method="average", normalize=True):
    """Compute vertex normals based on face normals.

    Available methods are
    * 'average': average of normals of all adjacent faces
    * 'area weighted': as above but weighted by face area
    * 'angle weighted': as above but weighted by face angle adjacent to vertex
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
