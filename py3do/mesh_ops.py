"""Operations on meshes."""

import numpy as np

from .mesh import Mesh
from .geom import vec_angle, vertex_normals
from .topo import connected_components
from .slice import slice_horiz_0

def split_mesh(m):
    """Split mesh into connected components.  Returns each component as a
    mesh.

    uses connected_components and renumbers face vertices
    appropriately."""
    n = m.vertices.shape[0]
    nc, component_vertices, component_faces = connected_components(m)
    if nc == 1:
        return [m]
    # mapping for face vertices
    face_vertex_map = np.arange(n)
    for cv in component_vertices:
        face_vertex_map[cv] = np.arange(len(cv))
    component_meshes = []
    for i in range(nc):
        mi = Mesh(m.vertices[component_vertices[i]],
                  face_vertex_map[m.faces[component_faces[i]]],
                  normals = m.normals[component_faces[i]]
                  )
        component_meshes.append(mi)
    return component_meshes

def chamfer_bottom(m, h, d=None, min_angle=15):
    """Chamfer the bottom of the model.

    This function is meant to help avoid elephant's foot, not get a
    mathematically correct chamfer.  Chamfer width is not constant,
    but all edge points will be shifted by exactly d.

    Chamfer starts at z=h and its depth at bottom is d.  Chamfer
    depths for points not at bottom are proportional to their z
    coordinate, it is 0 for z>=h.  

    If d is none, assume d=h.  Points whose vertex normal's angle with
    the vector (0,0,-1), i.e. reversed z axis is less than min_angle
    are not chamfered.

    Most predictable results are for flat bottom surrounded by
    vertical faces.

    """
    if d is None:
        d = h
    min_angle = np.deg2rad(min_angle)
    min_z = m.vertices[:,2].min()
    m.vertices[:,2] -= min_z
    m.vertices[:,2] -= h

    m = slice_horiz_0(m, keep="both")
    mask = (m.vertices[:,2] < 0)
    v_normals = vertex_normals(m, method="angle weighted")[mask]
    angles_to_z = vec_angle([0,0,1], v_normals)
    mask2 = (angles_to_z > min_angle)
    mask[mask] = mask2
    v_normals = v_normals[mask2]
    v_normals[:,2] = 0  # shift only x and y corrdinates
    # shift proportionally to z
    v_normals *= (-m.vertices[mask][:,2] / h).reshape(-1, 1)
    v_normals /= np.linalg.norm(v_normals, axis=1).reshape(-1,1)
    m.vertices[mask] -= v_normals * d

    m.vertices[:,2] += h
    m.vertices[:,2] += min_z
    return m
