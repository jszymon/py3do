"""Operations on meshes."""

import numpy as np

from .mesh import Mesh
from .geom import vec_angle, vertex_normals
from .topo import connected_components
from .slice import slice_horiz_0
from .shell import offset_mesh

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

def chamfer_bottom(m, h, d=None, eps=1e-4):
    """Chamfer the bottom of the model.

    Chamfer starts at z=h and its width at bottom is d.  Points within
    eps of the lowest z coordinate are considered the bottom face.

    If d is none, assume d=h.

    Most predictable results are for flat bottom surrounded by
    vertical faces.

    """
    if d is None:
        d = h
    min_z = m.vertices[:,2].min()
    m.vertices[:,2] -= min_z
    m.vertices[:,2] -= h
    m.vertices[:,2] += h
    m = slice_horiz_0(m, keep="both")
    mask = (m.vertices[:,2] <= eps)
    if np.any((m.vertices[:,2] > eps) & (m.vertices[:,2] < h)):
        print("Warning (chamfer_bottom): points are present between bottom face and chamfer hight.  This may give unexpected results")
    subm, map_submesh_mesh = m.get_submesh(mask)
    subm = offset_mesh(subm, -d)
    m.set_submesh_vertices(subm, map_submesh_mesh)

    m.vertices[:,2] += min_z
    return m
