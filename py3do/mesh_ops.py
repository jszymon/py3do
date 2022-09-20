"""Operations on meshes."""

import numpy as np

from .mesh import Mesh
from .topo import connected_components


def split_mesh(m):
    """Split mesh into connected component.  Returns each component as a
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
