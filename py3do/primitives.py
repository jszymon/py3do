"""Primitive solids."""

import numpy as np

from .mesh import Mesh

_cube_vertices = np.array(list(np.ndindex(2,2,2)))
_cube_faces = np.array([[0, 1, 3], [0, 2, 3],
                        [4, 5, 7], [4, 6, 7],
                        [0, 1, 5], [0, 4, 5],
                        [2, 3, 7], [2, 6, 7],
                        [0, 2, 6], [0, 4, 6],
                        [1, 3, 7], [1, 5, 7],
                       ])
_cube_normals = np.array([[-0.5, 0, 0], [-0.5, 0, 0],
                          [+0.5, 0, 0], [+0.5, 0, 0],
                          [0, -0.5, 0], [0, -0.5, 0],
                          [0, +0.5, 0], [0, +0.5, 0],
                          [0, 0, -0.5], [0, 0, -0.5],
                          [0, 0, +0.5], [0, 0, +0.5],
                         ])

def cube():
    c = Mesh(_cube_vertices, _cube_faces, normals=_cube_normals)
    return c
