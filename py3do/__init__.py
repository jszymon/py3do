"""py3do - Construct models for 3d printing in Python"""

from .mesh import Mesh
from .geom import normals_Newell
from .geom import normals_cross
from .geom import vec_angle
from .geom import face_angles
from .geom import vertex_normals
from .geom import edge_lengths
from .geom import volume
from .topo import repeated_face_vertices
from .topo import sorted_edges
from .topo import EdgeToFaceMap
from .topo import connected_components
from .primitives import cube
from .primitives import circle
from .primitives import cone_pipe
from .binary_relations import is_isomorphic
from .mesh_ops import split_mesh
from .shell import offset_mesh
from . import vis
from . import io
from . import gcode
