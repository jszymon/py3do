"""py3do - Construct models for 3d printing in Python"""

from .mesh import Mesh
from .geom import normals_Newell
from .geom import normals_cross
from .topo import repeated_face_vertices
from .primitives import cube
from .primitives import circle
from .primitives import cone_pipe
from .binary_relations import is_isomorphic
from . import vis
from . import io
