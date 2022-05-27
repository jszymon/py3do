import sys

from py3do.io import read_stl
from py3do.vis import view_pyglet
from py3do.topo import repeated_face_vertices
from py3do.topo import EdgeToFaceMap

m = read_stl(sys.argv[1])

rep_vert = repeated_face_vertices(m)
if len(rep_vert) > 0:
    print(f"Repeated vertices in {len(rep_vert)} faces")
else:
    print("No repeated vertices")

efm = EdgeToFaceMap(m)
print(f"Model is {'' if efm.watertight else 'NOT '}watertight")

view_pyglet(m)
