import sys
import time

from py3do.io import read_stl
from py3do.vis import view_pyglet
from py3do.topo import repeated_face_vertices
from py3do.topo import unused_vertices
from py3do.topo import EdgeToFaceMap
from py3do import connected_components

t0 = time.time()
m = read_stl(sys.argv[1])
print("STL reading time: ", round(time.time()-t0, 4))

rep_vert = repeated_face_vertices(m)
if len(rep_vert) > 0:
    print(f"Repeated vertices in {len(rep_vert)} faces")
else:
    print("No repeated vertices")

# can't really happen in an STL, but just in case...
unu_vert = unused_vertices(m)
if len(rep_vert) > 0:
    print(f"{len(unu_vert)} unused vertices")
else:
    print("No unused vertices")

nc, _, _ = connected_components(m)
if nc > 1:
    print(f"\nModel consists of {nc} connected components\n")

efm = EdgeToFaceMap(m)
print(f"Model edges are {'' if efm.oriented else 'NOT '}correctly oriented")
print(f"Model is {'' if efm.manifold else 'NOT '}manifold")
print(f"Model is {'' if efm.watertight else 'NOT '}watertight")

view_pyglet(m)
