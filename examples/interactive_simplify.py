"""Interactive simplification of a mesh.

Currently facilitates removal of needle like triangles."""

import sys

import numpy as np

from py3do.io import read_stl, write_binary_stl
from py3do import face_angles
from py3do.vis import view_pyglet_noblock

assert len(sys.argv) == 3

m = read_stl(sys.argv[1])

pv = view_pyglet_noblock(m)

k = 15
while True:
    a = face_angles(m)
    am = a.min(axis=1)
    ai = am.argsort()
    for i in range(k):
        idx = ai[i]
        print(f"{i:2}: min. face angle={np.rad2deg(am[idx]):.2f}")
    print("Select face ('q' to quit): ", end="")
    fi = input()
    if fi == "q" or fi == "Q":
        pv.close()
        break
    try:
        fi = int(fi)
        assert 0 <= fi < k
    except:
        print(f"Face must be an integer between 0 and {k-1}")
        continue
    fi = ai[fi]
    pv.set_model(marked_faces=[fi])
    pv.show_face(fi)
    # find shortest edge
    ei = a[fi].argmin()
