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

changed=False
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
    # find vertices of the shortest edge
    ei = a[fi].argmin()
    edge_verts = [0,1,2]
    edge_verts.remove(ei)
    edge_verts = m.faces[fi][edge_verts]
    print(edge_verts)
    pv.set_model(marked_faces=[fi], marked_edges=[edge_verts], marked_vertices=[edge_verts[0]])
    pv.show_face(fi)
    print("Delete edge?\n1. No\n2. Yes - keep marked vertex\n3. Yes - keep unmarked vertex")
    while True:
        ans = input()
        try:
            ans = int(ans)
            assert 1 <= ans <= 3
        except:
            print(f"Please answer 1-3")
            continue
        break
    if ans == 1:
        continue
    changed=True
    if ans == 2:
        m.delete_edge(edge_verts[1], edge_verts[0])
    if ans == 3:
        m.delete_edge(edge_verts[0], edge_verts[1])
    pv.set_model(m, marked_faces=[], marked_edges=[], marked_vertices=[])

if changed:
    print(f"Save model as {sys.argv[2]} (Y/N)?")
    while True:
        ans = input().lower()
        try:
            assert ans in ["y", "n"]
        except:
            print(f"Please answer Y/N")
            continue
        break
    if ans == "y":
        write_binary_stl(m, sys.argv[2])
