import sys

from py3do import Mesh
from py3do.io import read_stl, write_binary_stl
from py3do import split_mesh

stl_name = sys.argv[1]
if stl_name[-4:] != ".stl":
    print("Filename must end in .stl")
base_name = stl_name[:-4]
m = read_stl(stl_name)
ms = split_mesh(m)
if len(ms) == 1:
    print("Connected model.  Nothing to do.")
else:
    for i, mi in enumerate(ms):
        comp_fname = base_name + "_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] + ".stl"
        write_binary_stl(mi, comp_fname)
