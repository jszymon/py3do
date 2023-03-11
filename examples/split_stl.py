import sys

import itertools as it

from py3do import Mesh
from py3do.io import read_stl, write_binary_stl
from py3do import split_mesh

s = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
suffix_iter = it.islice(it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1)), 1, None)

stl_name = sys.argv[1]
if stl_name[-4:] != ".stl":
    print("Filename must end in .stl")
base_name = stl_name[:-4]
print("Reading STL")
m = read_stl(stl_name, fix_nan_normals=True)
print("Splitting...")
ms = split_mesh(m)
if len(ms) == 1:
    print("Connected model.  Nothing to do.")
else:
    print("Writing STL")
    for i, mi in enumerate(ms):
        comp_fname = base_name + "_" + "".join(next(suffix_iter)) + ".stl"
        write_binary_stl(mi, comp_fname)
