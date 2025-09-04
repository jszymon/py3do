import sys
import argparse

import itertools as it

from py3do import Mesh
from py3do.io import read_stl, write_binary_stl
from py3do.vis import view_pyglet

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Center and drop the model along specified axes.  Optionaly rotate by multiples 90 of degrees along selected axes prior to centering.")
parser.add_argument('stlfile', help='STL file name')
parser.add_argument('-c', '--center', help='Axes to center. A subset of XYZ', default="XY")
parser.add_argument('-d', '--drop', help='Axes to drop the model on (shift minimum coordinate to zero)', default="Z")
parser.add_argument('-r', '--rotate', help='Sequence of clockwise 90 degree rotations along axis, e.g. XXY rotates 180 degrees around X axis, then 90 degrees around Y', default="")
parser.add_argument('-v', '--view', help='Show the model', action='store_true')
#parser.print_help()

args = parser.parse_args()

m = read_stl(args.stlfile, fix_nan_normals=True)

# rotate
for axis in args.rotate.upper():
    ai = "XYZ".find(axis)
    if ai < 0:
        print("Error: argument to --rotate must be a string of XYZxyz")
        parser.print_usage()
        sys.exit(1)
    a0 = 1 if ai == 0 else 0
    a1 = 1 if ai == 2 else 2
    m.vertices[:,(a0,a1)] = m.vertices[:,(a1,a0)]
    m.vertices[:,a1] = -m.vertices[:,a1]
# center
for axis in args.center.upper():
    ai = "XYZ".find(axis)
    if ai < 0:
        print("Error: argument to --center must be a string of XYZxyz")
        parser.print_usage()
        sys.exit(1)
    m.vertices[:,ai] -= m.vertices[:,ai].mean()
# drop
for axis in args.drop.upper():
    ai = "XYZ".find(axis)
    if ai < 0:
        print("Error: argument to --drop must be a string of XYZxyz")
        parser.print_usage()
        sys.exit(1)
    m.vertices[:,ai] -= m.vertices[:,ai].min()

base_name = args.stlfile[:-4]
out_fname = base_name + "_c.stl"
print("Writing STL to", out_fname)
write_binary_stl(m, out_fname)


if args.view:
    view_pyglet(m)
