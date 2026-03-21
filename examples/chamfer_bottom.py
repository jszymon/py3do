import sys
import argparse

from py3do import cube, chamfer_bottom
from py3do.io import read_stl, write_binary_stl

from py3do.vis import view_pyglet

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Chamfer model bottom face.")
parser.add_argument('stlfile', help='STL file name')
parser.add_argument('height', help='Height at which chamfer should start')
parser.add_argument('-d', '--width', help='Width of chamfer at the bottom (default equal toheight)', default="")
parser.add_argument('-v', '--view', help='Show the model', action='store_true')
#parser.print_help()

args = parser.parse_args()
h = float(args.height)
if args.width == "":
    d = None
else:
    d = float(args.d)

m = read_stl(args.stlfile)
mc = chamfer_bottom(m, h, d=d)
if args.view:
    view_pyglet(mc)
