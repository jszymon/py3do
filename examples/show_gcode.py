import sys

import numpy as np

from matplotlib.collections import LineCollection

from py3do.gcode import GCode

g = GCode(sys.argv[1])
#clr = [(de < 0)*1 for de in g.des]
clr = [[1,0,0,1] if (de <= 0) else [0,1,0,1] for de in g.des]


# 2d plot with slider

import matplotlib.pyplot as plt


class GCodePlot2D:
    def __init__(self, g):
        self.g = g
        self.pos = np.array(g.pos)
    def plot_layer(g, l):
        xyz = self.pos[g.layers[l]:g.layers[l+1]]
        xy = xyz[:,[0,1]]
        z = xyz[:,2]
        points = xy.reshape(-1, 1, 2)
        segments = np.concatenate([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(segments, colors=clr)
        lc.set_linewidth(2)
        
# 3d plot

if False:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    points = np.array(g.pos).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, colors=clr)
    lc.set_linewidth(2)

    ax = plt.figure().add_subplot(projection='3d')
    ax.add_collection3d(lc)#, zs=z, zdir='z')
    ax.set_xlim(xmin=g.pos[:,0].min(), xmax=g.pos[:,0].max())
    ax.set_ylim(ymin=g.pos[:,1].min(), ymax=g.pos[:,1].max())
    ax.set_zlim(zmin=g.pos[:,2].min(), zmax=g.pos[:,2].max())
    plt.show()
