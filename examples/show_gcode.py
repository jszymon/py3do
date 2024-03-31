import sys

import numpy as np

from py3do.gcode import GCode

g = GCode(sys.argv[1])
#clr = [(de < 0)*1 for de in g.des]
clr = [[1,0,0,1] if (de <= 0) else [0,1,0,1] for de in g.des]


# 2d plot with slider

# 3d plot

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
