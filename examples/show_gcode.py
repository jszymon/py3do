import sys

from py3do.gcode import GCode

g = GCode(sys.argv[1])
c = [(de < 0)*1 for de in g.des]


import matplotlib.pyplot as plt
ax = plt.figure().add_subplot(projection='3d')
ax.plot(g.pos[:,0], g.pos[:,1], g.pos[:,2])
plt.show()
