import numpy as np
import matplotlib.pyplot as plt

from py3do.geom_2d import two_segment_fillet

for x0, x1, x2, y0, y1, y2 in [
        [0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 0.0, -1.0, 0.0], # CCW
        [0.0, 1.0, 2.0, 0.0, 3.0, 0.0],
        [0.0, 1.0, 2.0, 0.0, 3.0, 1.0],
        [0.0, 1.0, 2.0, 0.0, -3.0, 1.0], # CCW
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0], # collinear
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], # overlapping
        [0.0, 1.0, 0.0, 0.0, -1.0, 0.0], # overlapping
]:
    r = 0.5
    xy, (xc, yc), a = two_segment_fillet(x0, y0, x1, y1, x2, y2, r)
    ## plots
    for xx, yy in xy:
        plt.plot([xx], [yy], "ok")
    
    plt.plot([x0, x1, x2], [y0, y1, y2])
    plt.plot([xc], [yc], "o")
    plt.plot(xy[-1,0], xy[-1,1], "og")
    plt.plot(xy[0,0], xy[0,1], "or")
    plt.gca().add_patch(plt.Circle((xc, yc), r, fill=False))
    plt.show()
