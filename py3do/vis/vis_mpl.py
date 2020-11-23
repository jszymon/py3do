import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def view_matplotlib(m, *args, **kwargs):
    fvs = m.vertices[m.faces]  # vertices of faces
    bmin = fvs.min(axis=(0,1))
    bmax = fvs.max(axis=(0,1))
    center = (bmax+bmin)/2
    r = (bmax-bmin).max()
    c = Poly3DCollection(fvs, *args, **kwargs)
    c.set_edgecolor("black")
    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.add_collection3d(c)
    ax.set_xlim3d(center[0]-r, center[0]+r)
    ax.set_ylim3d(center[1]-r, center[1]+r)
    ax.set_zlim3d(center[2]-r, center[2]+r)
    plt.show()
