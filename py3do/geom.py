"""Low level geometric algorithms."""

import numpy as np

def normals_cross():
    """Caclulate normals using cross products."""
    pass

def normals_Newell(m):
    """Caclulate normals for mesh m using Newell's method."""
    fps = m.vertices[m.faces]  # points of faces
    normals = np.empty((m.faces.shape[0], 3))
    normals[:,0] =  (fps[:,0,1] - fps[:,1,1]) * (fps[:,0,2] + fps[:,1,2])
    normals[:,0] += (fps[:,1,1] - fps[:,2,1]) * (fps[:,1,2] + fps[:,2,2])
    normals[:,0] += (fps[:,2,1] - fps[:,0,1]) * (fps[:,2,2] + fps[:,0,2])
    normals[:,1] =  (fps[:,0,2] - fps[:,1,2]) * (fps[:,0,0] + fps[:,1,0])
    normals[:,1] += (fps[:,1,2] - fps[:,2,2]) * (fps[:,1,0] + fps[:,2,0])
    normals[:,1] += (fps[:,2,2] - fps[:,0,2]) * (fps[:,2,0] + fps[:,0,0])
    normals[:,2] =  (fps[:,0,0] - fps[:,1,0]) * (fps[:,0,1] + fps[:,1,1])
    normals[:,2] += (fps[:,1,0] - fps[:,2,0]) * (fps[:,1,1] + fps[:,2,1])
    normals[:,2] += (fps[:,2,0] - fps[:,0,0]) * (fps[:,2,1] + fps[:,0,1])
    areas = np.linalg.norm(normals, axis=1)
    normals /= areas.reshape(-1,1)
    return normals, areas
