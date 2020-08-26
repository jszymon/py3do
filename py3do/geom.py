"""Low level geometric algorithms."""

import numpy as np

def normals_cross(m):
    """Caclulate normals using cross products."""
    fvs = m.vertices[m.faces]  # vertices of faces
    d1 = (fvs[:,1,:] - fvs[:,0,:])
    d2 = (fvs[:,2,:] - fvs[:,0,:])
    normals = np.cross(d1, d2)
    areas = np.linalg.norm(normals, axis=1)
    normals /= areas.reshape(-1,1)
    return normals, areas
    

def normals_Newell(m):
    """Caclulate normals for mesh m using Newell's method."""
    fvs = m.vertices[m.faces]  # vertices of faces
    normals = np.empty((m.faces.shape[0], 3))
    normals[:,0] =  (fvs[:,0,1] - fvs[:,1,1]) * (fvs[:,0,2] + fvs[:,1,2])
    normals[:,0] += (fvs[:,1,1] - fvs[:,2,1]) * (fvs[:,1,2] + fvs[:,2,2])
    normals[:,0] += (fvs[:,2,1] - fvs[:,0,1]) * (fvs[:,2,2] + fvs[:,0,2])
    normals[:,1] =  (fvs[:,0,2] - fvs[:,1,2]) * (fvs[:,0,0] + fvs[:,1,0])
    normals[:,1] += (fvs[:,1,2] - fvs[:,2,2]) * (fvs[:,1,0] + fvs[:,2,0])
    normals[:,1] += (fvs[:,2,2] - fvs[:,0,2]) * (fvs[:,2,0] + fvs[:,0,0])
    normals[:,2] =  (fvs[:,0,0] - fvs[:,1,0]) * (fvs[:,0,1] + fvs[:,1,1])
    normals[:,2] += (fvs[:,1,0] - fvs[:,2,0]) * (fvs[:,1,1] + fvs[:,2,1])
    normals[:,2] += (fvs[:,2,0] - fvs[:,0,0]) * (fvs[:,2,1] + fvs[:,0,1])
    # simpler computations using np.roll (2x slower):
    #rolled_fvs = np.roll(fvs, -1, axis=1)
    #normals = (np.roll(fvs - rolled_fvs, -1, axis=2) *\
    #               np.roll(fvs + rolled_fvs, 1, axis=2)).sum(axis=1)
    areas = np.linalg.norm(normals, axis=1)
    normals /= areas.reshape(-1,1)
    return normals, areas
