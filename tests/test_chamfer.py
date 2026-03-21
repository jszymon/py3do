import numpy as np

from py3do import cube, chamfer_bottom

def test_chamfer_cube():
    m = cube()
    m.vertices -= [[0.5, 0.5, 0.5]]
    bot_verts_orig = m.vertices[m.vertices[:,2] < -0.45]
    bot_verts_orig = bot_verts_orig[:,:2]
    mc = chamfer_bottom(m, 0.1)
    bot_verts = mc.vertices[mc.vertices[:,2] < -0.45]
    bot_verts = bot_verts[:,:2]
    dists = np.linalg.norm(bot_verts-bot_verts_orig, axis=1)
    assert np.allclose(dists, 0.1)
