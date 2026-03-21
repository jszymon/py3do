import numpy as np

from py3do import cube, chamfer_bottom

def test_chamfer_cube():
    m = cube()
    m.vertices -= [[0.5, 0.5, 0.5]]
    mc = chamfer_bottom(m, 0.1)
    bot_verts = mc.vertices[mc.vertices[:,2] < -0.45]
    bot_verts = bot_verts[:,:2]
    assert np.allclose(np.abs(bot_verts), 0.45)
