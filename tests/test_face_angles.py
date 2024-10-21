import numpy as np

from py3do import cube, cone_pipe
from py3do import face_angles

def test_cube_angles():
    m = cube()
    a = face_angles(m)
    # each face has two pi/4 angles and one pi/2 angle
    assert (np.isclose(a, np.pi/2).sum(axis=1) == 1).all()
    assert (np.isclose(a, np.pi/4).sum(axis=1) == 2).all()

def _test_cylinder_angles(n):
    m = cone_pipe(1,0,1,1, n=n, close_bottom=True, close_top=True)
    fvs = m.vertices[m.faces]  # vertices of faces
    top_idx = (fvs[:,:,2] > 0.9).all(axis=1)
    bot_idx = (fvs[:,:,2] < 0.1).all(axis=1)
    top_bot_idx = (top_idx | bot_idx)
    side_idx = ~(top_bot_idx)

    a = face_angles(m)

    assert (np.isclose(a[top_bot_idx], 2*np.pi / n).sum(axis=1) == 1).all()
    assert (np.isclose(a[top_bot_idx], np.pi * (1/2 - 1/n)).sum(axis=1) == 2).all()
    assert (np.isclose(a[side_idx], np.pi/2).sum(axis=1) == 1).all()
    side_ang1 = np.arctan(2*np.sin(np.pi/n))
    side_ang2 = np.pi/2 - np.arctan(2*np.sin(np.pi/n))
    assert (np.isclose(a[side_idx], side_ang1).sum(axis=1) == 1).all()
    assert (np.isclose(a[side_idx], side_ang2).sum(axis=1) == 1).all()

def test_cylinder_angles():
    _test_cylinder_angles(3)
    _test_cylinder_angles(10)
    _test_cylinder_angles(100)
