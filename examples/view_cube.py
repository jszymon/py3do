from py3do import cube
from py3do.vis import view_pyglet_noblock

m = cube()
pv = view_pyglet_noblock(m, marked_vertices = [1,3,5,7], marked_edges=[[0,1]])
print("select face to highlight (0-11)")
while(True):
    try:
        face = int(input())
        assert 0 <= face < 12
    except:
        continue
    if not pv.is_active:
        break
    pv.set_model(marked_faces = [face])
    pv.show_face(face)
