from py3do import cube
from py3do.vis import view_pyglet

m = cube()
view_pyglet(m, marked_faces = [0,1], marked_vertices = [1,3,5,7], marked_edges=[[0,1]])
