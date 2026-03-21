from py3do import cube, chamfer_bottom

from py3do.vis import view_pyglet



m = cube()
mc = chamfer_bottom(m, 0.1)

view_pyglet(mc)
