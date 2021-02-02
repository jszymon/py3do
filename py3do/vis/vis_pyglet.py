import numpy as np

import pyglet
from pyglet.gl import *


class ProjectionOrtho(pyglet.window.Projection):
    def set(self, window_width, window_height, viewport_width, viewport_height):
        glViewport(0, 0, max(1, viewport_width), max(1, viewport_height))
        glMatrixMode(gl.GL_PROJECTION)
        glLoadIdentity()
        ww = max(1.0, window_width)
        wh = max(1.0, window_height)
        glOrtho(-ww/2, ww/2, -wh/2, wh/2, -1000.0, 1e5)
        glMatrixMode(GL_MODELVIEW)

scale = 1.0
rot_z = 0
rot_x = 0

def view_pyglet(m, *args, **kwargs):
    global scale, rot_z, rot_x
    # Direct OpenGL commands to this window.
    try:
        # Try and create a window with multisampling (antialiasing)
        config = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        window = pyglet.window.Window(width=960, height=540, resizable=True, config=config)
    except pyglet.window.NoSuchConfigException:
        # Fall back to no multisampling for old hardware
        window = pyglet.window.Window(resizable=True)

    window.projection = ProjectionOrtho()  # orthographic
    #window.projection = pyglet.window.Projection3D()  # perspective
    batch = pyglet.graphics.Batch()
    batch_edge = pyglet.graphics.Batch()
    print("OpenGL Context: {}".format(window.context.get_info().version))

    # each vertex is repeated for each triangle to get 'faceted' look
    vs = m.vertices[:, [0,2,1]]
    normals = m.normals[:, [0,2,1]]
    fvs = vs[m.faces]  # vertices of faces
    fvs -= vs.mean(axis=0)

    # Create a Material and Group for the Model
    diffuse = [0.5, 0.3, 0.0, 1.0]
    ambient = [0.5, 0.3, 0.0, 1.0]
    specular = [0.0, 0.0, 0.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 50

    material = pyglet.model.Material("custom", diffuse, ambient, specular, emission, shininess)
    group = pyglet.model.MaterialGroup(material=material)

    # Create a Material and Group for Edges
    #diffuse = [0.5, 0.0, 0.0, 1.0]
    #ambient = [0.5, 0.0, 0.3, 1.0]
    #specular = [0.0, 0.0, 0.0, 0.0]
    #emission = [0.0, 0.0, 0.0, 1.0]
    #shininess = 50
    #
    #material = pyglet.model.Material("custom", diffuse, ambient, specular, emission, shininess)
    #group = pyglet.model.MaterialGroup(material=material)

    fvs = fvs.ravel()
    n = len(fvs) // 3
    batch.add(n, GL_TRIANGLES, group,
                          ('v3f', fvs), ('n3f', normals.repeat(3,0).ravel()),
                          )
    batch_edge.add(n, GL_TRIANGLES, None,
                          ('v3f', fvs), #('n3f', normals.repeat(3,0).ravel()),
                          ('c4f', n * (0.0, 0, 0, 1.0)),
                          )

    @window.event
    def on_draw():
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_z, 0, 1, 0)
        glScalef(scale, scale, scale)
        # based on https://community.khronos.org/t/solid-wireframe-in-the-same-time/43077/5
        window.clear()
        #glPolygonOffset(1,1)
        #glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        #glEnable(GL_POLYGON_OFFSET_FILL)
        batch.draw()
        #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        #glDisable(GL_POLYGON_OFFSET_FILL)
        #batch_edge.draw()
    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        global rot_z, rot_x
        rot_x -= dy / 6
        rot_z += dx / 3
    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        global scale
        scale = scale * (1 + scroll_y * 0.1)
    @window.event
    def on_key_press(symbol, modifiers):
        if (modifiers & pyglet.window.key.MOD_CTRL) and symbol == pyglet.window.key.W:
            window.close()
    @window.event
    def on_text_motion(motion):
        global rot_x, rot_z, scale
        if motion == pyglet.window.key.MOTION_LEFT:
            rot_z += -5
        elif motion == pyglet.window.key.MOTION_RIGHT:
            rot_z += 5
        elif motion == pyglet.window.key.MOTION_UP:
            rot_x += -5
        elif motion == pyglet.window.key.MOTION_DOWN:
            rot_x += 5
        elif motion == pyglet.window.key.MOTION_NEXT_PAGE:
            scale *= 0.9
        elif motion == pyglet.window.key.MOTION_PREVIOUS_PAGE:
            scale *= 1.1

    def update(dt):
            pass
    def vec(*args):
        return (GLfloat * len(args))(*args)


    glEnable(GL_MULTISAMPLE_ARB)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)  # FIX LIGHTING WITH glScale!!!!
    glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1.0, 1.0, 1.0, 1.0))
    #glLightfv(GL_LIGHT0, GL_SPECULAR, vec(0.7, 0.7, 0.7, 1))
    glLightfv(GL_LIGHT0, GL_POSITION, vec(0.0, 0.0, 1.0, 0.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    
    gluLookAt(0, 0, -100,
              0, 0, 0,
              0, 1, 0, )
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()

