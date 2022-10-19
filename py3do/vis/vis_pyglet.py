import numpy as np

try:
    import pyglet
    from pyglet.gl import *
    have_pyglet = True
except:
    have_pyglet = False

if have_pyglet:
    class ProjectionOrtho(pyglet.window.Projection):
        def set(self, window_width, window_height, viewport_width, viewport_height):
            glViewport(0, 0, max(1, viewport_width), max(1, viewport_height))
            glMatrixMode(gl.GL_PROJECTION)
            glLoadIdentity()
            ww = max(1.0, window_width)
            wh = max(1.0, window_height)
            glOrtho(-ww/2, ww/2, -wh/2, wh/2, -10000.0, 1e5)
            glMatrixMode(GL_MODELVIEW)

scale = 1.0
rot_z = 0.0
rot_x = 0.0
wireframe = True

_point_mark = np.array([
[0,0,1], [1,0,0], [0,1,0],
[0,0,1], [0,1,0], [-1,0,0],
[0,0,1], [-1,0,0], [0,-1,0],
[0,0,1], [0,-1,0], [1,0,0],
[0,0,-1], [1,0,0], [0,1,0],
[0,0,-1], [0,1,0], [-1,0,0],
[0,0,-1], [-1,0,0], [0,-1,0],
[0,0,-1], [0,-1,0], [1,0,0],
    ], dtype=np.float)[:,[0,2,1]]

def view_pyglet(m, marked_vertices=None, vertex_marker_size=0.05, *args, **kwargs):
    if not have_pyglet:
        raise RuntimeError("Pyglet not available")
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

    print("OpenGL Context: {}".format(window.context.get_info().version))

    
    batch_ui = pyglet.graphics.Batch()
    batch_model = pyglet.graphics.Batch()
    batch_edge = pyglet.graphics.Batch()
    
    # Create a Material and Group for UI
    diffuse = [1.0, 1.0, 1.0, 1.0]
    ambient = [1.0, 1.0, 1.0, 1.0]
    specular = [0.0, 0.0, 0.0, 0.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 50
    material = pyglet.model.Material("custom_ui", diffuse, ambient, specular, emission, shininess)
    group_ui = pyglet.model.MaterialGroup(material=material)

    # Create a Material and Group for the Model
    diffuse = [0.5, 0.3, 0.0, 1.0]
    ambient = [0.5, 0.3, 0.0, 1.0]
    specular = [0.0, 0.0, 0.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 50
    material = pyglet.model.Material("custom", diffuse, ambient, specular, emission, shininess)
    group = pyglet.model.MaterialGroup(material=material)

    # Create a Material and Group for Edges
    diffuse = [0.0, 0.0, 0.0, 0.0]
    ambient = [0.5, 1.0, 1.0, 1.0]
    specular = [0.0, 0.0, 0.0, 0.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 50
    material = pyglet.model.Material("custom2", diffuse, ambient, specular, emission, shininess)
    group2 = pyglet.model.MaterialGroup(material=material)

    # Create a Material and Group for marked vertices
    diffuse = [1.0, 0.0, 0.0, 1.0]
    ambient = [1.0, 0.0, 0.0, 1.0]
    specular = [0.0, 0.0, 0.0, 0.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 1.0
    material = pyglet.model.Material("custom3", diffuse, ambient, specular, emission, shininess)
    group_vertex_mark = pyglet.model.MaterialGroup(material=material)

    # prepare UI
    label = pyglet.text.Label('Wireframe',
                              #anchor_x="center",
                              #font_name='Times New Roman',
                              #font_size=36,
                              x=10, y=10,
                              batch=batch_ui, group=group_ui)
    #labelb = pyglet.shapes.Line(10, 10, 150,10, width=5,
    #                            batch=batch_ui, group=group_ui)
    
    # prepare model and edges
    
    # each vertex is repeated for each triangle to get 'faceted' look
    vs = m.vertices[:, [0,2,1]]
    normals = m.normals[:, [0,2,1]]
    fvs = vs[m.faces]  # vertices of faces
    center = vs.mean(axis=0) 
    fvs -= center

    fvs = fvs.ravel()
    n = len(fvs) // 3
    batch_model.add(n, GL_TRIANGLES, group,
                          ('v3f', fvs), ('n3f', normals.repeat(3,0).ravel()),
                          )
    batch_edge.add(n, GL_TRIANGLES, group2,
                          ('v3f', fvs), #('n3f', normals.repeat(3,0).ravel()),
                          ('c4f', n * (0.0, 0, 0, 1.0)),
                          )
    # add vertex marks
    if marked_vertices is not None:
        marked_vertices = np.asarray(marked_vertices)
        assert len(marked_vertices.shape) == 1
        mv = m.vertices[marked_vertices,:][:,[0,2,1]] - center
        mv = mv.repeat(_point_mark.shape[0], axis=0) \
            + np.tile(_point_mark*vertex_marker_size,
                      (marked_vertices.shape[0],1))
        batch_model.add(mv.shape[0], GL_TRIANGLES, group_vertex_mark,
                        ('v3f', mv.ravel()),
                        ('n3f', np.tile(_point_mark,
                                        (marked_vertices.shape[0],1)).ravel()),
                        )

    @window.event
    def on_draw():
        global rot_x, rot_z, scale
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_z, 0, 1, 0)
        glScalef(scale, scale, scale)
        window.clear()
        # based on https://community.khronos.org/t/solid-wireframe-in-the-same-time/43077/5
        glPolygonOffset(1,1)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        if wireframe:
            glEnable(GL_POLYGON_OFFSET_FILL)
        batch_model.draw()
        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_POLYGON_OFFSET_FILL)
            batch_edge.draw()
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        # draw UI
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(-window.width/2,0,10000) # = projection's near val
        batch_ui.draw()
        glPopMatrix()
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
        elif motion == pyglet.window.key.MOTION_BEGINNING_OF_LINE:
            scale = 1.0
            rot_z = 0.0
            rot_x = 0.0

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
    glLightfv(GL_LIGHT1, GL_AMBIENT, vec(1.0, 1.0, 1.0, 1))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    
    gluLookAt(0, 0, -100,
              0, 0, 0,
              0, 1, 0, )
    #pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()

