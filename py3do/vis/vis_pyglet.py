import pyglet
from pyglet.gl import *

def view_pyglet(m, *args, **kwargs):
    # Direct OpenGL commands to this window.
    try:
        # Try and create a window with multisampling (antialiasing)
        config = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        window = pyglet.window.Window(width=960, height=540, resizable=True, config=config)
    except pyglet.window.NoSuchConfigException:
        # Fall back to no multisampling for old hardware
        window = pyglet.window.Window(resizable=True)

    #window.projection = pyglet.window.Projection2D()  # orthographic
    window.projection = pyglet.window.Projection3D()  # perspective
    batch = pyglet.graphics.Batch()
    batch_edge = pyglet.graphics.Batch()
    print("OpenGL Context: {}".format(window.context.get_info().version))

    # each vertex is repeated for each triangle to get 'faceted' look
    fvs = m.vertices[m.faces]  # vertices of faces
    fvs -= m.vertices.mean(axis=0)
    #fvs *= 10

    # Create a Material and Group for the Model
    diffuse = [0.5, 0.0, 0.0, 1.0]
    ambient = [0.5, 0.0, 0.3, 1.0]
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

    #n = len(m.vertices)
    #batch.add_indexed(n, GL_TRIANGLES, group, m.faces.ravel(),
    #                      ('v3f', v.ravel()),
    #                      
    #                      )
    fvs = fvs.ravel()
    n = len(fvs) // 3
    batch.add(n, GL_TRIANGLES, group,
                          ('v3f', fvs), ('n3f', m.normals.repeat(3,0).ravel()),
                          )
    batch_edge.add(n, GL_TRIANGLES, None,
                          ('v3f', fvs), #('n3f', m.normals.repeat(3,0).ravel()),
                          ('c4f', n * (0.0, 0, 0, 1.0)),
                          )

    @window.event
    def on_draw():
        # based on https://community.khronos.org/t/solid-wireframe-in-the-same-time/43077/5
        window.clear()
        #glPolygonOffset(1,1)
        #glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        #glEnable(GL_POLYGON_OFFSET_FILL)
        glEnable(GL_CULL_FACE)
        batch.draw()
        #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        #glDisable(GL_POLYGON_OFFSET_FILL)
        #batch_edge.draw()
    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        #glLoadIdentity()
        glRotatef(1, dy, dx, 0)
        #glTranslatef(0, 0, -100)

    def update(dt):
        pass
        #glRotatef(0.5, dt, dt, dt)
    def vec(*args):
        return (GLfloat * len(args))(*args)
    
    #glEnable(GL_MULTISAMPLE_ARB)
    glEnable(GL_DEPTH_TEST)
    glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(0.7, 0.7, 0.7, 1))
    glLightfv(GL_LIGHT0, GL_POSITION, vec(100.0, 100.0, 100.0, 0.0))
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    #glEnable(GL_LIGHT1)
    
    # Uncomment this line for a wireframe view:
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    
    #glTranslatef(100,100,10)
    #glTranslatef(0, 0, -100)
    gluLookAt(0, 0, -100,
              0, 0, 0,
              0, 1, 0, )
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()

