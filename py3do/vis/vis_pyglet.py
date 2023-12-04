import numpy as np

try:
    import pyglet
    import pyglet.math
    import pyglet.gl as gl
    have_pyglet = True
except:
    have_pyglet = False


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
    ], dtype=np.double)[:,[0,2,1]]

def view_pyglet(m, marked_vertices=None, vertex_marker_size=0.05,
                marked_faces=None, *args, **kwargs):
    if not have_pyglet:
        raise RuntimeError("Pyglet not available")
    global scale, rot_z, rot_x

    # Direct OpenGL commands to this window.
    try:
        # Try and create a window with multisampling (antialiasing)
        config = gl.Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        window = pyglet.window.Window(width=960, height=540, resizable=True, config=config)
    except pyglet.window.NoSuchConfigException:
        # Fall back to no multisampling for old hardware
        window = pyglet.window.Window(resizable=True)

    print("OpenGL Context: {}".format(window.context.get_info().version))


    
    batch_ui = pyglet.graphics.Batch()
    batch_model = pyglet.graphics.Batch()
    batch_edge = pyglet.graphics.Batch()

    from pyglet.graphics.shader import Shader, ShaderProgram

    class FixedColorMaterialGroup(pyglet.graphics.ShaderGroup):
        """Draw everything in given color.  No lighting.

        Need own class: pyglet's class is incomplete.

        """
        vertex_source = """#version 330 core
            in vec3 position;
            uniform WindowBlock
            {
                mat4 projection;
                mat4 view;
            } window;

            uniform mat4 model;

            void main()
            {
                gl_Position = window.projection * model * vec4(position, 1);
            }
            """
        fragment_source = """#version 330 core
            void main()
            {{
                gl_FragColor = vec4({r}, {g}, {b}, {a});
            }}
            """
        def __init__(self, color=[0.5, 1.0, 1.0, 1.0]):
            if len(color) == 3:
                r, g, b = color
                a = 1.0
            elif len(color) == 4:
                r, g, b, a = color
            else:
                assert 0 <= color <= 1
                r = g = b = color
                a = 1.0
            self.color = color
            self.vert_shader = Shader(self.vertex_source, 'vertex')
            frag_src = self.fragment_source.format(r=r, g=g, b=b, a=a)
            self.frag_shader = Shader(frag_src, 'fragment')
            self.program = ShaderProgram(self.vert_shader, self.frag_shader)
            super().__init__(self.program)

    class DiffuseMaterialGroup(pyglet.graphics.ShaderGroup):
        """Diffuse material group.

        Lighting changes with angle.  Need own class: pyglet's class
        is incomplete.

        """
        vertex_source = """#version 330 core
            in vec3 position;
            in vec3 normal;
            in vec4 colors;
            out vec4 vertex_colors;
            out vec3 vertex_normal;
            out vec3 vertex_position;

            uniform WindowBlock
            {
                mat4 projection;
                mat4 view;
            } window;

            uniform mat4 model;

            void main()
            {
                gl_Position = window.projection * model * vec4(position, 1);
                mat3 normal_matrix = transpose(inverse(mat3(model)));
                vertex_normal = normal_matrix * normal;
                vertex_colors = colors;
                vertex_position = (model * vec4(position, 1)).xyz;
            }
        """
        fragment_source = """#version 330 core
            in vec4 vertex_colors;
            in vec3 vertex_normal;
            in vec3 vertex_position;
            out vec4 final_color;

            void main()
            {{
                vec3 light_position = vec3(0.0, 0.0, 1000.0);
                float l = dot(normalize(-light_position), normalize(vertex_normal));
                final_color = vertex_colors * (l * {brightness_scale} + {ambient});
            }}
        """
        def __init__(self, ambient=0.0, brightness_scale=1.0):
            self.ambient = ambient
            self.brightness_scale = brightness_scale
            self.vert_shader = Shader(self.vertex_source, 'vertex')
            frag_src = self.fragment_source.format(ambient=ambient,
                                                   brightness_scale=brightness_scale)
            self.frag_shader = Shader(frag_src, 'fragment')
            self.program = ShaderProgram(self.vert_shader, self.frag_shader)
            super().__init__(self.program)

    # Create a Material Groups for model, edges, UI, vertex marks
    group_ui = FixedColorMaterialGroup([1.0, 1.0, 1.0, 1.0])
    group_model = DiffuseMaterialGroup(ambient=0.5)
    group_edges = FixedColorMaterialGroup([0.5, 1.0, 1.0, 1.0])

    # Create a Material and Group for marked vertices
    diffuse = [1.0, 0.0, 0.0, 1.0]
    ambient = [1.0, 0.0, 0.0, 1.0]
    specular = [0.0, 0.0, 0.0, 0.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 1.0
    material = pyglet.model.Material("custom3", diffuse, ambient, specular, emission, shininess)
    group_vertex_mark = DiffuseMaterialGroup(ambient=0.5)

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
    # set face colors
    vert_colors = np.repeat([[0.65,0.39,0.13,1.0]], n, axis=0)
    # change colors of marked faces
    if marked_faces is not None:
        marked_faces = np.asarray(marked_faces)
        assert len(marked_faces.shape) == 1
        marked_faces *= 3
        for j in range(3):
            vert_colors[marked_faces+j] = [1,0,0,1]
    vertex_list1 = group_model.program.vertex_list(n, gl.GL_TRIANGLES, batch_model, group_model,
                                             position=('f', fvs),
                                             normal=('f', normals.repeat(3,0).ravel()),
                                             colors=('f', vert_colors.ravel()))
    vertex_list2 = group_edges.program.vertex_list(n, gl.GL_TRIANGLES, batch_edge,
                                                   group_edges,
                                                   position=('f', fvs))
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
        model_tr = pyglet.math.Mat4()
        model_tr = model_tr.scale((scale,scale,scale))
        model_tr = model_tr.rotate(rot_x, (1,0,0))
        model_tr = model_tr.rotate(rot_z, (0,1,0))
        group_model.program['model'] = model_tr
        #program_edge['model'] = model_tr
        group_edges.program['model'] = model_tr

        
        window.clear()
        # based on https://community.khronos.org/t/solid-wireframe-in-the-same-time/43077/5
        gl.glPolygonOffset(1,1)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        if wireframe:
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        batch_model.draw()
        if wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            batch_edge.draw()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        # draw UI
        ### glPushMatrix()
        ### glLoadIdentity()
        ### glTranslatef(-window.width/2,0,10000) # = projection's near val
        ### batch_ui.draw()
        ### glPopMatrix()

    @window.event
    def on_resize(width, height):
        w, h = window.get_framebuffer_size()
        s = min(w, h)
        offset_x = max(0, (w-h)//2)
        offset_y = max(0, (h-w)//2)
        gl.glViewport(offset_x, offset_y, s, s)
        s = max(w, h)
        offset_x = min(0, (w-h)//2)
        offset_y = min(0, (h-w)//2)
        gl.glViewport(offset_x, offset_y, s, s)
        proj = pyglet.math.Mat4.orthogonal_projection(-1, 1,
                                                      -1, 1,
                                                      1e3, -1e3)
        window.projection = proj
        look_at = pyglet.math.Mat4.look_at(pyglet.math.Vec3(0, 0, -100),
                                           pyglet.math.Vec3(0, 0, 0),
                                           pyglet.math.Vec3(0, 1, 0))
        window.view = look_at
        return pyglet.event.EVENT_HANDLED
    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        global rot_z, rot_x
        rot_x += dy / 600
        rot_z -= dx / 300
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
            rot_z += -0.1
        elif motion == pyglet.window.key.MOTION_RIGHT:
            rot_z += 0.1
        elif motion == pyglet.window.key.MOTION_UP:
            rot_x += -0.1
        elif motion == pyglet.window.key.MOTION_DOWN:
            rot_x += 0.1
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


    gl.glEnable(gl.GL_MULTISAMPLE_ARB)
    gl.glEnable(gl.GL_DEPTH_TEST)
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()

