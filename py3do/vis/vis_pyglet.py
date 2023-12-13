import numpy as np

try:
    import pyglet
    import pyglet.math
    import pyglet.gl as gl
    have_pyglet = True
except:
    have_pyglet = False


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

class MyMaterialGroup(pyglet.graphics.ShaderGroup):
    def __init__(self, vertex_source, fragment_source):
        self.vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
        self.frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
        self.program = pyglet.graphics.shader.ShaderProgram(self.vert_shader, self.frag_shader)
        super().__init__(self.program)
        
class FixedColorMaterialGroup(MyMaterialGroup):
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
        frag_src = self.fragment_source.format(r=r, g=g, b=b, a=a)
        super().__init__(self.vertex_source, frag_src)
        
class DiffuseMaterialGroup(MyMaterialGroup):
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
        frag_src = self.fragment_source.format(ambient=ambient,
                                               brightness_scale=brightness_scale)
        super().__init__(self.vertex_source, frag_src)

class PygletViewer(pyglet.window.Window):
    def __init__(self, m, **kwargs):
        self.scale = 1.0
        self.rot_z = 0.0
        self.rot_x = 0.0
        self.wireframe = True

        self.vertex_marker_size = 0.05

        self.marked_faces = None
        self.marked_edges = None
        self.marked_vertices = None

        self.m = m

        # Initialize window
        self.gl_config = gl.Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        super().__init__(width=960, height=540, resizable=True, config=self.gl_config)
        print("OpenGL Context: {}".format(self.context.get_info().version))

        # build model visualization
        self.set_model(m, **kwargs)

    def set_model(self, m=None, marked_vertices=None, marked_edges=None, marked_faces=None):
        if m is not None:
            self.m = m
        if marked_faces is not None:
            self.marked_faces = marked_faces
        if marked_edges is not None:
            self.marked_edges = marked_edges
        if marked_vertices is not None:
            self.marked_vertices = marked_vertices

        # UI batches and groups
        self.batch_ui = pyglet.graphics.Batch()
        self.group_ui = FixedColorMaterialGroup([1.0, 1.0, 1.0, 1.0])
        # create batches and groups for model
        self.batch_model = pyglet.graphics.Batch()
        self.batch_edge = pyglet.graphics.Batch()
        self.group_model = DiffuseMaterialGroup(ambient=0.5)
        self.group_edges = FixedColorMaterialGroup([0.5, 1.0, 1.0, 1.0])
        self.group_vertex_mark = DiffuseMaterialGroup(ambient=0.75)

        # calculate vertices and faces
        vs = self.m.vertices[:, [0,2,1]]
        normals = m.normals[:, [0,2,1]]
        fvs = vs[m.faces]  # vertices of faces
        center = vs.mean(axis=0) 
        fvs -= center
        fvs = fvs.ravel()
        n = len(fvs) // 3

        # calculate face colors
        vert_colors = np.repeat([[0.65,0.39,0.13,1.0]], n, axis=0)
        if self.marked_faces is not None:
            marked_faces = np.asarray(self.marked_faces)
            assert len(marked_faces.shape) == 1
            marked_faces *= 3
            for j in range(3):
                vert_colors[marked_faces+j] = [1,0,0,1]

        # add vertices/faces to GL
        vertex_list1 = self.group_model.program.vertex_list(n, gl.GL_TRIANGLES,
                                            self.batch_model, self.group_model,
                                            position=('f', fvs),
                                            normal=('f', normals.repeat(3,0).ravel()),
                                            colors=('f', vert_colors.ravel()))
        vertex_list2 = self.group_edges.program.vertex_list(n, gl.GL_TRIANGLES, self.batch_edge,
                                            self.group_edges,
                                            position=('f', fvs))

        # add vertex marks
        if self.marked_vertices is not None:
            marked_vertices = np.asarray(self.marked_vertices)
            assert len(marked_vertices.shape) == 1
            mv = self.m.vertices[marked_vertices,:][:,[0,2,1]] - center
            mv = mv.repeat(_point_mark.shape[0], axis=0) \
                + np.tile(_point_mark*self.vertex_marker_size,
                          (marked_vertices.shape[0], 1))
            mark_colors = np.repeat([[1.0, 0, 0, 1.0]], mv.shape[0], axis=0)
            self.group_vertex_mark.program.vertex_list(mv.shape[0], gl.GL_TRIANGLES,
                                                       self.batch_model, self.group_vertex_mark,
                                                       position=('f', mv.ravel()),
                                                       normal=('f', np.tile(_point_mark, (marked_vertices.shape[0],1)).ravel()),
                                                       colors=('f', mark_colors.ravel())
                            )

        # add UI
        label = pyglet.text.Label('Wireframe',
                                  #anchor_x="center",
                                  #font_name='Times New Roman',
                                  #font_size=36,
                                  x=10, y=10,
                                  batch=self.batch_ui, group=self.group_ui)
        #labelb = pyglet.shapes.Line(10, 10, 150,10, width=5,
        #                         batch=batch_ui, group=group_ui)

    def on_draw(self):
        model_tr = pyglet.math.Mat4()
        model_tr = model_tr.scale((self.scale, self.scale, self.scale))
        model_tr = model_tr.rotate(self.rot_x, (1,0,0))
        model_tr = model_tr.rotate(self.rot_z, (0,1,0))
        self.group_model.program['model'] = model_tr
        self.group_edges.program['model'] = model_tr
        self.group_vertex_mark.program['model'] = model_tr
        
        self.clear()
        # based on https://community.khronos.org/t/solid-wireframe-in-the-same-time/43077/5
        gl.glPolygonOffset(1,1)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        if self.wireframe:
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        self.batch_model.draw()
        if self.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            self.batch_edge.draw()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        # draw UI
        ### glPushMatrix()
        ### glLoadIdentity()
        ### glTranslatef(-window.width/2,0,10000) # = projection's near val
        ### batch_ui.draw()
        ### glPopMatrix()

    def close(self):
        self.is_active = False
        super().close()

    def on_resize(self, width, height):
        w, h = self.get_framebuffer_size()
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
        self.projection = proj
        look_at = pyglet.math.Mat4.look_at(pyglet.math.Vec3(0, 0, -100),
                                           pyglet.math.Vec3(0, 0, 0),
                                           pyglet.math.Vec3(0, 1, 0))
        self.view = look_at
        return pyglet.event.EVENT_HANDLED
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.rot_x += dy / 600
        self.rot_z -= dx / 300
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.scale *= (1 + scroll_y * 0.1)
    def on_key_press(self, symbol, modifiers):
        if (modifiers & pyglet.window.key.MOD_CTRL) and symbol == pyglet.window.key.W:
            self.close()
    def on_text_motion(self, motion):
        if motion == pyglet.window.key.MOTION_LEFT:
            self.rot_z += -0.1
        elif motion == pyglet.window.key.MOTION_RIGHT:
            self.rot_z += 0.1
        elif motion == pyglet.window.key.MOTION_UP:
            self.rot_x += -0.1
        elif motion == pyglet.window.key.MOTION_DOWN:
            self.rot_x += 0.1
        elif motion == pyglet.window.key.MOTION_NEXT_PAGE:
            self.scale *= 0.9
        elif motion == pyglet.window.key.MOTION_PREVIOUS_PAGE:
            self.scale *= 1.1
        elif motion == pyglet.window.key.MOTION_BEGINNING_OF_LINE:
            self.scale = 1.0
            self.rot_z = 0.0
            self.rot_x = 0.0

    def update(self, dt):
        pass
    def run(self):
        gl.glEnable(gl.GL_MULTISAMPLE_ARB)
        gl.glEnable(gl.GL_DEPTH_TEST)
        pyglet.clock.schedule_interval(self.update, 1/60)
        self.is_active = True
        pyglet.app.run()


def view_pyglet_noblock(m, **kwargs):
    from threading import Thread
    from threading import Event
    pv_created = Event()
    def run(ret_pv):
        pv = PygletViewer(m, **kwargs)
        ret_pv[0] = pv
        pv_created.set()
        pv.run()
    ret_pv = [None]
    t = Thread(target=run, args = (ret_pv,))
    t.start()
    pv_created.wait()
    pv = ret_pv[0]
    return pv

def view_pyglet_block(m, **kwargs):
    pv = PygletViewer(m, **kwargs)
    pv.run()


#view_pyglet = view_pyglet_block
def view_pyglet(m, *args, **kwargs):
    pv = view_pyglet_noblock(m, *args, **kwargs)
    print("AAAA")
    while(True):
        ang = float(input())
        if not pv.is_active:
            break
        pv.rot_z = ang
