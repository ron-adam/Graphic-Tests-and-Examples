"""
    Rotating cubes with moving lights.

    This was created for discussing the higher level programming API
    for Datoviz.

"""
import time
import numpy as np

import vispy
from vispy import gloo, app
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import create_cube

vispy.use("pyqt5")


def vec3(x, y, z):
    return np.array([(x, y, z)], dtype='f')


def normalize(vec):
    vec -= vec.min()
    vec /= vec.max()
    return vec


def mid_polate(A, B, C, t):
    """
        Cubic interpolate (B-Spline) value between midpoint of (A,B),
        and (B,C), at position t between the midpoints.

        Args:
            A: The starting value(s).
            B: The midpoint value(s).
            C: The ending value(s).
            t: The interpolation index between 0.0 and 1.0.

        Returns:
            The interpolated value(s).

        Author: Ronald Adam
    """
    return B + ((A - B) * (1 - t)**2 + (C - B) * t**2) / 2


class SmoothPath:
    """
        Animation path over time period.

        points:  A series of points on a path.
        period:  Time to travel points in seconds.

        note:  The path wraps beck to beginning if time exceed the period.
    """
    def __init__(self, points, period):
        self.points = np.array(points, dtype='f')
        self.period = period
        self.tm = 0.0
        self._inds = np.array((len(points)-1, 0, 1), dtype='i')

    def __call__(self, td):
        self.tm += td / self.period
        if self.tm >= 1.0:
            self.tm %= 1.0
            self._inds = (self._inds + 1) % len(self.points)
        return mid_polate(*self.points[self._inds], self.tm)

    def __repr__(self):
        return f"SmoothPath({self.points}, {self.period})"


def checkerboard(size, color0=(0,0,0), color1=(1,1,1)):
    """
        Create an array with a checkerboard pattern.

        Args:
            size:   (x, y)
            color0:  Color for odd locations.
            color1:  Color for even locations.

        Result:
            Returns a rgb array with alternating colored pixels.

    """
    x, y = size
    ix, iy = np.indices(size)
    sx = ix % 2
    sy = iy % 2
    s = (sx + sy) % 2
    g = np.ones((x, y, 3), dtype='f')
    g[s==1] = color0
    g[s==0] = color1
    return g


class SceneCanvas(app.Canvas):
    """
        A scene canvas with animated objects.


    """
    def __init__(self):
        app.Canvas.__init__(self, size=(640, 480), title='Animated Lights and Cubes', keys="interactive")

        # OpenGL initialization
        gloo.set_state(clear_color=(0.1, 0.1, 0.15, 1.0),
                       depth_test=True,
                       polygon_offset=(1, 1),
                       blend_func=('src_alpha', 'one_minus_src_alpha'),
                       line_width=0.75)
        x, y = self.physical_size
        self.aspect = x/y
        self.proj = perspective(45.0, self.aspect, 0.1, 100.0)

        # Transforms.
        self.mvp = {
            "model": np.eye(4),                       # mat4
            "view": np.eye(4),                        # mat4
            "proj": self.proj,                        # mat4
            "panel_size": list(self.physical_size),   # vec2
        }

        self.camera = None

        # Light position/color pairs will go here.
        # Use add_light to update.
        self.lights = []

        # Items to display.
        # Use add_item method to update.
        self.items = []

        # Start timer after resize.
        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.tm = time.time()
        self.show()

    def on_timer(self, event):
        self.tm = time.time()
        self.update()

    def on_resize(self, event):
        self.resize_view()

    def resize_view(self):
        x, y = self.physical_size
        proj = perspective(45.0, x/y, 0.1, 100.0)

        # Mutating these values enable other objects to see the change.
        self.mvp['proj'][:] = proj
        self.mvp['panel_size'][:] = x, y

    def add_camera(self, camera):
        self.camera=camera

    def add_light(self, light):
        self.lights.append(light)

    def add_item(self, item):
        self.items.append(item)

    def on_draw(self, event):
        for light in self.lights:
            light.draw(self.tm)

        for item in self.items:
            item.draw(self.tm)


class Camera:
    """
        Standard 3D camera.

        A Camera defines the view matrix used by objects.

    """
    def __init__(self, position, scene):
        self.position = position
        self.scene = scene

        self.mvp = scene.mvp.copy()                     # Copy to avoid changes to parent values.
        self.mvp['view'][:] = translate(self.position)

        #self.lights = scene.lights


class PointLight:
    """
        Visible Point light source.

        Lights are both objects and values objects need.

        Lights can be included in the scene as a visible object.

        Objects need the position, and color of each light.
            More specifically it needs the distance, direction, and color.

        OPTIONS:  Define a light group,  Or have a light tree structure.

    """

    vertex = """
        #version 450
        
        in vec3  a_position;
        in vec3  a_color;

        // Possibly make this a structure.
        uniform mat4 proj;
        uniform mat4 view;
        uniform mat4 model;
        uniform vec2 panel_size;
                
        out vec4 Color;
        
        void main()
        {
            Color = vec4(a_color, 1.0);
            
            gl_Position = proj * view * model * vec4(a_position, 1.0);
            
            vec4 eye_pos = view * model * vec4(a_position, 1.0);
            float distance_to_camera = length(eye_pos.xyz);
            gl_PointSize = panel_size.y / distance_to_camera;
        } 
        """

    fragment = """
        #version 450
        
        in vec4 Color;
        out vec4 FragColor;
        
        void main()
        {
            // Fragment position within the point sprite
            vec2 coord = 2.0 * gl_PointCoord - 1.0;
            float dist_squared = dot(coord, coord);
            if (dist_squared > 1.0)
                discard;
            FragColor = max(Color, 1.0 - dist_squared);
        }
        """

    def __init__(self, path, color, scene):
        self.path = path     # callable with time returns position.
        self.color = color
        self.scene = scene

        self.pos = vec3(5,0,5)   # Accessed by lighted objects.

        self.mvp = scene.mvp.copy()

        self.program = Program(PointLight.vertex, PointLight.fragment)
        self.update_program()
        self.tm = time.time()

    def update_program(self):
        self.values = {
            'a_position': vec3(0,0,0),
            'a_color': self.color,
        }
        self.values.update(self.mvp)
        for key in self.values:
            self.program[key] = self.values[key]

    def animate(self, tm):
        # Update location along path.
        td = tm - self.tm
        self.tm = tm
        self.pos = self.path(td)      # Keep reference updated for objects.
        model = translate(self.pos)
        self.program['model'] = model
        self.program['view'] = self.mvp['view']
        self.program['proj'] = self.mvp['proj']              # May have been resized.
        self.program['panel_size'] = self.mvp['panel_size']  # May have been resized.

    def draw(self, tm):
        self.animate(tm)
        self.program.draw('points')


class Cube:
    vertex = """
        #version 450

        layout (location = 0) in vec3 position;    // Vertex positions.
        layout (location = 1) in vec3 normal;      // Vertex normals.
        layout (location = 2) in vec3 color;       // Vertex colors.
        layout (location = 3) in vec2 texcoord;    // Texture coordinates.

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 proj;
        uniform vec2 panel_size;
        uniform sampler2D texture;

        out vec4 FragPos;
        out vec3 Normal;
        out vec3 Color;
        out vec3 CamPos;
        out vec2 TexCoord;

        void main()
        {
            FragPos = model * vec4(position, 1.0);
            Normal = mat3(transpose(inverse(model))) * normal;
            CamPos = (inverse(view) * vec4(0, 0, 0, 1)).xyz;
            TexCoord = texcoord;
            gl_Position = proj * view * FragPos;
        }
        """

    fragment = """
        #version 450

        in vec3 Normal;
        in vec4 FragPos;
        in vec3 CamPos;
        in vec2 TexCoord;

        uniform vec4 material;
        uniform vec3 light_pos[3];
        uniform vec3 light_color[3];
        uniform sampler2D texture;

        out vec4 FragColor;
        
        
        vec3 lighting(vec3 normal, vec3 texColor,
                      vec3 camPos, vec3 fragPos,
                      vec3 lightPos[3], vec3 lightColor[3])
        {
            vec3 norm = normalize(normal);
            vec3 viewDir = normalize(CamPos - fragPos);
            
            // Ambient color.
            vec3 color = vec3(material.x);

            for (int i=0; i<3; i++) {
                
                vec3 lightDir = normalize(lightPos[i] - fragPos);
                float fragDist = distance(lightPos[i], fragPos); 
                float att = 1.0 / (1.0 + 0.01 * fragDist + 0.05 * (fragDist * fragDist));
                 
                float diffuse = att * material.y * max(dot(norm, lightDir), 0.0);
                
                vec3 halfAngle = normalize(viewDir + lightDir);
                float specular = att * material.z * pow(max(dot(norm, halfAngle), 0.0), material.w);

                color += (diffuse + specular) * lightColor[i];
            }
            return color * texColor;
        }


        void main()
        {   
            vec3 tex_color = texture2D(texture, TexCoord).xyz;
            vec3 color = lighting(Normal, tex_color, CamPos, FragPos.xyz, light_pos, light_color);
            FragColor = min(vec4(color, 1.0), 1.0);
        }
        """

    def __init__(self, path, material, texture, parent):

        self.path = path
        self.material = material
        self.texture = texture
        self.parent = parent

        self.mvp = parent.mvp.copy()
        self.lights = parent.lights

        # Build cube data
        V, F, outline = create_cube()
        vertices = VertexBuffer(V)
        self.faces = IndexBuffer(F)
        self.outline = IndexBuffer(outline)

        # Build program
        self.program = Program(Cube.vertex, Cube.fragment)
        self.program.bind(vertices)

        self.phi, self.theta = (np.random.random(size=2) - .5) * 100

        self.update_program()
        self.tm = time.time()


    def update_program(self):

        self.values = {
            'material': self.material,
            'texture': self.texture,
            'light_pos[0]': self.lights[0].pos,
            'light_color[0]': self.lights[0].color,
            'light_pos[1]': self.lights[1].pos,
            'light_color[1]': self.lights[1].color,
            'light_pos[2]': self.lights[2].pos,
            'light_color[2]': self.lights[2].color,
        }
        self.values.update(self.mvp)
        for key in self.values:
            self.program[key] = self.values[key]

    def animate(self, tm):
        td = tm - self.tm
        self.tm = tm

        #
        # Updating values that may have changed.
        #

        # Vispy requires this work around for uniform arrays.
        self.program['light_pos[0]'] = self.lights[0].pos
        self.program['light_color[0]'] = self.lights[0].color
        self.program['light_pos[1]'] = self.lights[1].pos
        self.program['light_color[1]'] = self.lights[1].color
        self.program['light_pos[2]'] = self.lights[2].pos
        self.program['light_color[2]'] = self.lights[2].color

        # Panel may have been resized.
        self.program['panel_size'] = self.mvp['panel_size']
        self.program['proj'] = self.mvp['proj']  # May have been resized.

        #
        # Animate cube.
        #

        # Cube rotation.
        theta = self.theta * td
        phi = self.phi * td
        rot = np.dot(rotate(theta, (0, 0, 1)), rotate(phi, (0, 1, 0)))
        model = rot @ self.mvp['model']
        self.mvp['model'] = model

        # Cube position.
        pos = self.path(td)
        self.program['model'] =  model @ rot @ translate(pos)

    def draw(self, tm):
        self.animate(tm)
        self.program.draw('triangles', self.faces)


if __name__ == '__main__':

    scene = SceneCanvas()

    camera = Camera((0, 0, -15), scene)
    scene.add_camera(camera)

    # Add lights to scene.
    r = 6.0      # Curve radius lights follow.
    points = [(-r, 0, 0), (0, 0, -r), (r, 0, 0), (0, 0, r)]
    path = SmoothPath(points, period=1.0)
    color = (0, 1, 1)
    light1 = PointLight(path, color, scene)
    scene.add_light(light1)

    points = [(0, r, 0), (0, 0, r), (0, -r, 0), (0, 0, -r)]
    path = SmoothPath(points, period=1.0)
    color = (1, 0, 1)
    light2 = PointLight(path, color, scene)
    scene.add_light(light2)

    points = [(r, 0, 0), (0, -r, 0), (-r, 0, 0), (0, r, 0)]
    path = SmoothPath(points, period=1.0)
    color = (1, 1, 0)
    light3 = PointLight(path, color, scene)
    scene.add_light(light3)

    # Add Cubes to scene.
    for n in range(8):
        points = (np.random.random(size=(3,3))-.5) * 5.0
        path = SmoothPath(points, period=2.0)
        material = (0.05, 0.7, 1.0, 60.0)
        color = normalize(np.random.random(size=3))
        texture = checkerboard((n+1, n+1), color, 1 - color)
        cube = Cube(path, material, texture, parent=scene)
        scene.add_item(cube)

    app.run()
