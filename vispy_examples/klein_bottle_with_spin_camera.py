# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Demonstration of the Klein bottle rendering using Mesh.

    This uses a modified arcball camera to give continuous spin interaction
    with the mouse. Possibly this could be improved further.  - Ron Adam

"""
import sys, time
import numpy as np
from vispy import app, scene
from vispy.geometry.parametric import surface
from vispy.util.quaternion import Quaternion


QAngle = Quaternion.create_from_euler_angles


class SpinCamera(scene.cameras.ArcballCamera):
    """
        Move camera posityion around object.

        angle values are:
            x = rotate camera clockwise counter clockwise
            y = move camera left or right around scene
            z = move camera up or down around scene
    """
    def __init__(self, parent, *args, **kwds):
        print(dir(parent))
        canvas = parent.parent.canvas
        scene.cameras.ArcballCamera.__init__(self, *args, **kwds)
        self._press_event = None
        self.rotate_speed = np.zeros(3)
        canvas.events.mouse_press.connect(self.on_mouse_press)
        canvas.events.mouse_release.connect(self.on_mouse_release)
        self._timer = app.Timer(1/60, start=False, connect=self.on_timer)
        self._timer.start()
        self.last_time = time.time()

    def rotate_camera_view(self, angle):
        qa = QAngle(*angle, degrees=True)
        cq = self.get_state()['_quaternion']
        self.set_state({'_quaternion': qa*cq})
        self.view_changed()

    def on_mouse_press(self, event):
        self.rotate_speed *= 0
        self._press_event = event

    def on_mouse_release(self, event):
        if event.time - self._press_event.time < .5:
            de = event.drag_events()
            if de is not None and len(de) > 4:
                time = (de[-2].time - de[-4].time)
                tr = event.trail()[-4:-1]
                dist = (tr[0] - tr[-1])/6
                self.rotate_speed[1:3] = dist / time
        else:
            self.rotate_speed *= 0

    def on_timer(self, event):
        """ Called frequently to keep view spinning. """
        try:
            dt = event.dt
        except AttributeError:
            t = time.time()
            dt = t - self.last_time
            self.last_time = t
        self.rotate_camera_view(self.rotate_speed * dt)


def klein(u, v):
    from math import pi, cos, sin
    if u < pi:
        x = 3 * cos(u) * (1 + sin(u)) + \
            (2 * (1 - cos(u) / 2)) * cos(u) * cos(v)
        z = -8 * sin(u) - 2 * (1 - cos(u) / 2) * sin(u) * cos(v)
    else:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(v + pi)
        z = -8 * sin(u)
    y = -2 * (1 - cos(u) / 2) * sin(v)
    return x/5, y/5, z/5


# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Add camera
#view.camera = scene.cameras.ArcballCamera(parent=view.scene)
view.camera = SpinCamera(parent=view.scene)

# Add mesh
vertices, indices = surface(klein, urepeat=3)
indices = indices.reshape(len(indices)//3, 3)

# Note: Shading is set to 'flat' instead of 'smooth' due to incorrect
#       rendering of 'smooth' shading. I have not looked further to find the
#       cause. It displays alternating black and white triangles.

mesh = scene.visuals.Mesh(vertices=vertices['position'], faces=indices,
                          color='white', parent=view.scene, shading='flat')


if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()
