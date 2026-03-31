# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
    A spin ArcBall camera example.

    This uses a modified arcball camera to give continuous spin interaction
    with the mouse. Possibly this could be improved further.  - Ron Adam

    Modified from VisPy example Klein bottle rendering using Mesh.
"""
import sys, time
import numpy as np
from vispy import app, scene
from vispy.geometry.parametric import surface
from vispy.util.quaternion import Quaternion


QAngle = Quaternion.create_from_euler_angles

# ArcBall Rotation angles for reference:
#   x = rotate camera clockwise / counter - clockwise(roll)
#   y = rotate camera around scene horizontally(yaw)
#   z = rotate  up / down around scene(pitch)


class SpinCamera(scene.cameras.ArcballCamera):
    """
    Arcball camera with continuous spin interaction from mouse drag gestures.

    This camera extends VisPy's ArcballCamera to provide a smooth continuous 
    spin effect based on mouse drag velocity. When the user releases the mouse 
    after dragging, the camera continues to spin and gradually decelerates 
    based on the damping factor.

    The spin is constrained to the horizontal (yaw) and vertical (pitch) axes,
    providing intuitive rotation around the scene object while maintaining
    the camera's up direction.

    Parameters
    ----------
    parent : vispy.scene.Scene
        The parent scene object.
    damping_factor : float, optional
        The factor by which rotation speed is multiplied each frame
        (0 < damping_factor < 1). Higher values result in slower deceleration.
        Default is 0.99.
    *args, **kwds
        Additional arguments passed to the parent ArcballCamera.__init__.

    Attributes
    ----------
    rotate_speed : ndarray
        Current rotation speed in degrees/second as [roll, yaw, pitch].
        Only yaw (index 1) and pitch (index 2) are used for continuous spin.
    damping_factor : float
        Deceleration multiplier applied each frame for smooth spin slowdown.
    _press_event : vispy.event.MouseEvent or None
        Records the mouse press event to calculate drag duration and velocity.
    _timer : vispy.app.Timer
        Timer that triggers on_timer() ~60 times per second to update spin.

    Notes
    -----
    - Drag duration > 0.5 seconds: drag is ignored (user is inspecting the view)
    - Drag with < 5 mouse events: spin is not activated (too short/fast movement)
    - Only yaw and pitch axes are used for spin (roll is ignored)
    - Spin gradually decelerates until the camera stops

    """
    def __init__(self, parent, damping_factor=0.99, *args, **kwds):
        """
        Initialize the SpinCamera.

        Parameters
        ----------
        parent : vispy.scene.Scene
            The parent scene object.
        damping_factor : float, optional
            The factor by which rotation speed is multiplied each frame
            (0 < damping_factor < 1).
            Higher values result in slower deceleration. Default is 0.99.
        *args, **kwds
            Additional arguments passed to the parent ArcballCamera.__init__.
        """
        canvas = parent.parent.canvas
        super().__init__(*args, **kwds)
        self._press_event = None
        self.rotate_speed = np.zeros(3)
        self.damping_factor = damping_factor
        canvas.events.mouse_press.connect(self.on_mouse_press)
        canvas.events.mouse_release.connect(self.on_mouse_release)
        self._timer = app.Timer(1/60, start=False, connect=self.on_timer)
        self._timer.start()
        self.last_time = time.time()

    def rotate_camera_view(self, angle: np.ndarray) -> None:
        """
        Rotate camera view by specified angles in degrees.

        Parameters
        ----------
        angle : ndarray
            Rotation angles in degrees as [roll, yaw, pitch].
        """
        qa = QAngle(*angle, degrees=True)
        cq = self.get_state()['_quaternion']
        self.set_state({'_quaternion': qa*cq})
        self.view_changed()

    def on_mouse_press(self, event):
        """
        Handle mouse press event: reset rotation speed and record press event.

        Called when the user presses the mouse button. Stops any current spin
        and records the press event to measure subsequent drag duration and velocity.

        Parameters
        ----------
        event : vispy.event.MouseEvent
            The mouse press event containing position and timing information.
        """
        self.rotate_speed *= 0
        self._press_event = event

    def on_mouse_release(self, event):
        """
        Handle mouse release event: calculate rotation speed from drag.

        Called when the user releases the mouse button after a drag. Calculates 
        the average drag velocity and sets the camera's rotation speed to enable
        smooth continuous spin. The spin decelerates over time based on the 
        damping_factor.

        Conditions for spin activation:
        - Drag duration must be less than 0.5 seconds (longer drags are ignored)
        - Must have at least 5 drag events (ensures substantial mouse movement)

        Parameters
        ----------
        event : vispy.event.MouseEvent
            The mouse release event containing drag trail and timing information.
        """
        if self._press_event is None:
            return

        drag_time = event.time - self._press_event.time

        # No spin if drag is held longer than 0.5 seconds.
        # It is probably being held at the end of a drag to inspect the view.
        if drag_time < .5:
            de = event.drag_events()
            # Do not spin on short clicks that may only move a few pixels.
            if de is not None and len(de) > 4:
                trail = event.trail()
                drag_rates = np.diff(trail, axis=0)
                vector = drag_rates.mean(axis=0) * 20  # May need adjustment.
                self.rotate_speed[1:3] = -vector
                return

        self.rotate_speed *= 0

    def on_timer(self, event):
        """
        Called frequently to update and animate the camera spin.

        This method is triggered approximately 60 times per second by the internal
        timer. It applies damping to the rotation speed and applies the resulting
        rotation to the camera view. The damping factor causes the spin to gradually
        slow down until it stops.

        Parameters
        ----------
        event : vispy.app.Timer
            Timer event with dt attribute (delta time since last call in seconds).
            If dt is not available, the method calculates it using system time.
        """
        # Scale spin to real time.
        if hasattr(event, 'dt'):
            dt = event.dt
        else:
            t = time.time()
            dt = t - self.last_time
            self.last_time = t
        self.rotate_speed *= self.damping_factor
        self.rotate_camera_view(self.rotate_speed * dt)



def klein(u, v):
    from math import pi, cos, sin
    PI = pi
    SCALE = 5

    if u < PI:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(u) * cos(v)
        z = -8 * sin(u) - 2 * (1 - cos(u) / 2) * sin(u) * cos(v)
    else:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(v + PI)
        z = -8 * sin(u)

    y = -2 * (1 - cos(u) / 2) * sin(v)
    return x / SCALE, y / SCALE, z / SCALE

# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Add camera
view.camera = SpinCamera(parent=view.scene)

# Add mesh
vertices, indices = surface(klein, urepeat=3)
indices = indices.reshape(len(indices)//3, 3)

# Note: Shading is set to 'flat' instead of 'smooth' due to incorrect
#       rendering of 'smooth' shading. I have not looked further to find the
#       cause. It displays alternating black and white triangles.

mesh = scene.visuals.Mesh(vertices=vertices['position'], faces=indices,
                          color='grey', parent=view.scene, shading='flat')


if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()
