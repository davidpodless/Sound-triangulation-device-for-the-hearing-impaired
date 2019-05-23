from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.app import App
from kivy.graphics import *
from kivy.core.window import Window
import math
import util
import random
import graphics.DataCollectingThread as DataCollectingThread

RADIUS = 180


class SoundCircle(Widget):
    def __init__(self, **kwarg):
        super(SoundCircle, self).__init__(**kwarg)
        self.line = None
        self.circle_center = None
        with self.canvas:
            Color(0.35, 0.35, 0.65, 1)
            self.circle_center = Window.size[0] / 2, Window.size[1] / 2
            circle_points = []

            for i in range(0, 360):
                i_rad = math.radians(i)
                x, y = util.pol2cart(RADIUS, i_rad)
                circle_points.append(self.circle_center[0] + x)
                circle_points.append(self.circle_center[1] + y)

            SoundCircle.original_circle = circle_points[:]
            self.line = Line(points=circle_points,
                             close=True,
                             width=3)

            Clock.schedule_interval(self.update, 1 / 30)

    def update(self, *args):
        self.line.points = SoundCircle.original_circle[:]

        angle_list = DataCollectingThread.DataCollectingThread.angle_list

        for coor_ind in range(0, len(self.line.points), 2):
            x = self.line.points[coor_ind] - self.circle_center[0]
            y = self.line.points[coor_ind + 1] - self.circle_center[1]
            _, ang_rad = util.cart2pol(x, y)
            ang_deg = math.degrees(ang_rad)
            x_new, y_new = util.pol2cart(RADIUS * (
                util.build_multiple_gausians(angle_list)(ang_deg)),
                                         ang_rad)
            self.line.points[coor_ind] = x_new + self.circle_center[0]
            self.line.points[coor_ind + 1] = y_new + self.circle_center[1]


class SoundCircleApp(App):
    def build(self):
        return SoundCircle()
