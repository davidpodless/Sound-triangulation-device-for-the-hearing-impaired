from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.app import App
from kivy.graphics import *
from kivy.properties import ListProperty
from kivy.core.window import Window
import math
import util


RADIUS = 180


class SoundCircle(Widget):
	def __init__(self, **kwarg):
		super(SoundCircle, self).__init__(**kwarg)
		self.line: Line = None
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

			self.line = Line(points=circle_points,
			                 close=True,
			                 width=3)

			Clock.schedule_interval(self.update, 1 / 60)

	def update(self, *args):
		for coor_ind in range(0, len(self.line.points), 2):
			x = self.line.points[coor_ind] - self.circle_center[0]
			y = self.line.points[coor_ind + 1] - self.circle_center[1]
			_, ang_rad = util.cart2pol(x, y)
			ang_deg = math.degrees(ang_rad)

			x_new, y_new = util.pol2cart(RADIUS * 1 + (util.build_gaussian_for_circle(120, 60)(ang_deg)) + (util.build_gaussian_for_circle(30, 50)(ang_deg)) + (util.build_gaussian_for_circle(220, 80)(ang_deg)), ang_rad)
			self.line.points[coor_ind] = x_new + self.circle_center[0]
			self.line.points[coor_ind + 1] = y_new + self.circle_center[1]

class SoundCircleApp(App):
	def build(self):
		foo = lambda x: print("here") if x else print("there")
		return SoundCircle()
