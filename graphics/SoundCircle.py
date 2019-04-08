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

RADIUS = 150


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
				x = RADIUS * math.cos(i_rad)
				y = RADIUS * math.sin(i_rad)
				circle_points.append(self.circle_center[0] + x)
				circle_points.append(self.circle_center[1] + y)

			self.line = Line(points=circle_points,
			                 close=True,
			                 width=3)

			Clock.schedule_interval(self.update, 1 / 60)

	def update(self, *args):
		test_ang = 30
		for coor_ind in range(0, len(self.line.points), 2):
			x = self.line.points[coor_ind] - self.circle_center[0]
			y = self.line.points[coor_ind + 1] - self.circle_center[1]
			if x == 0:
				continue
			ang = math.degrees(math.atan(y/x))
			if y < 0:
				ang = ang + 180
			if test_ang - 5 < ang < test_ang + 5:
				print("x: ", x)
				print("y: ", y)
				print("ang: ", ang)

				ang_rad = math.radians(ang)
				x_new = RADIUS * 1.2 * math.cos(ang_rad)
				y_new = RADIUS * 1.2 * math.sin(ang_rad)
				self.line.points[coor_ind] = x_new + self.circle_center[0]
		# 		self.line.points[coor_ind] = x*1.1
		# 		self.line.points[coor_ind + 1] = y*1.1
				self.line.points[coor_ind + 1] = y_new + self.circle_center[1]

		# print(self.line.points)
		# self.line.points[0] = 0
		# self.line.points[1] = 0
		# self.x += self.velocity[0]
		# self.y += self.velocity[1]
		# if self.x < 0 or self.x > Window.width:
		# 	self.velocity[0] *= -1
		# if self.y < 0 or self.y > Window.height:
		# 	self.velocity[1] *= -1
		pass


# class Root(BoxLayout):
# 	pass

# class ClockRect(Widget):
# 	velocity = ListProperty([10, 15])
#
# 	def __init__(self, **kwargs):
#
# 		super(ClockRect, self).__init__(**kwargs)
#
# 		# Clock.schedule_interval(self.update, 1/60)
#
# 	def update(self, *args):
# 		self.x += self.velocity[0]
# 		self.y += self.velocity[1]
# 		if self.x < 0 or self.x > Window.width:
#  			self.velocity[0] *= -1
# 		if self.y < 0 or self.y > Window.height:
# 			self.velocity[1] *= -1

class SoundCircleApp(App):
	def build(self):
		# with self.canvas:
		# Add a red color
		# Color(1., 0, 0)

		# Add a rectangle
		# Rectangle(pos=(10, 10), size=(500, 500))

		# runTouchApp(Root())
		return SoundCircle()
