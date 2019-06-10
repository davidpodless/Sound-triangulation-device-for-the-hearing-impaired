import threading
import graphics.SoundCircle as SoundCircle
from kivy.core.window import Window

class GraphicRunnerThread(threading.Thread):
	def run(self):
		self.graphicThread = SoundCircle.SoundCircleApp()
		self.graphicThread.run()
		Window.fullscreen = True
