import threading
import graphics.SoundCircle as SoundCircle
from kivy.core.window import Window

class GraphicRunnerThread(threading.Thread):
	def run(self):
		self.graphicThread = SoundCircle.SoundCircleApp()
		Window.fullscreen = True
		self.graphicThread.run()
