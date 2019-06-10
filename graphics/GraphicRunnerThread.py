import threading
import graphics.SoundCircle as SoundCircle
from kivy.core.window import Window
from kivy.config import Config

class GraphicRunnerThread(threading.Thread):
	def run(self):
		Config.set('graphics', 'fullscreen', 'auto')
		Window.fullscreen = True
		self.graphicThread = SoundCircle.SoundCircleApp()
		self.graphicThread.run()
