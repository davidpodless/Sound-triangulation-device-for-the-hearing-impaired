import threading
import graphics.SoundCircle as SoundCircle


class GraphicRunnerThread(threading.Thread):
	def run(self):
		SoundCircle.SoundCircleApp().run()
