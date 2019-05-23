import threading
import graphics.SoundCircle as SoundCircle


class GraphicRunnerThread(threading.Thread):
    def run(self):
        self.graphicThread = SoundCircle.SoundCircleApp()
        self.graphicThread.run()
