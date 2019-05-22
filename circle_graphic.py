from kivy.app import App
# from kivy.uix.widget import Widget

from graphics.SoundCircle import SoundCircleApp
import graphics.DataCollectingThread as DataCollectingThread

def data_collecting_thread():
	pass

def main():
	myThread = DataCollectingThread.DataCollectingThread(name="Thread-{}".format(1))
	myThread.start()
	SoundCircleApp().run()


if __name__ == '__main__':
	main()