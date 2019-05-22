import threading
import time
import random


class DataCollectingThread(threading.Thread):
	angle_list = []

	def run(self):
		while True:
			DataCollectingThread.angle_list = []
			for _ in range(4):
				pair = (random.randint(0, 360), random.uniform(0, 3))
				DataCollectingThread.angle_list.append(pair)
			time.sleep(1)


def angle_list():
	return DataCollectingThread.angle_list
