import threading
import time
import random

def add_angle_and_time(angle_and_amp):
	DataCollectingThread.angle_list.append((angle_and_amp, time.time()))

def add_angles():
	while DataCollectingThread.data:
		toPrint = DataCollectingThread.data.pop()
		if isinstance(toPrint, int):
			continue

		if (toPrint[0] != 0):
			print(toPrint)
			angle_and_amp = (toPrint[0], toPrint[1]/toPrint[1])
			DataCollectingThread.angle_list.append((angle_and_amp, time.time()))


class DataCollectingThread(threading.Thread):
	angle_list = []
	data = None

	def __init__(self, data):
		DataCollectingThread.data = data

	def run(self):
		while True:
			DataCollectingThread.angle_list = []
			for _ in range(4):
				pair = (random.randint(0, 360), random.uniform(0, 3))
				add_angle_and_time(pair)
			time.sleep(0.25)

	def check_angles_and_time(self):
		for angle in DataCollectingThread.angle_list:
			continue



def angle_list():
	return DataCollectingThread.angle_list
