import threading
import time
import random

def add_angle_and_time(angle_and_amp):
	DataCollectingThread.angle_list.append(angle_and_amp)

def add_angles():
	while DataCollectingThread.data[0]:
		toPrint = DataCollectingThread.data[0].pop()

		if isinstance(toPrint, int):
			continue
		if len(toPrint) == 0:
			continue
		if toPrint[0] != 0:
			print('toPrint')
			print(toPrint[0])
			if (toPrint[0][1] != 0):
				angle_and_amp = (toPrint[0][0], toPrint[0][1]/toPrint[0][1])
				add_angle_and_time(angle_and_amp)


class DataCollectingThread(threading.Thread):
	angle_list = []
	data = None

	def __init__(self, data, name=''):
		super().__init__(name=name)
		DataCollectingThread.data = data

	def run(self):
		while True:
			DataCollectingThread.angle_list = []
			add_angles()
			time.sleep(0.25)

	def check_angles_and_time(self):
		for angle in DataCollectingThread.angle_list:
			continue



def angle_list():
	return DataCollectingThread.angle_list
