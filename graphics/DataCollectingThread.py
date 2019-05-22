import threading
import time


def add_angle_and_time(angle_and_amp):
	DataCollectingThread.angle_list.append(angle_and_amp)
	DataCollectingThread.time_list.append(time.time())
	print('DataCollectingThread.angle_list')
	print(DataCollectingThread.angle_list)


def add_angles():
	while DataCollectingThread.data[0]:
		toPrint = DataCollectingThread.data[0].pop()

		if isinstance(toPrint, int):
			continue
		if len(toPrint) == 0:
			continue
		if toPrint[0] != 0:
			if (toPrint[0][1] != 0):
				angle_and_amp = (toPrint[0][0], toPrint[0][1] / toPrint[0][1])
				add_angle_and_time(angle_and_amp)


class DataCollectingThread(threading.Thread):
	angle_list = []
	time_list = []
	data = None

	def __init__(self, data, name=''):
		super().__init__(name=name)
		DataCollectingThread.data = data

	def run(self):
		while True:
			add_angles()
			self.check_angles_and_time()
			time.sleep(1 / 30)

	def check_angles_and_time(self):
		for prev_time_ind in range(len(DataCollectingThread.time_list)):
			prev_time = DataCollectingThread.time_list[prev_time_ind]
			time_passed = time.time() - prev_time
			if time_passed > 3:
				del DataCollectingThread.time_list[prev_time_ind]
				del DataCollectingThread.angle_list[prev_time_ind]

			else:
				print(DataCollectingThread.angle_list)
				print(time_passed)
				DataCollectingThread.angle_list[prev_time_ind] = \
					(DataCollectingThread.angle_list[prev_time_ind][0],
					 DataCollectingThread.angle_list[prev_time_ind][1] * time_passed)


def angle_list():
	return DataCollectingThread.angle_list
