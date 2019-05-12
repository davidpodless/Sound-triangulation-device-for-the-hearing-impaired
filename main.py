from os import listdir
from os.path import isfile, join
import threading
from collections import deque
import wave
from scipy.io import wavfile
import numpy as np
import collections
# import recording
import computing
import matplotlib.pyplot as plt
from systemConstants import *
'''this program uses MKS system'''

CHUNK_RECORDING = True
keepRecording = True
averageNoiseArray = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
averageNoise = 0
newNoise = 0


def getFileslist():
	# return [f for f in listdir('./wav_files') if isfile(join('./', f)) and f.endswith(".wav")]
	return ['./wav_files/48k_350_45_v2.wav']


def fake_record(files, frames):
	print("was in fake record")
	print(files)
	for file in files:
		wav = wave.open(file)
		print(file.title())
		if not CHUNK_RECORDING:
			while True:
				data = wav.readframes(CHUNK)
				if data != b'':
					frames.appendleft(data)
				else:
					break
		else:
			counter = 0
			lst = []
			while True:
				data = wav.readframes(CHUNK)
				if data != b'':
					lst.append(data)
					counter = (counter + 1) % NUM_OF_SNAPSHOTS_FOR_MUSIC
					if counter == 0:
						frames.appendleft(lst.copy())
						lst.clear()
				else:
					break


if __name__ == '__main__':
	# print(THRESHOLD_FOR_MODE)
	# print(RECORD_BUFFER_MAX) todo: delete this
	frames = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	results = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	# recordingThread = threading.Thread(group=None, target=recording.record, name="recording thread", args=(frames, results))
	fakeRecordingThread = threading.Thread(group=None, target=fake_record, name="fake recording thread", args=(getFileslist(), frames))
	computingThread = threading.Thread(group=None, target=computing.extract_data, name="compute thread", args=(frames, results))

	# recordingThread.start()
	fakeRecordingThread.start()
	# fakeRecordingThread.join()
	computingThread.start()

	computingThread.join()
	# computing.draw_graph()
	points_in_data = []
	while results:
		#
		toPrint = results.pop()
		if toPrint == 0 or not toPrint:
			continue

		points_in_data.append(toPrint[0][2])
		# for result in toPrint:
			# print(result[1])
			# if result[1] not in mode:  # count how many time specific frequency is in the data
			# 	mode[result[1]] = 1
			# else:
			# 	mode[result[1]] += 1
	# ordered_mode = collections.OrderedDict(mode)
	# print(sorted(mode.items(), key=lambda x:x[1]))
	data = (np.asarray(points_in_data).T)
	music = data[0]
	mle = data[1]
	for i, point in enumerate(music):
		if point > 180:
			music[i] -= 360
	for i, point in enumerate(mle):
		if point > 180:
			mle[i] -= 360

	print(music, '\n', mle)

	x = [-60] * len(music)

	plt.plot(x, label="real angle")
	plt.plot(music, 'ro', label = "music algorithm")
	plt.plot(mle, 'gx', label = "mle")
	plt.ylim(-180,180)
	plt.ylabel("degree")
	plt.xlabel("snapshot")
	plt.legend()
	plt.savefig("./graphs/300 degree diff.png", dpi=720)
