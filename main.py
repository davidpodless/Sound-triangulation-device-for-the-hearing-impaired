from os import listdir
from os.path import isfile, join
import threading
from collections import deque
import wave
from scipy.io import wavfile
import numpy as np

# import recording
import computing
from systemConstants import *
'''this program uses MKS system'''

CHUNK_RECORDING = True
keepRecording = True
averageNoiseArray = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
averageNoise = 0
newNoise = 0

def getFileslist():
	# return [f for f in listdir('./') if isfile(join('./', f)) and f.endswith(".wav")]
	return ['./500HZ sine 0 angle.wav']



def fake_record(files, frames):
	print("was in fake record")
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
	while results:
		toPrint = results.pop()
		if toPrint == 0:
			continue
		# print(toPrint)

