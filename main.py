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

keepRecording = True
averageNoiseArray = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
averageNoise = 0
newNoise = 0

def getFileslist():
	# return [f for f in listdir('./') if isfile(join('./', f)) and f.endswith(".wav")]
	return ['./600HZ sine 45 angle & 450 sine 90 angle v2.wav']



def fake_record(files, frames):
	print("was in fake record")
	# lastFileIndex = len(files)
	# counterFileIndex = 0
	# check = False
	for file in files:
		# if check:
		# 	break
		wav = wave.open(file)
		print(file.title())
		# print(wav.getnchannels())
		# print(wav.getsampwidth())
		# print(wav.getnframes())
		while True:
			data = wav.readframes(CHUNK)
			if data != b'':
				frames.appendleft(data)
			else:
				# counterFileIndex += 1
				# if counterFileIndex == lastFileIndex:
				# 	check = True
				break
		# print(frames.pop())


if __name__ == '__main__':
	# print(RECORD_BUFFER_MAX) todo: delete this
	frames = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	results = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	# recordingThread = threading.Thread(group=None, target=recording.record, name="recording thread", args=(frames, results))
	fakeRecordingThread = threading.Thread(group=None, target=fake_record, name="fake recording thread", args=(getFileslist(), frames))
	computingThread = threading.Thread(group=None, target=computing.extractAndCompute, name="compute thread", args=(frames, results))

	# recordingThread.start()
	fakeRecordingThread.start()
	# fakeRecordingThread.join()
	computingThread.start()


	computingThread.join()
	while results:
		toPrint = results.pop()
		if toPrint == 0:
			continue
		print(toPrint)

