from os import listdir
from os.path import isfile, join
import threading
from collections import deque
import wave
from scipy.io import wavfile
import numpy as np

# import recording
import computing

RESPEAKER_RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 32
RECORD_BUFFER_MAX = (RESPEAKER_RATE * RECORD_SECONDS) // CHUNK
keepRecording = True

def getFileslist():
	# return [f for f in listdir('./') if isfile(join('./', f)) and f.endswith(".wav")]
	return ['./500HZ sine 0 angle.wav']



def fake_record(files, frames):
	print("was in fake record")
	check = False
	for file in files:
		if check:
			break
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
				check = True
				break
		# print(frames.pop())


if __name__ == '__main__':
	# print(RECORD_BUFFER_MAX)
	frames = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	results = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	# recordingThread = threading.Thread(group=None, target=recording.record, name="recording thread", args=(frames, results))
	fakeRecordingThread = threading.Thread(group=None, target=fake_record, name="fake recording thread", args=(getFileslist(), frames))
	computingThread = threading.Thread(group=None, target=computing.extractAndCompute, name="compute thread", args=(frames, results))
	# recordingThread.start()
	fakeRecordingThread.start()
	computingThread.start()

