import threading
from collections import deque
import wave
# import recording
import computing
from systemConstants import *
import graphics.GraphicRunnerThread as GraphicRunnerThread
import graphics.SoundCircle as SoundCircle
import graphics.DataCollectingThread as DataCollectingThread
import time


'''this program uses MKS system'''

CHUNK_RECORDING = True
keepRecording = True
averageNoiseArray = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
averageNoise = 0
newNoise = 0


def getFileslist():
	# return [f for f in listdir('./wav_files') if isfile(join('./', f)) and f.endswith(".wav")]
	return ['./wav_files/48k_600_45_output.wav']


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
				time.sleep(0.001)


def main():
	frames = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	results = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	# recordingThread = threading.Thread(group=None, target=recording.record, name="recording thread", args=(frames, results))
	computingThread = threading.Thread(group=None, target=computing.extract_data, name="compute thread", args=(frames, results))
	fakeRecordingThread = threading.Thread(group=None, target=fake_record, name="fake recording thread", args=(getFileslist(), frames))

	fakeRecordingThread.start()

	# recordingThread.start()
	computingThread.start()

	dataHandleThread = DataCollectingThread.DataCollectingThread([results], name='Data thread')
	dataHandleThread.start()

	# graphicThread = GraphicRunnerThread.GraphicRunnerThread(name='Graphic thread')
	# graphicThread.start()
	graphicThread = SoundCircle.SoundCircleApp()
	graphicThread.run()

	computingThread.join()


if __name__ == '__main__':
	main()