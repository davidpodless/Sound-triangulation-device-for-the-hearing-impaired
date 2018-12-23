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

def fake_record(file, frames):
	wav = wave.open(file)
	print(wav.getnchannels())
	print(wav.getsampwidth())
	print(wav.getnframes())
	wav.readframes(CHUNK)
	wav.readframes(CHUNK)
	wav.readframes(CHUNK)
	data = wav.readframes(CHUNK)
	frames.appendleft(data)


if __name__ == '__main__':
	frames = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	results = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	keepRecording = True
	# recordingThread = threading.Thread(group=None, target=recording.record, name="recording thread", args=(frames))
	computingThread = threading.Thread(group=None, target=computing.extractAndCompute, name="compute thread", args=(frames, results))
	fake_record('./output2.wav', frames)
	# recordingThread.run()
	computingThread.run()