import recording
import threading
from collections import deque

RESPEAKER_RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 32
RECORD_BUFFER_MAX = (RESPEAKER_RATE / CHUNK) * RECORD_SECONDS

if __name__ == '__main__':
	frames = deque(RECORD_BUFFER_MAX*[0], RECORD_BUFFER_MAX)
	keepRecording = True
	recordingThread = threading.Thread(group=None, target=recording.record, name="recording thread", args=(frames))
