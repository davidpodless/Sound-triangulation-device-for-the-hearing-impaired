from usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time
import pyaudio
import wave
import numpy as np
# from scipy.io import wavfile
import binascii
import main

def extractAndCompute(frames, results):
	while True:
		if(frames):
			nextSample = frames.pop()
			if(nextSample):
				channel1 = np.frombuffer(nextSample[2::6], int32)
				channel2 = np.frombuffer(nextSample[3::6], int32)
				channel3 = np.frombuffer(nextSample[4::6], int32)
				channel4 = np.frombuffer(nextSample[5::6], int32)
				results.appendleft(compute(channel1, channel2, channel3, channel4))
			else:
				continue
		else:
			continue


def compute(channel1, channel2, channel3, channel4):
	freq1 = np.fft(channel1)
