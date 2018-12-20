from usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time
# import pyaudio
import wave
import numpy as np
# from scipy.io import wavfile
import binascii
import main
import matplotlib.pyplot as plt


def extractAndCompute(frames, results):
	while True:
		nextSample = 0
		if(frames):
			while type(nextSample) == int:
				nextSample = frames.pop()
			# if(nextSample != 0):
			channel1 = nextSample[2::6]
			channel2 = nextSample[3::6]
			channel3 = nextSample[4::6]
			channel4 = nextSample[5::6]

			channel1 = nextSample[1::2]
			channel2 = nextSample[2::2]
			if(type(nextSample) != np.ndarray):
				channel1 = np.frombuffer(nextSample[2::6], int16)
				channel2 = np.frombuffer(nextSample[3::6], int16)
				channel3 = np.frombuffer(nextSample[4::6], int16)
				channel4 = np.frombuffer(nextSample[5::6], int16)
			results.appendleft(compute(channel1, channel2, channel3, channel4))
			# else:
			# 	continue
		else:
			break


def compute(channel1, channel2, channel3, channel4):
	print(channel1)
	b=[(ele/2**8.)*2-1 for ele in channel1]
	freq1 = np.fft.fft2(b)
	# print(freq1)
	d = (len(freq1)) // 2
	plt.plot(abs(freq1))
	# plt.xlim(-len(freq1), len(freq1))
	plt.show()
