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
import scipy.fftpack
from scipy import signal


def extractAndCompute(frames, results):
	while True:
		print("I am ALIVE")
		nextSample = 0
		if(frames):
			while type(nextSample) == int:
				nextSample = frames.pop()
			# if(nextSample != 0):
			data = np.fromstring(nextSample, dtype=np.int16)
			print(data)
			# channel1 = data[1::6]
			ch_data = [np.ndarray]*4
			for i in range(1, 5):
				ch_data[i-1] = data[i::6]
			results.appendleft(compute(ch_data))
			# else:
			# 	continue
		else:
			continue


def compute(channels):
	# print(type(channels[0]))
	# print(channel1.shape)
	# b=[(ele/2**8.)*2-1 for ele in channel1]
	# freq1 = np.fft.fft(channels[0])
	# print(freq1)
	# print(freq1)
	# d = (len(freq1)) // 2
	# plt.plot(channels)
	# plt.xlim(-len(freq1), len(freq1))
	# plt.show()
	N = main.CHUNK
	# sample spacing
	T = 1.0 / 16000.0
	x = np.linspace(0.0, N * T, N)
	y = channels[0]
	yf = scipy.fftpack.fft(y)

	findfeq = 2.0 / N * np.abs(yf[:N // 2])
	results = []
	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	for i in range(len(findfeq)):
		if findfeq[i] > 50 and xf[i] > 100:  # 50 - threshold that works for specific example, TODO - find the n(3?) max freqs
			results.append((xf[i], findfeq[i]))  # xf[i] - the freqs of the signal

	# fig, ax = plt.subplots()
	# ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
	# plt.show()

	freq = results[0][0]

	BPS = signal.firwin(150, [freq-50, freq+50], pass_zero=False, nyq=16000.)  # creating Band-Pass filter TODO - check the numbers
	afterBPS = signal.lfilter(BPS, [1.0], channels)
	plt.plot(afterBPS[2], 'r')
	plt.plot(afterBPS[1], 'g')
	plt.plot(afterBPS[3], 'b')
	plt.plot(afterBPS[0], 'y')
	plt.show()
