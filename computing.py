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
import time

FORMAT_TO_SAVE = 'png'

def extractAndCompute(frames, results):
	# time.sleep(3)
	print("was in extract and compute")
	is_still_empty = False
	thread_counter = 0
	while True:
		# print("computing alive")
		nextSample = 0
		while type(nextSample) == int and frames:
			nextSample = frames.pop()
		if(frames):
			# print(type(nextSample))
			# if(nextSample != 0):
			data = np.fromstring(nextSample, dtype=np.int16)
			# print(data)
			# channel1 = data[1::6]
			ch_data = [np.ndarray]*4
			for i in range(1, 5):
				ch_data[i-1] = data[i::6]
			results.appendleft(compute(ch_data, thread_counter))
			thread_counter += 1
			# else:
			# 	continue
		else:
			if is_still_empty:
				break
			else:
				is_still_empty = True
				print("sleeping")
				time.sleep(0.005)


def compute(channels, counter):
	N = main.CHUNK
	T = 1.0 / 16000.0  # sample spacing
	x = np.linspace(0.0, N * T, N)
	y = channels[0]

	yf = scipy.fftpack.fft(y)

	findfeq = 2.0 / N * np.abs(yf[:N // 2])
	results = []
	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	for i in range(len(findfeq)):
		if findfeq[i] > 50 and xf[i] > 100:  # 50 - threshold that works for specific example, TODO - find the n(3?) max freqs
			results.append((xf[i], findfeq[i]))  # xf[i] - the freqs of the signal
	# for i in range(len(findfeq)):
	# 	if xf[i] > 50:
	# 		results.append((xf[i], findfeq[i]))
	# results.sort(key=lambda tup: tup[1], reverse=False)
	# print(results.pop())
	# print(results[-3:-1])

	# fig, ax = plt.subplots()
	# ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
	# plt.show()
	if results and counter < 5:
		plt.title = "audio without BPF " + str(counter)
		channel2, = plt.plot(x, channels[2], 'r', label='mic 3')
		channel1, = plt.plot(x, channels[1], 'g', label='mic 2')
		channel3, = plt.plot(x, channels[3], 'b', label='mic 4')
		channel0, = plt.plot(x, channels[0], 'y', label='mic 1')
		name = "before BPF " + str(counter)
		plt.legend(handles=[channel0, channel1, channel2, channel3], loc=1)
		plt.savefig(name + "."+FORMAT_TO_SAVE, format=FORMAT_TO_SAVE, dpi=600)
		plt.cla()
		for frequency in results:
			freq = frequency[0]

			BPF = signal.firwin(200, [freq-5, freq+5], pass_zero=False, nyq=16000.)  # creating Band-Pass filter TODO - check the numbers
			after_BPF = signal.lfilter(BPF, [1.0], channels)
			plt.title = "after BPF graph " + str(counter) + "in freq = " + str(freq)
			plt.plot(x, after_BPF[2], 'r', label='mic 3')
			plt.plot(x, after_BPF[1], 'g', label='mic 2')
			plt.plot(x, after_BPF[3], 'b', label='mic 4')
			plt.plot(x, after_BPF[0], 'y', label='mic 1')
			plt.legend(handles=[channel0, channel1, channel2, channel3], loc=1)
			name = "after BPF " + str(counter) + "with freq " + str(freq)
			plt.savefig(name + "."+FORMAT_TO_SAVE, format=FORMAT_TO_SAVE, dpi=600)
			plt.cla()
			# plt.show()
