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
from systemConstants import *


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
	T = 1.0 / main.SAMPLE_RATE  # sample spacing
	x = np.linspace(0.0, N * T, N)
	y = channels[0]

	yf = scipy.fftpack.fft(y)

	channelsFFT = scipy.fftpack.fft(channels)
	angle_of_channels = np.angle(channelsFFT[:, :N // 2])

	abs_of_yf = np.abs(yf[:N // 2])
	angle_of_yf = np.angle(yf[:N // 2])
	findfreq = 2.0 / N * abs_of_yf

	for i in range(len(angle_of_yf)):
		if angle_of_yf[i] != angle_of_channels[0][i]:
			print("error!!!")

	results = []
	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	for i in signal.find_peaks(findfreq)[0]: # 50 - threshold that works for specific example, TODO - find the n(3?) max freqs
		if findfreq[i] > 50 and xf[i] > 50: # xf[i] - the freqs of the signal TODO - add to the threshold the average of the old signals
			results.append((xf[i], angle_of_yf[i], (angle_of_channels[:, i] - angle_of_yf[i])%(2*PI)))

	# fig, ax = plt.subplots()
	# ax.plot(xf, np.abs(yf[:N // 2]))
	# plt.show()
	if results: #and counter < 5:
		# plt.title = "audio without BPF " + str(counter)
		# channel2, = plt.plot(x, channels[2], 'r', label='mic 3')
		# channel1, = plt.plot(x, channels[1], 'g', label='mic 2')
		# channel3, = plt.plot(x, channels[3], 'b', label='mic 4')
		# channel0, = plt.plot(x, channels[0], 'y', label='mic 1')
		# name = "before BPF " + str(counter)
		# plt.legend(handles=[channel0, channel1, channel2, channel3], loc=1)
		# plt.savefig(name + "."+FORMAT_TO_SAVE, format=FORMAT_TO_SAVE, dpi=600)
		# plt.cla()
		for frequency in results:
			angle = frequency[2]
			# print(frequency[0], angle)
			tests = potential_phi(frequency[0])
			norm = []
			for i in range(len(tests)):
				norm.append(np.linalg.norm((tests[i] - angle)))
			index = np.argmin(norm)
			if(frequency[0] < 500):
				print(frequency[0], index, norm[index], tests[index], angle)
			# for i in range(0, 360):
			# 	print(i, tests[i])
			# creating Band-Pass filter TODO - check the numbers :
			# BPF = signal.firwin(50, [angle-5, angle+5], pass_zero=False, nyq=16000.)

			# after_BPF = signal.lfilter(BPF, [1.0], channels)
			# print(after_BPF[0])
			# yfBPF = scipy.fftpack.fft(after_BPF)
			# plt.plot(xf, 2.0 / N * np.abs(yfBPF[0][:N // 2]))
			# plt.show()
			# plt.title = "after BPF graph " + str(counter) + "in angle = " + str(angle)
			# plt.plot(x, after_BPF[2], 'r', label='mic 3')
			# plt.plot(x, after_BPF[1], 'g', label='mic 2')
			# plt.plot(x, after_BPF[3], 'b', label='mic 4')
			# plt.plot(x, after_BPF[0], 'y', label='mic 1')
			# plt.legend(handles=[channel0, channel1, channel2, channel3], loc=1)
			# name = "after BPF " + str(counter) + "with angle " + str(angle)
			# plt.savefig(name + "."+FORMAT_TO_SAVE, format=FORMAT_TO_SAVE, dpi=600)
			# plt.cla()
	# 		plt.show()


def potential_phi(freq):
	lst_to_return = []
	for i in range(360):
		results = []
		results.append(0)
		rads = math.radians(i)
		results.append(((-freq * D * math.cos(rads))/SPEED_OF_SOUND) % (2*PI))
		results.append(((-freq * D * math.sqrt(2) * math.cos((PI / 4) - rads))/SPEED_OF_SOUND) % (2*PI))
		results.append(((-freq * D * math.cos((PI/2) - rads))/SPEED_OF_SOUND) % (2*PI))
		lst_to_return.append(results)
	return lst_to_return
