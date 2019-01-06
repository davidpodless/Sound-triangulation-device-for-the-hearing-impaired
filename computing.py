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
	'''
	:param frames: deque of the raw data from the mics
	:param results: deque where the thread will save the data
	:return: None
	'''
	# time.sleep(3)
	# print("was in extract and compute")
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
	'''
	:param channels: array, n=4, in each cell the signal from the ith mic
	:param counter: for testing, counting how much into the signal the compute will go
	:return: the frequency and the angles of the signal. in case where there is more than one frequency - for each one
	'''
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
	db_of_yf = 20 * scipy.log10(findfreq)
	for i in range(len(angle_of_yf)):
		if angle_of_yf[i] != angle_of_channels[0][i]:
			print("error!!!")

	results = []
	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	# x = scipy.mean(db_of_yf) TODO - creating average for signal detecting
	# main.newNoise = x
	# main.averageNoiseArray.appendleft(x)
	# average()
	for i in signal.find_peaks(db_of_yf)[0]: # 40 - threshold that works for specific example,
		if db_of_yf[i] > 35 and xf[i] > 50 and db_of_yf[i] > main.averageNoise: # xf[i] - the freqs of the signal TODO - add to the threshold the average of the old signals
			'''this line need a little bit explanation:
				xf[i] is the frequency of the signal, the angle of yf[i] is the angle in the raw data
				the last element is the reletive angle between the first angle and the i-th angle modulus 2PI
				  - in complex numbers, multi in polar mode become +/- of the angle'''
			results.append((xf[i], angle_of_yf[i], (angle_of_channels[:, i] - angle_of_yf[i])%(2*PI)))




	# fig, ax = plt.subplots()
	# ax.plot(xf, db_of_yf)
	# plt.show()
	toReturn = []
	if results:
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
			print(frequency[0], index, norm[index], tests[index], angle)
			toReturn.append((frequency[0], index))
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
	return toReturn


def potential_phi(freq):
	'''
	:param freq: frequency to check
	:return: array, n=360, each cell represent the value that should be in the Vector if the signal come from that angle
	'''
	lst_to_return = []
	for i in range(360):
		results = []
		# results.append(0)

		rads = math.radians(i)
		deltaX = [0, D * math.cos(rads), math.sqrt(2) * D * math.cos((PI/4) - rads), D * math.cos(PI/2 - rads)]
		# time approach:
		# results.append(((-freq * D * math.cos(rads))/SPEED_OF_SOUND))
		# results.append(((-freq * D * math.sqrt(2) * math.cos((PI / 4) - rads))/SPEED_OF_SOUND))
		# results.append(((-freq * D * math.cos((PI/2) - rads))/SPEED_OF_SOUND))

		# phase approach:
		phaseChange = (2*PI*freq / SPEED_OF_SOUND)
		# phaseChangeArray = [phaseChange, phaseChange, phaseChange, phaseChange]
		for dx in deltaX:
			# print(dx, phaseChange, dx*phaseChange)
			results.append(dx * phaseChange)
		# print(results)

		lst_to_return.append(results)
	return lst_to_return


def average():
	oldValue = main.averageNoiseArray.pop()
	main.averageNoise = main.averageNoise - (oldValue / main.RECORD_BUFFER_MAX) + (
				main.newNoise / main.RECORD_BUFFER_MAX)
	print("average is: ", main.averageNoise)
	print("old value is: ", oldValue)
	print("new value is: ", main.newNoise)