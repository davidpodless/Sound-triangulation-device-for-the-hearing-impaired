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
from scipy.stats.mstats import gmean
import time
from systemConstants import *


FORMAT_TO_SAVE = 'png'

def extractAndCompute(frames, results):
	'''
	:param frames: deque of the raw data from the mics
	:param results: deque where the thread will save the data
	:return: None
	'''
	# time.sleep(3) # TODO - delete it
	# print("was in extract and compute")
	is_still_empty = False
	thread_counter = 0
	while True:
		# print("computing alive")
		next_sample = 0
		while type(next_sample) == int and frames:
			next_sample = frames.pop()
		if frames:
			# 6 channels in one stream
			data = np.fromstring(next_sample, dtype=np.int16)
			# 4 channels for 4 mics
			ch_data = [np.ndarray]*4
			for i in range(1, 5):
				ch_data[i-1] = data[i::6]
				# print(np.average(ch_data[i-1]))
			# ch_data = ch_data - np.average(ch_data)
			results.appendleft(calc_angle(ch_data, thread_counter))
			thread_counter += 1
		else:
			if is_still_empty:
				break
			else:
				is_still_empty = True
				print("sleeping")
				time.sleep(0.005)


def calc_angle(channels, counter):
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
	# all mics
	channels_fft = scipy.fftpack.fft(channels)
	channels_fft = channels_fft[:, :N // 2]
	angle_of_channels = np.angle(channels_fft[:, :N // 2])


	# mic[0]
	abs_of_yf = np.abs(yf[:N // 2])
	angle_of_yf = np.angle(yf[:N // 2])
	magnitude_of_frequency = 2.0 / N * abs_of_yf
	db_of_yf = 20 * scipy.log10(magnitude_of_frequency)
	for i in range(len(angle_of_yf)):
		if angle_of_yf[i] != angle_of_channels[0][i]:
			print("error!!!")

	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

	# x = scipy.mean(db_of_yf) TODO - creating average for signal detecting
	# main.newNoise = x
	# main.averageNoiseArray.appendleft(x)
	# average()
	k_channels = [0] * 4
	for i in range(4):
		k_channels[i] = np.array_split(channels[i], NUM_OF_SNAPSHOTS_FOR_MUSIC)
	small_chunk = len(k_channels[0][0])
	k_channels_fft = scipy.fftpack.fft(k_channels)
	k_channels_fft = k_channels_fft[:, :, :small_chunk//2]
	k_xf = np.linspace(0.0, 1.0 / (2.0 * T), small_chunk / 2)
	# print(k_xf, xf)
	results = []
	for i in signal.find_peaks(db_of_yf)[0]:  # 40 - threshold that works for specific example,
		# results for MUSIC algorithm:
		# if db_of_yf[i] > 35 and xf[i] > 100:
			# index = (np.abs(k_xf - xf[i])).argmin()
			# MUSIC_algorithm(k_channels_fft[:,:,index], xf[i])
			# MUSIC_array.append(channels_fft[:, i])
			# print(counter)
		# results for one signal algorithm
		if db_of_yf[i] > 35 and xf[i] > 100 and db_of_yf[i] > main.averageNoise:  # xf[i] - the freqs of the signal TODO - add to the threshold the average of the old signals
			'''this line need a little bit explanation:
				xf[i] is the frequency of the signal, the angle of yf[i] is the angle in the raw data of mic[0]
				the last element is the reletive angle between the first angle and the i-th angle modulus 2PI
				  - in complex numbers, multi in polar mode become +/- of the angle'''
			results.append((xf[i], db_of_yf[i], (angle_of_channels[:, i] - angle_of_yf[i]) % (2 * PI)))








	# fig, ax = plt.subplots()
	# ax.plot(xf, db_of_yf)
	# plt.show()
	return one_signal_algorithm(results)




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


def MUSIC_algorithm(vector_of_signals, freq):
	# print(len(vector_of_signals.transpose()), vector_of_signals.transpose())
	# In this function, N - number of mics, M number of signals
	R = np.zeros([4,4], dtype=np.complex64)
	# print(R)
	for vector in vector_of_signals.transpose():
		R += np.outer(vector, vector.conj().T)
		# print(R)
	R /= NUM_OF_SNAPSHOTS_FOR_MUSIC
	# print(R)
	# print("finito")
	# now, R is N*N matrix with rank M. meaning, there is N-M eigenvectors corresponding to the zero eigenvalue
	eigenvalues, eigenvectors = np.linalg.eig(R)
	Lambda = np.diag(eigenvalues)
	# print(freq, np.abs(eigenvalues))
	find_num_of_signals(eigenvalues)
	# todo - how to continue? the instructions are not clear


def find_num_of_signals(eigenvalues): # todo - ask or about this
	N = len(eigenvalues)
	print(np.abs(eigenvalues))
	MDL = []
	for d in range(N-1):
		L = -NUM_OF_SNAPSHOTS_FOR_MUSIC * (NUM_OF_MICS - 1) * np.log10(gmean(eigenvalues[d:N-1]) / np.mean(eigenvalues[d:N-1]))
		MDL.append(L + 0.5*d*(2*NUM_OF_MICS - d)*np.log10(NUM_OF_SNAPSHOTS_FOR_MUSIC))
		# print(L)
	index = np.real(MDL).argmin()
	# print(len(MDL))
	return index


def one_signal_algorithm(results):
	to_return = []
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
			# print(frequency[0], index, norm[index], tests[index], angle)
			# string = str(frequency[0]) + ": the angle is " + str(index) + " and the db is " + str(frequency[1])
			# to_return.append(string)
			to_return.append((frequency[0], index, frequency[1]))

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
	return to_return
