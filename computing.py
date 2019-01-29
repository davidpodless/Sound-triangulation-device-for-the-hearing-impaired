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
from cmath import rect


FORMAT_TO_SAVE = 'png'

# TODO - instead of one CHUNK at a time, we need to take NUM_OF_SNAPSHOTS_FOR_MUSIC CHUNKs, we will compute fft for each one, the magnitude and the db of the signal, and the average then find the peaks of the average and find the sample corrolate with the average pick and use them for the MUSIC/FFT BASE algorithms
# TODO - possible way: the recording Thread should append to a list NUM_OF_SNAPSHOTS_FOR_MUSIC CHUNKs, and append the result to the deque. in this senerio - we need to do the splitting to 4 channels diffrently. ADVENATEGE - can't get "error" in the number of sample that we have, CONS - takes more time to start the signal proccessing.
# TODO - what happens if there is no sound to be found?


def extract_data(frames, results):
	'''
	:param frames: deque of the raw data from the mics
	:param results: deque where the thread will save the data
	:return: None
	'''
	is_still_empty = False
	thread_counter = 0
	while True:
		# print("computing alive")
		next_sample = 0
		while type(next_sample) == int and frames:
			next_sample = frames.pop()
		if frames:
			list_of_data_sent_to_calc = []
			for frame in next_sample:
				# 6 channels in one stream
				np_data = np.fromstring(frame, dtype=np.int16)
				# 4 channels for 4 mics
				ch_data = [np.ndarray]*4
				for i in range(1, 5):
					ch_data[i-1] = np_data[i::6]
					# print(np.average(ch_data[i-1]))
				# ch_data = ch_data - np.aveage(ch_data)
				list_of_data_sent_to_calc.append(ch_data)
			results.appendleft(calc_angle(list_of_data_sent_to_calc, thread_counter))
			thread_counter += 1
		else:
			if is_still_empty:
				break
			else:
				is_still_empty = True
				print("sleeping")
				time.sleep(0.005)


def calc_angle(lst_of_data, counter):
	'''
	:param lst_of_data: list of NUM_OF_SNAPSHOTS_FOR_MUSIC arrays, for each array: n=4, in each cell the signal from the ith mic
	:param counter: for testing, counting how much into the signal the compute will go
	:return: the frequency and the angles of the signal. in case where there is more than one frequency - for each one
	'''
	peaks_in_data = []
	mode_of_freqs = {}
	for snapshot in lst_of_data:
		results = find_peaks(snapshot[0], 0)
		for index in results[1]:
			if index >= 2:
				if index not in mode_of_freqs:
					mode_of_freqs[index] = 1
				else:
					mode_of_freqs[index] += 1
		peaks_in_data.append(results)

	# mean_of_db_for_this_snapshot = np.array([x[2] for x in lst], dtype=np.float64).mean()
	lst = []
	for index in mode_of_freqs:
		if mode_of_freqs[index] >= THRESHOLD_FOR_MODE:
			lst.append(index)
	# print(lst, peaks_in_data)
	fft_signal = scipy.fftpack.fft(lst_of_data)
	temp = fft_signal[:,:, lst]
	seperated_vector_for_music = []
	for i in range(len(lst)):
		# angles_vector = np.angle()
		seperated_vector_for_music.append(temp[:, :, i])

	# print(seperated_angles_vector_for_music[0])
	N = CHUNK
	T = 1.0 / SAMPLE_RATE
	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	for index, fft_vector in enumerate(seperated_vector_for_music):
		MUSIC_algorithm(fft_vector, xf[lst[index]], counter)

	# N = main.CHUNK
	# T = 1.0 / main.SAMPLE_RATE  # sample spacing
	#
	# # all mics
	# channels_fft = scipy.fftpack.fft(lst_of_data)
	# channels_fft = channels_fft[:, :N // 2]
	# angle_of_channels = np.angle(channels_fft[:, :N // 2])
	# # xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	#
	# # x = scipy.mean(db_of_yf) TODO - creating average for signal detecting
	# # main.newNoise = x
	# # main.averageNoiseArray.appendleft(x)
	# # average()
	#
	# results = []
	# for i in peaks_in_data:  # 40 - threshold that works for specific example,
	# 	# results for MUSIC algorithm:
	# 	vec = []
	# 	print(i)
	#
	# 	for j in range(len(angle_of_channels)):
	# 		temp = angle_of_channels[j,:,i]
	# 		vec.append((temp - temp[0]) % (2*PI))
	# 	print(vec, len(vec))
	# 	# index = (np.abs(k_xf - xf[i])).argmin()
	# 	# MUSIC_algorithm(k_channels_fft[:,:,index], xf[i])
	# 	# MUSIC_array.append(channels_fft[:, i])
	# 	# print(counter)
	#
	# 	# results for one signal algorithm
	# 	'''this line need a little bit explanation:
	# 		xf[i] is the frequency of the signal,
	# 		the last element is the reletive angle between the first angle and the i-th angle modulus 2PI
	# 		  - in complex numbers, multi in polar mode become +/- of the angle'''
	# 	# results.append((xf[i], (angle_of_channels[0][:, i] - angle_of_channels[0][0][i]) % (2 * PI)))
	# # fig, ax = plt.subplots()
	# # ax.plot(xf, db_of_yf)
	# # plt.show()
	# # return one_signal_algorithm(results)
	# exit(123)


def find_peaks(raw_signal, avr):
	'''
	:param raw_signal: raw signal from the mics
	:param avr: the db average of the signal for the last RECORD_SECONDS seconds
	:return: array [list of the freq peaks in the signal, the location in the array of it, the fft of those locations, the average of the db of the signal]
	'''
	N = CHUNK
	T = 1.0 / SAMPLE_RATE  # sample spacing
	x = np.linspace(0.0, N * T, N)

	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	yf = scipy.fftpack.fft(raw_signal)
	abs_of_yf = np.abs(yf[:N // 2])
	magnitude_of_frequency = 2.0 / N * abs_of_yf
	db_of_yf = 20 * scipy.log10(magnitude_of_frequency)
	result = signal.find_peaks(db_of_yf, height=max(30, avr))
	return [xf[result[0]], result[0],  db_of_yf.mean()]






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


def MUSIC_algorithm(vector_of_signals, freq, counter):
	# print(vector_of_signals)
	# print(len(vector_of_signals), vector_of_signals)
	# In this function, N - number of mics, M number of signals
	R = np.zeros([4,4], dtype=np.complex64)
	# print(R)
	for vector in vector_of_signals:
		R += np.outer(vector, vector.conj().T)
		# print(R)
	R /= NUM_OF_SNAPSHOTS_FOR_MUSIC
	# print(R, "\n\n\n")
	# print("finito")
	# now, R is N*N matrix with rank M. meaning, there is N-M eigenvectors corresponding to the zero eigenvalue
	eigenvalues, eigenvectors = np.linalg.eig(R)
	# print(eigenvalues, eigenvectors, "\n\n\n")
	idx = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:, idx]
	test = eigenvectors[-1]
	np.delete(eigenvectors, -1)
	np.delete(eigenvectors, -1)


	# np.set_printoptions(suppress=True,
	#                     formatter={'float_kind': '{:f}'.format})
	# TODO - choose thershold for the eigenvalues, use records for that. using magintuted or dB?
	# print((2 / CHUNK * np.real(eigenvalues)))
	# Lambda = np.diag(eigenvalues)
	# print(Lambda)
	# print(R)
	# print(freq, np.abs(eigenvalues))
	# find_num_of_signals(eigenvalues)
	# todo - how to continue? for 1<=m<=3, find the N-M smallest |lambda|s, take their eigenvectors.
    # TODO - equ. 50 from the paper should work, S is a complex vector r = 1, phi = Delta_Phi that we find in potential_phi(). find the M phis that give as the maxest values
	M = 1

	nprect = np.vectorize(rect)

	s_phi = nprect(1, potential_phi(freq))
	assert (np.abs(np.angle(s_phi) - potential_phi(freq)) < 0.000000001).all()
	# print((s_phi[5]))
	P_MUSIC_phi = []
	# for angle in s_phi:
	# 	P_MUSIC_phi.append(np.square(np.abs(np.dot(test.conj().T,angle))))
	# print(freq, np.argmax(P_MUSIC_phi))
	for angle in s_phi:
		result = sum(np.square(np.abs(np.dot(eigenvectors.conj().T, angle))))
		P_MUSIC_phi.append(1 / result)
	plt.plot(range(360), P_MUSIC_phi)
	plt.title(counter)
	plt.show()
	final_angle = np.argmax(P_MUSIC_phi)
	print(freq, final_angle, P_MUSIC_phi[0], P_MUSIC_phi[int(final_angle)])
	# exit(1)


def find_num_of_signals(eigenvalues): # todo - ask orr about this
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


def one_signal_algorithm(peaks):
	'''
	:param peaks: list of tuples (freq, db of the signal, angle) that represent peaks in frequency
	:return: the direction which the signal come from in a tuple (freq, direction, db of the signal)
	this is the naive and not necessarily work approach.
	'''
	to_return = []
	if peaks:
		# plt.title = "audio without BPF " + str(counter)
		# channel2, = plt.plot(x, channels[2], 'r', label='mic 3')
		# channel1, = plt.plot(x, channels[1], 'g', label='mic 2')
		# channel3, = plt.plot(x, channels[3], 'b', label='mic 4')
		# channel0, = plt.plot(x, channels[0], 'y', label='mic 1')
		# name = "before BPF " + str(counter)
		# plt.legend(handles=[channel0, channel1, channel2, channel3], loc=1)
		# plt.savefig(name + "."+FORMAT_TO_SAVE, format=FORMAT_TO_SAVE, dpi=600)
		# plt.cla()
		for frequency in peaks:
			angle = frequency[1]
			# print(frequency[0], angle)
			tests = potential_phi(frequency[0])
			norm = []
			for i in range(len(tests)):
				norm.append(np.linalg.norm((tests[i] - angle)))
			index = np.argmin(norm)
			# print(frequency[0], index, norm[index], tests[index], angle)
			# string = str(frequency[0]) + ": the angle is " + str(index) + " and the db is " + str(frequency[1])
			# to_return.append(string)
			to_return.append((frequency[0], index))

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
