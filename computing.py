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
import statistics

all_fft = []
average_DB = 0
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
	global average_DB
	peaks_in_data = []
	mode_of_freqs = {}
	for snapshot in lst_of_data:
		# draw_graph(snapshot)
		# results contains: frequency, loction in sanpshot, mean of db.
		results = find_peaks(snapshot[0], average_DB)


		for index in results[1]:
			if index >= 2:  # ignore low frequencies because of nosies
				if index not in mode_of_freqs: # count how many time specific frequency is in the data
					mode_of_freqs[index] = 1
				else:
					mode_of_freqs[index] += 1
		peaks_in_data.append(results)
	# for average DB filtering (meaning - ignore signals that weaker than the average noise around the user)
	# average_DB = ((average_DB * (RATE_OF_AVERAGING - 1)) + results[2]) / (RATE_OF_AVERAGING)
	location_of_real_peaks_in_data = []
	for index in mode_of_freqs:
		if mode_of_freqs[index] >= THRESHOLD_FOR_MODE:
			location_of_real_peaks_in_data.append(index)

	fft_signal = scipy.fftpack.fft(lst_of_data)
	# vector for all relevant frequencies
	temp = fft_signal[:,:, location_of_real_peaks_in_data]

	# each frequency in a special vector
	separated_vector_for_music = []
	for i in range(len(location_of_real_peaks_in_data)):
		# angles_vector = np.angle()
		separated_vector_for_music.append(temp[:, :, i])

	# print(separated_vector_for_music[0])
	N = CHUNK
	T = 1.0 / SAMPLE_RATE
	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	# print(xf[location_of_real_peaks_in_data])
	to_return = []
	# exit()
	global all_fft
	for index, fft_vector in enumerate(separated_vector_for_music):
		# to_return.append(MUSIC_algorithm(fft_vector, xf[location_of_real_peaks_in_data[index]], counter))
		# to_return.append(one_signal_algorithm((xf[location_of_real_peaks_in_data[index]], np.angle(fft_vector))))
		all_fft.append(fft_vector)
	# print(to_return)
	# exit(1)
	# return to_return


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
	# TODO - should I return the db of the peaks? for deciding which freq to choose? no idea
	# print(result[1])
	realDB = result[1]['peak_heights']
	# print(typ)
	if realDB.size == 0:
		realDB = np.append(realDB, [0])
	return [xf[result[0]], result[0], realDB.mean()]






def potential_phi(freq):
	'''
	:param freq: frequency to check
	:return: array, n=360, each cell represent the value that should be in the Vector if the signal come from that angle
	'''
	lst_to_return = []
	for i in range(NUM_OF_DIRECTIONS):
		results = []
		# results.append(0)

		rads = math.radians(i*ANGLE_OF_DIRECTIONS)
		deltaX = [0, D * math.cos(rads), math.sqrt(2) * D * math.cos((PI/4) - rads), D * math.sin(rads)]
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
	'''
	:param vector_of_signals: vector of NUM_OF_SNAPSHOTS_FOR_MUSIC snapshot, each snapshot contain signal in frequency freq
	:param freq: the frequency of the signal
	:param counter: for debug purpose
	:return: the angles where the signals came from
	'''

	''' In this function, N - number of mics, M number of signals'''

	R = np.zeros([NUM_OF_MICS,NUM_OF_MICS], dtype=np.complex64)
	assert len(vector_of_signals) == NUM_OF_SNAPSHOTS_FOR_MUSIC
	for vector in vector_of_signals:
		normalized = vector[0]
		for i in range(len(vector)):
			vector[i] = rect(1, np.angle(vector[i]) - np.angle(normalized))
			# print(np.angle(vector[i]), end="\t")
		# print("\n")
		# print("tada: ", vector)
		nprect = np.vectorize(rect)
		temp = potential_phi(freq)
		s_phi = nprect(1, temp)
		# print("s_phi: ", s_phi[45])
		# print("\n\n\n")
		# exit()
		R += np.outer(vector, vector.conj().T)
	# exit(111)
	R /= NUM_OF_SNAPSHOTS_FOR_MUSIC

	# print(R,"\n\n",np.abs(R),"\n\n" ,np.angle(R), "\n\n\n")
	'''now, R is N*N matrix with rank M. meaning, there is N-M eigenvectors corresponding to the zero eigenvalue'''
	eigenvalues, eigenvectors = np.linalg.eig(R)
	# print(eigenvalues,"\n", eigenvectors)
	# print("\n\n\n\n\n\n")
	idx = eigenvalues.argsort()
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[idx]
	# print("eigenvalues: \n",eigenvalues, "\n\n\n", "eigenvectors: \n", eigenvectors, "\n\n\n\n\n\n\n\n")

	# exit(123)

	# np.set_printoptions(suppress=True,
	#                     formatter={'float_kind': '{:f}'.format})
	# TODO - choose thershold for the eigenvalues, use records for that. using magintuted or dB?

	DB_of_eigenvalues = 20 * scipy.log10(2 / CHUNK * np.abs(eigenvalues))
	# print("db: ", DB_of_eigenvalues, "\tfreq: ", freq)
	# find_num_of_signals(eigenvalues)
	# todo - how to continue? for 1<=m<=3, find the N-M smallest |lambda|s, take their eigenvectors.
    # TODO - equ. 50 from the paper should work, S is a complex vector r = 1, phi = Delta_Phi that we find in potential_phi(). find the M phis that give as the maxest values

	M = 1
	nprect = np.vectorize(rect)
	temp = potential_phi(freq)
	s_phi = nprect(1, temp)
	# print(np.angle(s_phi))

	# just for proving a point:
	# for j in range(len(s_phi)):
	# 	for i in range(4):
	# 		s_phi[j] = rect(np.abs(R[i][i]), temp[j][i])
			# print(R[i][i])
	# print("\n\n\n\n\n")

	# assert (np.abs(np.angle(s_phi) - temp) < 0.0000001).all(), (freq, np.angle(s_phi) - temp)
	# print((s_phi[5]))
	P_MUSIC_phi = []
	# for angle in s_phi:
	# 	P_MUSIC_phi.append(np.square(np.abs(np.dot(test.conj().T,angle))))
	# print(freq, np.argmax(P_MUSIC_phi))
	j = 0
	for angle in s_phi:
		result = 0
		# assert len(eigenvalues) - M == 3
		for i in range(len(eigenvalues) - M):
			result += np.square(np.abs(np.dot(eigenvectors[i], angle.conj().T)))
			# print(i, DB_of_eigenvalues[i], end=" ")
		# print(j)
		j += 1
		P_MUSIC_phi.append(1 / result)
	# x = ANGLE_OF_DIRECTIONS * np.arange(0,NUM_OF_DIRECTIONS,1)
	# plt.plot(x, P_MUSIC_phi)
	# title = str(counter) +" " + str(freq)
	# plt.title(title)
	# plt.show()
	# exit(1)
	# print(P_MUSIC_phi)
	# print(signal.find_peaks(P_MUSIC_phi), ANGLE_OF_DIRECTIONS  )
	final_angle = np.argmax(P_MUSIC_phi) * ANGLE_OF_DIRECTIONS # TODO - return the M maxes, not only 1
	return freq, final_angle
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
	if peaks[0] < 100:
		return
	# print(peaks[0])
	to_return = []
	s_phi = potential_phi(peaks[0])
	if peaks:
		for vector in peaks[1]:
			normalized = vector[0]
			for i in range(len(vector)):
				vector[i] = (vector[i] - normalized) % MOD_2_PI - NOISE_CANCELING

		# vector = np.concatenate(peaks[1])
		# print(peaks[1] % MOD_2_PI)
		# print(s_phi[40])
		final_angle = np.zeros(shape=(1,4), dtype=float)

		for angle in peaks[1]:
			# print(angle)
			final_angle += angle

			# x = ANGLE_OF_DIRECTIONS * np.arange(0,NUM_OF_DIRECTIONS,1)
			# plt.plot(x, norm)
			# title = str(counter) +" " + str(freq)
			# plt.title(title)
			# plt.show()
			# print(norm[np.argmin(norm)])
		# exit(1)
		final_angle /= NUM_OF_SNAPSHOTS_FOR_MUSIC
		# print("avrage: ", final_angle)
		final_angle = final_angle[0]
		# math_angle = [math.acos(final_angle[1])/D, math.acos(final_angle[2])/(math.sqrt(2) * D), math.asin(final_angle[3])/D]
		# print(math_angle)
		# print(s_phi[45])
		# exit(12)
		norm = []
		for i in range(len(s_phi)):
			norm.append(np.linalg.norm((s_phi[i] - final_angle)))
		final_angle = np.argmin(norm)
		print(norm[final_angle])
		# for frequency in peaks:
		# 	angle = frequency[1]
		# 	# print(frequency[0], angle)
		# 	s_phi = potential_phi(frequency[0])
		# 	norm = []
		# 	for i in range(len(s_phi)):
		# 		norm.append(np.linalg.norm((s_phi[i] - angle)))
		# 	index = np.argmin(norm)
		# 	# print(frequency[0], index, norm[index], tests[index], angle)
		# 	# string = str(frequency[0]) + ": the angle is " + str(index) + " and the db is " + str(frequency[1])
		# 	# to_return.append(string)
		to_return.append((peaks[0], final_angle*ANGLE_OF_DIRECTIONS))
		# print(peaks[0], final_angle)
	return to_return


def draw_graph():
	# N = CHUNK
	# T = 1.0 / SAMPLE_RATE  # sample spacing
	# x = np.linspace(0.0, N * T, N)
	#
	# xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	# yf = scipy.fftpack.fft(data)
	# abs_of_yf = np.abs(yf[:N // 2])
	# magnitude_of_frequency = 2.0 / N * abs_of_yf
	# db_of_yf = 20 * scipy.log10(magnitude_of_frequency)
	# angle_mics = np.angle(yf[:N//2])
	# for i, mic in enumerate(db_of_yf):
	# 	title = "mic " + str(i) + " db"
	# 	plt.plot(xf, mic[:N // 2])
	# 	plt.title(title)
	# 	plt.savefig(title + ".png", format="PNG", dpi = 720)
	# 	plt.close()
	#
	# 	title = "mic " + str(i) + " angle"
	# 	plt.plot(xf, angle_mics[i][:N // 2])
	# 	plt.title(title)
	# 	plt.savefig(title + ".png", format="PNG", dpi=720)
	# 	plt.close()
	# exit(13)

	mics = [[],[],[],[]]
	for vector_of_snapshots in all_fft:
		# print(vector_of_snapshots)
		for snapshot in vector_of_snapshots:
			for i, mic in enumerate(snapshot):
				mics[i].append(mic)
	# scipy.fftpack.fft(mics)
	real = np.abs(mics)
	angle = np.angle(mics)

	angle_norm = angle[0]
	angle_normalized = (angle - angle_norm) % MOD_2_PI
	location_for_save = "./graphs/"
	print(angle_normalized)
	for i in range(len(real)):
		if i == 0:
			continue
		title = "90 angle mic " + str(i) + " inner product"
		# plt.plot(real[i], label="real")
		# plt.title(title + " real")
		# plt.savefig(location_for_save + title + " real" + ".png", format=FORMAT_TO_SAVE, dpi=1080, linewidth=0.005)
		# plt.close()
		# plt.plot(20 * scipy.log10(2.0 / CHUNK * real[i]), label="real")
		# plt.title(title + " DB")
		# plt.savefig(location_for_save + title + " db" + ".png", format=FORMAT_TO_SAVE, dpi=1080, linewidth=0.005)
		# plt.close()
		nprect = np.vectorize(rect)
		temp = potential_phi(600)
		for_our_mic = []
		for angle in temp:
			for_our_mic.append(angle[i])
		# exit(123)
		s_phi = nprect(1, for_our_mic)
		angle_from_mic = angle_normalized[i]
		for j in range(len(angle_from_mic)):

			complex_from_mic = rect(1, angle_from_mic[j])

			results = np.inner(s_phi, complex_from_mic)

			plt.plot(results, label="phase normalized")
			# plt.plot(angle[i], label="phase normalized")
			# plt.legend()
		plt.title(title + " each one")
		plt.savefig(location_for_save + title + " each one.png", format=FORMAT_TO_SAVE, dpi=1080, linewidth=0.005)
		# plt.show()
		# exit(123)
		plt.close()
		# plt.plot(angle[i], label="phase")
		# plt.title(title + " phase")
		# plt.savefig(location_for_save + title + " phase" + ".png", format=FORMAT_TO_SAVE, dpi=1080, linewidth=0.005)
		# plt.close()
