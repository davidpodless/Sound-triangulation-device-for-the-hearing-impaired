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
frequency_for_draw = 0
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
		if next_sample:
			list_of_data_sent_to_calc = []
			for frame in next_sample:
				# 6 channels in one stream
				np_data = np.fromstring(frame, dtype=np.int16)
				# 4 channels for 4 mics
				ch_data = [np.ndarray]*4
				for i in range(1, 5):
					ch_data[i-1] = np_data[i::6]
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
	:param lst_of_data: list of NUM_OF_SNAPSHOTS_FOR_MUSIC arrays, for each array: n=4, in each cell the signal from the i-th mic
	:param counter: for testing, counting how much into the signal the compute will go
	:return: the frequency and the angles of the signal. in case where there is more than one frequency - for each one
	'''
	global average_DB
	peaks_in_data = []
	mode_of_freqs = {}
	N = CHUNK
	T = 1.0 / SAMPLE_RATE
	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	# find the frequencies of a signal
	for snapshot in lst_of_data:
		# draw_graph(snapshot)
		# results contains: frequency, loction in sanpshot, mean of db.
		results = find_peaks(snapshot[0], average_DB)

		for index in results[1]:
			if xf[index] >= 100:  # ignore low frequencies because of nosies
				if index not in mode_of_freqs: # count how many time specific frequency is in the data
					mode_of_freqs[index] = 1
				else:
					mode_of_freqs[index] += 1
		peaks_in_data.append(results)

	# for average DB filtering (meaning - ignore signals that weaker than the average noise around the user)
	# average_DB = ((average_DB * (RATE_OF_AVERAGING - 1)) + results[2]) / (RATE_OF_AVERAGING)
	location_of_real_peaks_in_data = []
	for index in mode_of_freqs:
		# print(xf[index], mode_of_freqs[index])
		if mode_of_freqs[index] >= THRESHOLD_FOR_MODE:
			# print(xf[index])
			location_of_real_peaks_in_data.append(index)
	# print(xf)
	# exit()
	fft_signal = scipy.fftpack.fft(lst_of_data)
	fft_signal = fft_signal[:,:,:N // 2]
	# vector for all relevant frequencies
	# temp = fft_signal[:,:, location_of_real_peaks_in_data]
	separated_vector_for_music = []
	for i in location_of_real_peaks_in_data:
		# each frequency in a special vector
		# angles_vector = np.angle()
		separated_vector_for_music.append(fft_signal[:, :, i])

	# print(separated_vector_for_music[0])

	# print(xf[location_of_real_peaks_in_data])
	to_return = []
	# exit()
	global all_fft
	global frequency_for_draw
	for index, fft_vector in enumerate(separated_vector_for_music):
		to_return.append(MUSIC_algorithm(fft_vector, xf[location_of_real_peaks_in_data[index]], 20 * scipy.log10(2.0 / N * np.abs(fft_vector))))
		# to_return.append(one_signal_algorithm((xf[location_of_real_peaks_in_data[index]], np.angle(fft_vector), 20 * scipy.log10(2.0 / N * np.abs(fft_vector)))))
		# if(xf[location_of_real_peaks_in_data[index]] < 2000):
		# 	frequency_for_draw = xf[location_of_real_peaks_in_data[index]]
		# 	all_fft.append(fft_vector)
	print(to_return)
	# exit(1)
	return to_return


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
	# print(len(abs_of_yf), len(yf))
	# exit(12)
	magnitude_of_frequency = 2.0 / N * abs_of_yf
	db_of_yf = 20 * scipy.log10(magnitude_of_frequency)
	result = signal.find_peaks(db_of_yf, height=max(30, avr))
	# TODO - should I return the db of the peaks? for deciding which freq to choose? no idea
	# print(result[1])
	# plt.plot(xf, db_of_yf)
	# plt.show()
	# exit()
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


def MUSIC_algorithm(vector_of_signals, freq, db_of_signal):
	'''
	:param vector_of_signals: vector of NUM_OF_SNAPSHOTS_FOR_MUSIC snapshot, each snapshot contain signal in frequency freq
	:param freq: the frequency of the signal
	:param counter: for debug purpose
	:return: the angles where the signals came from
	'''

	''' In this function, N - number of mics, M number of signals'''
	nprect = np.vectorize(rect)
	s_phi = nprect(1, potential_phi(freq))
	R = np.zeros([NUM_OF_MICS,NUM_OF_MICS], dtype=np.complex64)
	assert len(vector_of_signals) == NUM_OF_SNAPSHOTS_FOR_MUSIC
	for vector in vector_of_signals:
		normalized = vector[0]
		for i in range(len(vector)):
			vector[i] = rect(1, np.angle(vector[i]) - np.angle(normalized))
			# print(np.angle(vector[i]), end="\t")
		# print("\n")
		# print("tada: ", vector)

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

	M = 0
	for i in np.abs(eigenvalues):
		print(i)
	# exit(12)
	for i in eigenvalues:
		if np.abs(i) > 0.1: # TODO: when using pure sine - the lambdas are *very* small, there are a lot more noise when the siganl is not pure. is that an indacation that this is not the correct angle?
			# TODO - the stupid way to check - run on all the frequencies and check which one will result in the correct angle.
			# TODO 3: record two pure signals from two different angles, what is the values of the np.abs(i)s?
			M += 1
			print(i)
		else:
			print(i)
	# if M == 4:
	# 	print(np.abs(eigenvalues))
		# raise Exception
	# exit(1)

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
	j = 0
	super_result = 0
	for index, angle in enumerate(s_phi):
		result = 0
		# assert len(eigenvalues) - M == 3
		for i in range(len(eigenvalues) - M):
			# print(np.abs(np.vdot(eigenvectors[i].T, angle)))
			result += np.square(np.abs(np.vdot(eigenvectors[i].T, angle)))
			# if index == 45 or index == 315:
			# 	print(index, angle, np.square(np.abs(np.vdot(eigenvectors[i].T, angle))))

			# print(i, DB_of_eigenvalues[i], end=" ")
		# if index == 45 or index == 330:
		# 	print(index, 1 / result)
		# print(result)
		super_result += result
		j += 1
		P_MUSIC_phi.append(1 / result)
	# print("average: ", super_result / (len(s_phi) * NUM_OF_MICS - M))
	x = ANGLE_OF_DIRECTIONS * np.arange(0,NUM_OF_DIRECTIONS,1)
	plt.plot(x, P_MUSIC_phi)
	# title = str(counter) +" " + str(freq)
	# plt.title(title)
	# plt.show()
	# exit(1)
	# print(P_MUSIC_phi)
	# print(signal.find_peaks(P_MUSIC_phi), ANGLE_OF_DIRECTIONS  )
	final_angle = np.argmax(P_MUSIC_phi) * ANGLE_OF_DIRECTIONS # TODO - return the M maxes, not only 1
	return freq, final_angle, statistics.mean(gmean(db_of_signal))
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
	:param peaks: list of tuples (freq, angle, db of the signal) that represent peaks in frequency
	:return: the direction which the signal come from in a tuple (freq, direction, db of the signal)
	this is the naive and not necessarily work approach.
	'''

	to_return = []
	nprect = np.vectorize(rect)
	s_phi = nprect(1,potential_phi(peaks[0]))
	if peaks:
		if peaks[0] < 100:
			return
		final_angle = rect(0, 1)
		counter = 0
		for snapshot in peaks[1]:
			vector = np.angle(snapshot)
			db_of_vector = 20 * scipy.log10(2.0 / CHUNK * np.abs(snapshot))
			if statistics.mean(db_of_vector) < 30:
				# print(statistics.mean(db_of_vector))
				continue
			normalized = vector[0]
			for i in range(len(vector)):
				vector[i] -= normalized
			# print(vector)
			complex_vector = nprect(1, vector)
			# assert (vector - np.angle(complex_vector) < 0.0001).all()
			final_angle += complex_vector
			counter += 1
			# print(complex_vector)
		if counter < THRESHOLD_FOR_MODE:
			return None
		# print(counter)
		final_angle /= counter

		# print(final_angle)
		# print("avrage: ", final_angle)
		results = []
		for phi in s_phi:
			results.append(np.vdot(phi, final_angle))
		final_angle = np.argmax(np.abs(results))
		# db = statistics.mean(gmean(peaks[2]))
		to_return.append((peaks[0], final_angle*ANGLE_OF_DIRECTIONS, 0))

	# print(to_return)
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
	nprect = np.vectorize(rect)
	angle_from_math = potential_phi(frequency_for_draw)
	mean_complex = rect(0,1)
	len_of_vector_of_snapshots = int(len(all_fft[0]))
	x = np.linspace(0.0, 360, NUM_OF_DIRECTIONS)
	s_phi = nprect(1, angle_from_math)
	title = "inner product for 90 angle 600HZ"
	location_to_save = "./graphs/"
	for vector_of_snapshots in all_fft:
		# print(vector_of_snapshots)
		for snapshot in vector_of_snapshots:
			angle_tag = np.angle(snapshot)
			angle_tag_norm = angle_tag[0]
			angle_tag_normalized = (angle_tag - angle_tag_norm) % MOD_2_PI
			complex_from_mic = nprect(1, angle_tag_normalized)
			mean_complex += complex_from_mic
			results = []
			for phi in s_phi:
				results.append(np.vdot(phi, complex_from_mic))
			plt.plot(x, np.abs(results), label="phase normalized")
			# print(len(s_phi), len(s_phi[0]), len(complex_from_mic), len(results), results[0])
	plt.title(title)
	plt.savefig(location_to_save+"all " + title + ".png", format=FORMAT_TO_SAVE, dpi = 720)
	plt.close()

	mean_complex /= (len(all_fft) * len_of_vector_of_snapshots)
	# print(np.abs(s_phi))
	results = []
	for phi in s_phi:
		results.append(np.vdot(phi, mean_complex))
	# print(np.abs(results))
	plt.plot(x, np.abs(results))
	plt.title(title)
	# plt.savefig(location_to_save+title + ".png", format=FORMAT_TO_SAVE, dpi=720)
	# plt.close()
	plt.show()
