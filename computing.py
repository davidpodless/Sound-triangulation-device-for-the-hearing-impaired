import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import signal
from scipy.stats.mstats import gmean
import time
from systemConstants import *
from cmath import rect
import statistics

all_fft = []
frequency_for_draw = 0
FORMAT_TO_SAVE = 'png'

# TODO - check value for 2 signals, should work, verify that.
# TODO 2 - dealing with complex signals (different frequencies in the same NUM_OF_SNAPSHOT)


def extract_data(frames, results):
	"""
	:param frames: deque of the raw data from the mics
	:param results: deque where the thread will save the data
	:return: None
	"""
	is_still_empty = False
	thread_counter = 0
	while True:
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
					ch_data[i-1] = (np_data[i::6])
				list_of_data_sent_to_calc.append(ch_data)
			results.appendleft(calc_angle(list_of_data_sent_to_calc, thread_counter))
			thread_counter += 1
		else:
			if is_still_empty:
				break
			else:
				is_still_empty = True
				time.sleep(0.005)


def calc_angle(lst_of_data, counter):
	"""
	:param lst_of_data: list of NUM_OF_SNAPSHOTS_FOR_MUSIC arrays, for each
	array: n=4, in each cell the signal from the i-th mic
	:param counter: for testing, counting how much into the signal the
	compute will go
	:return: the frequency and the angles of the signal. in case where there
	is more than one frequency - for each one
	"""
	peaks_in_data = []
	mode_of_frequencies = {}
	n = CHUNK
	t = 1.0 / SAMPLE_RATE
	xf = np.linspace(0.0, 1.0 / (2.0 * t), n / 2)
	# find the frequencies of a signal
	for snapshot in lst_of_data:
		# results contains: frequency, location in snapshot, mean of db.
		results = find_peaks(snapshot[0])

		for index in results[1]:
			if xf[index] >= 100:  # ignore low frequencies because of nosies
				if index not in mode_of_frequencies:
					# count how many time specific frequency is in the data
					mode_of_frequencies[index] = 1
				else:
					mode_of_frequencies[index] += 1
		peaks_in_data.append(results)

	location_of_real_peaks_in_data = []
	for index in mode_of_frequencies:
		if mode_of_frequencies[index] >= THRESHOLD_FOR_MODE:
			location_of_real_peaks_in_data.append(index)
	fft_signal = scipy.fftpack.fft(lst_of_data)
	fft_signal = fft_signal[:, :, :n // 2]
	separated_vector_for_music = []
	for i in location_of_real_peaks_in_data:
		# each frequency in a special vector
		separated_vector_for_music.append(fft_signal[:, :, i])
	to_return = []
	global all_fft
	global frequency_for_draw
	for index, fft_vector in enumerate(separated_vector_for_music):
		db = 20 * scipy.log10(2.0 / n * np.abs(fft_vector))
		angles = MUSIC_algorithm(fft_vector, xf[
										location_of_real_peaks_in_data[
											index]], counter)
		# print(xf[location_of_real_peaks_in_data[index]], db.mean(), angles)
		# print(not angles)
		if angles:
			to_return.append([xf[location_of_real_peaks_in_data[index]], db.mean(), angles])

	return to_return


def find_peaks(raw_signal):
	"""
	:param raw_signal: raw signal from the mics
	:param avr: the db average of the signal for the last RECORD_SECONDS seconds
	:return: array [list of the freq peaks in the signal, the location in the 
	array of it, the fft of those locations, the average of the db of the signal]
	"""
	n = CHUNK
	t = 1.0 / SAMPLE_RATE  # sample spacing
	xf = np.linspace(0.0, 1.0 / (2.0 * t), n / 2)
	yf = scipy.fftpack.fft(raw_signal)
	abs_of_yf = np.abs(yf[:n // 2])

	magnitude_of_frequency = 2.0 / n * abs_of_yf
	db_of_yf = 20 * scipy.log10(magnitude_of_frequency)
	result = signal.find_peaks(db_of_yf, 30)
	real_db = result[1]['peak_heights']
	if real_db.size == 0:
		real_db = np.append(real_db, [0])
	return [xf[result[0]], result[0], real_db.mean()]


def potential_phi(freq):
	"""
	:param freq: frequency to check
	:return: array, n=360, each cell represent the value that should be in 
	the Vector if the signal come from that angle
	"""
	lst_to_return = []
	for i in range(NUM_OF_DIRECTIONS):
		results = []
		rads = math.radians(i*ANGLE_OF_DIRECTIONS)
		delta_x = [0, D * math.cos(rads), math.sqrt(2) * D * math.cos((PI/4) - rads), D * math.sin(rads)]
		r = 1
		phase = 2*PI*freq / SPEED_OF_SOUND
		for dx in delta_x:
			results.append(complex(r*math.cos(dx * phase), r*math.sin(dx * phase)))
		lst_to_return.append(results)

	return lst_to_return


def matrix_from_vector(vector):
	"""
	:param vector: vector to create covariance matrix from
	:return: the covariance matrix of the :param vector
	"""
	dimension = len(vector)
	# vector = vector.reshape((dimension, 1))
	matrix_to_return = np.zeros([dimension, dimension], dtype=np.complex64)
	for i in range(dimension):
		for j in range(dimension):
			matrix_to_return[i][j] = vector[i] * vector[j].conj()
	return matrix_to_return


def sum_of_matrix(m1, m2):
	"""
	:param m1: a matrix
	:param m2: a matrix
	:return: the sum of the two matrices
	"""
	if m1.shape != m2.shape:
		raise ValueError("matrices not in the same shape")
	matrix_to_return = np.zeros(m1.shape, dtype=complex)
	for i in range(np.size(m1, 0)):
		for j in range(np.size(m1, 1)):
			matrix_to_return[i, j] = (m1[i, j] + m2[i, j])
	return matrix_to_return


def MUSIC_algorithm(vector_of_signals, freq, counter):
	"""
	:param vector_of_signals: vector of NUM_OF_SNAPSHOTS_FOR_MUSIC snapshot,
	each snapshot contain signal in frequency freq
	:param freq: the frequency of the signal
	:param counter: for debug purpose
	:return: the angles where the signals came from
	"""

	""" In this function, N - number of mics, M number of signals"""
	nprect = np.vectorize(rect)

	s_phi = potential_phi(freq)
	R = np.zeros([NUM_OF_MICS,NUM_OF_MICS], dtype=np.complex64)

	assert len(vector_of_signals) == NUM_OF_SNAPSHOTS_FOR_MUSIC
	# MLE:
	theta = [] # mean of angle from the samples
	angle = (np.angle(vector_of_signals) % MOD_2_PI)
	for snapshot in angle:
		norm = snapshot[0]
		for i, mic in enumerate(snapshot):
			snapshot[i] -= norm
	for i, mic in enumerate(angle.T):
		xcos = []
		ysin = []
		for point in mic:
			xcos.append(math.cos(point))
			ysin.append(math.sin(point))
		x = np.mean(xcos)
		y = np.mean(ysin)
		theta.append(math.atan2(y, x))
	MLE_complex = nprect(1, theta)
	results = []
	for phi in s_phi:
		results.append(np.vdot(phi, MLE_complex))
	MLE = np.argmax(np.abs(results)) * ANGLE_OF_DIRECTIONS
	# END MLE
	# draw_graph(np.abs(results), "MLE", freq, counter)
	# MUSIC algorithm
	for vector in vector_of_signals:
		vector = nprect(1, np.angle(vector))
		R = sum_of_matrix(R, matrix_from_vector(vector))
	R /= NUM_OF_SNAPSHOTS_FOR_MUSIC
	"""now, R is N*N matrix with rank M. meaning, there is N-M eigenvectors
	corresponding to the zero eigenvalue"""
	eigenvalues, eigenvectors = np.linalg.eig(R)
	# eigenvectors in rows
	eigenvectors = eigenvectors.T
	# sort the eigenvalues and eigenvectors from the smallest to the largest
	idx = eigenvalues.argsort()
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[idx]

	'''# MSE for 1 signal case
	tester = eigenvectors[-1]
	results = []
	for phi in s_phi:
		results.append(np.vdot(phi, tester))
	mse_final_angle_for_one_signal = (signal.argrelmax(np.abs(results), mode='warp'))[0] * ANGLE_OF_DIRECTIONS'''

	# determine how many signals, according to eigenvalues
	# large eigenvalue mean signal, the noise should be the eigenvalue 0.
	M = 0
	# print(np.abs(eigenvalues))
	for i in eigenvalues:
		# TODO - choose threshold for the eigenvalues, use records for that.
		if np.abs(i) > THRESHOLD_FOR_EIGENVALUES:
			M += 1

	if M >= 4:
		return "Too much signals to process"

	P_MUSIC_phi = []
	for index, angle in enumerate(s_phi):
		result = 0
		for i in range(len(eigenvalues) - M):
			result += np.square(np.abs(np.vdot(eigenvectors[i].T, angle)))
		P_MUSIC_phi.append(1 / result)
	# draw_graph(P_MUSIC_phi, "MUSIC", freq, counter)
	final_angle = (signal.argrelmax(np.asarray(P_MUSIC_phi), mode='warp')[0])

	MUSIC_results = []
	for j in range(M):
		max = -10
		for i in final_angle:
			if i in MUSIC_results:
				continue
			else:
				if P_MUSIC_phi[i] > THRESHOLD_FOR_MUSIC_PEAK and P_MUSIC_phi[i] > max:
					max = i
		if max > -10:
			MUSIC_results.append(max)
	for i in range(len(MUSIC_results)):
		MUSIC_results[i] *= ANGLE_OF_DIRECTIONS

	if len(MUSIC_results) == 1:
		MUSIC_results = MUSIC_results[0]

	to_return = [MUSIC_results, MLE]
	return to_return

'''
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
	# angle_from_math = potential_phi(frequency_for_draw)
	mean_complex = rect(0,1)
	len_of_vector_of_snapshots = int(len(all_fft[0]))
	x = np.linspace(0.0, 360, NUM_OF_DIRECTIONS)
	# s_phi = nprect(1, angle_from_math)
	s_phi = potential_phi(frequency_for_draw)
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
	plt.show()'''

def draw_graph(results, type, freq, counter):
	'''
	:param results: array of real numbers
	:param type: how the data was estimated
	:param freq: the frequency of the signal
	:param counter:
	:return:
	'''
	title = type + " no. " + str(counter) + " for " + str(np.argmax(results) * ANGLE_OF_DIRECTIONS) + " degree and " + str(int(freq)) + "HZ"
	# print(title)
	# exit()
	X = ANGLE_OF_DIRECTIONS * np.arange(0, NUM_OF_DIRECTIONS, 1)
	location_to_save = "./graphs/"
	plt.plot(X, results)
	plt.xlabel("angle")
	plt.title(title)
	plt.savefig(location_to_save+title + ".png", format=FORMAT_TO_SAVE, dpi=720)
	plt.close()
