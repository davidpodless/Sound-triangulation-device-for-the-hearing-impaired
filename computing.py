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
average_DB = 0
frequency_for_draw = 0
FORMAT_TO_SAVE = 'png'

# TODO - fixing the DB


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
				# TODO - is the dtype correct? changing to 32 break everything
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
	global average_DB
	peaks_in_data = []
	mode_of_frequencies = {}
	n = CHUNK
	t = 1.0 / SAMPLE_RATE
	xf = np.linspace(0.0, 1.0 / (2.0 * t), n / 2)
	# find the frequencies of a signal
	for snapshot in lst_of_data:
		# results contains: frequency, location in snapshot, mean of db.
		results = find_peaks(snapshot[0], average_DB)

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
		to_return.append(xf[location_of_real_peaks_in_data[index]])
		db = 20 * scipy.log10(2.0 / n * np.abs(fft_vector))
		to_return.append(MUSIC_algorithm(fft_vector, xf[
										location_of_real_peaks_in_data[
											index]], db, counter))
		# to_return.append(one_signal_algorithm(
		# 	(xf[location_of_real_peaks_in_data[index]], np.angle(fft_vector), db)))
	return to_return


def find_peaks(raw_signal, avr):
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
	result = signal.find_peaks(db_of_yf, height=max(30, avr))
	# TODO - should I return the db of the peaks? for deciding which freq to choose? no idea
	real_db = result[1]['peak_heights']
	if real_db.size == 0:
		real_db = np.append(real_db, [0])

	return [xf[result[0]], result[0], real_db.mean()]


def phase_change(freq): return 2*PI*freq / SPEED_OF_SOUND


def potential_phi(freq):
	"""
	:param freq: frequency to check
	:return: array, n=360, each cell represent the value that should be in 
	the Vector if the signal come from that angle
	"""
	lst_to_return = []
	find_max = []
	for i in range(NUM_OF_DIRECTIONS):
		results = []
		dummy = []
		rads = math.radians(i*ANGLE_OF_DIRECTIONS)
		delta_x = [0, D * math.cos(rads), math.sqrt(2) * D * math.cos((PI/4) - rads), D * math.sin(rads)]
		# phase approach:
		r = 1
		phase = phase_change(freq)
		for dx in delta_x:
			results.append(complex(r*math.cos(dx * phase), r*math.sin(dx * phase)))
			dummy.append(dx*phase)
		find_max.append(dummy)
		lst_to_return.append(results)
	# print(freq)
	max_for_mic = []
	for mic in np.asarray(find_max).T:
		mic_max = np.max(mic)
		max_for_mic.append(mic_max + ERROR_THRESHOLD*mic_max)
	return lst_to_return, max_for_mic


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


def MUSIC_algorithm(vector_of_signals, freq, db_of_signal, counter):
	"""
	:param vector_of_signals: vector of NUM_OF_SNAPSHOTS_FOR_MUSIC snapshot,
	each snapshot contain signal in frequency freq
	:param freq: the frequency of the signal
	:param counter: for debug purpose
	:return: the angles where the signals came from
	"""

	""" In this function, N - number of mics, M number of signals"""
	# if freq < 250:
	# 	return None
	nprect = np.vectorize(rect)
	x = ANGLE_OF_DIRECTIONS * np.arange(0, NUM_OF_DIRECTIONS, 1)
	s_phi, max_for_mics = potential_phi(freq)
	R = np.zeros([NUM_OF_MICS,NUM_OF_MICS], dtype=np.complex64)

	assert len(vector_of_signals) == NUM_OF_SNAPSHOTS_FOR_MUSIC
	# MLE:
	sigma = []
	angle = (np.angle(vector_of_signals) % MOD_2_PI)
	for snapshot in angle:
		norm = snapshot[0]
		for i,mic in enumerate(snapshot):
			snapshot[i] -= norm
	print(angle)
	for i, mic in enumerate(angle.T):
		xcos = []
		ysin = []
		for point in mic:
			# if np.abs(point)> max_for_mics[i] and (MOD_2_PI - np.abs(point) > max_for_mics[i]):
			# 	print(i, point, " this point was deleted")
			# 	continue
			xcos.append(math.cos(point))
			ysin.append(math.sin(point))
		x = np.mean(xcos)
		y = np.mean(ysin)
		sigma.append(math.atan2(y, x))
	# print(angle, "\n", sigma, "\n\n\n\n")
	# print(sigma)
	MLE_complex = nprect(1, sigma)
	results = []
	for phi in s_phi:
		results.append(np.vdot(phi, MLE_complex))
	MLE = np.argmax(np.abs(results)) * ANGLE_OF_DIRECTIONS
	print("MLE: ", MLE)
	# END MLE
	skipped = 0
	for vector in vector_of_signals:  # TODO: is normalized was the problem?!
		normalized = vector[0]

		# print(vector)
		# print(len(vector))
		for i in range(len(vector)):
			# print(np.abs(vector[i]) / np.abs(normalized))
			angle = np.angle(vector[i])
			# r = np.abs(vector[i])
			r = 1
			# angle = (np.angle(vector[i]) - np.angle(normalized)) % MOD_2_PI
			# if np.abs(angle) > max_for_mics[i] and (MOD_2_PI - np.abs(angle) > max_for_mics[i]):
			# 	print(i, (MOD_2_PI - np.abs(angle), np.angle(vector)), " this vector was deleted")
			# 	skipped += 1
			# 	break
			vector[i] = complex(r*math.cos(angle), r*math.sin(angle))
			# print(i, angle)
			# sigma[i] += angle
			# vector[i] = rect(1, np.angle(vector[i]) - np.angle(normalized))
			# vector[i] = rect(1, np.angle(vector[i]))

		# exit()
		# print(vector, vector.conj().T)
		R = sum_of_matrix(R, matrix_from_vector(vector))
	# for i, s in enumerate(sigma):
	# 	sigma[i] = s / (NUM_OF_SNAPSHOTS_FOR_MUSIC - counter)
	# print("result: ", sigma, "\n\n")
	# exit()
	# R = np.add(np.outer(vector, vector.conj()), R)
	# print(np.angle(vector_of_signals))
	# exit(321)
	# print(R)
	if skipped > THRESHOLD_FOR_MODE:
		return "unable to detect"
	R /= (NUM_OF_SNAPSHOTS_FOR_MUSIC - skipped)
	# print(np.abs(R), np.angle(R))
	# exit(12)
	# print(R,"\n\n",np.abs(R),"\n\n" ,np.angle(R), "\n\n\n")
	# exit(12)
	"""now, R is N*N matrix with rank M. meaning, there is N-M eigenvectors
	corresponding to the zero eigenvalue"""
	eigenvalues, eigenvectors = np.linalg.eig(R)
	# sort the eigenvalues and eigenvectors from the smallest to the largest
	idx = eigenvalues.argsort()
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[idx]
	tester = eigenvectors[-1]
	results = []
	for phi in s_phi:
		results.append(np.vdot(phi, tester))
	mse_final_angle_for_one_signal = np.argmax(np.abs(results)) * ANGLE_OF_DIRECTIONS

	DB_of_eigenvalues = 20 * scipy.log10(2 / CHUNK * np.abs(eigenvalues))
	M = 0
	# for i in DB_of_eigenvalues:
	# 	print(i)
	# exit(12)
	for i in eigenvalues:
		# TODO - choose threshold for the eigenvalues, use records for that.
		if np.abs(i) > 0.1:
			M += 1
			# print(i)
		# else:
		# 	print(i)
	# if M == 4:
	# 	print(np.abs(eigenvalues))
		# raise Exception
	# exit(1)

	M = 1
	P_MUSIC_phi = []
	j = 0
	super_result = 0
	for index, angle in enumerate(s_phi):
		result = 0
		for i in range(len(eigenvalues) - M):
			result += np.square(np.abs(np.vdot(eigenvectors[i].T, angle)))
		super_result += result
		j += 1
		P_MUSIC_phi.append(1 / result)
	# print(signal.find_peaks(P_MUSIC_phi), ANGLE_OF_DIRECTIONS  )
	# plt.plot(x, P_MUSIC_phi)
	# plt.show()
	# TODO - return the M maxes, not only 1
	# final_angle = np.argmax(P_MUSIC_phi) * ANGLE_OF_DIRECTIONS
	final_angle = (signal.find_peaks(P_MUSIC_phi)[0]) * ANGLE_OF_DIRECTIONS
	# print(final_angle)
	# exit()
	# return freq, final_angle, statistics.mean(gmean(db_of_signal))
	print("MUSIC: " + str(final_angle), "MLE: ", MLE, " MSE from MUSIC: ", mse_final_angle_for_one_signal, "\n\n\n\n")
	return "MUSIC: " + str(final_angle) + " MSE: " + str(mse_final_angle_for_one_signal) + " MLE: " + str(MLE)
	# return "MLE: " + str(MLE)


def one_signal_algorithm(peaks):
	"""
	:param peaks: list of tuples (freq, angle, db of the signal) that represent peaks in frequency
	:return: the direction which the signal come from in a tuple (freq, direction, db of the signal)
	this is the naive and not necessarily work approach.
	"""

	to_return = []
	nprect = np.vectorize(rect)
	s_phi, max_for_mics = potential_phi(peaks[0])
	if peaks:
		if peaks[0] < 100:
			return
		final_angle = rect(0, 1)
		counter = 0
		for snapshot in peaks[1]:
			vector = np.angle(snapshot)
			db_of_vector = 20 * scipy.log10(2.0 / CHUNK * np.abs(snapshot))
			# if statistics.mean(db_of_vector) < 30:
			# 	print(statistics.mean(db_of_vector))
			# 	continue
			# print(statistics.mean(db_of_vector))
			normalized = vector[0]
			for i in range(len(vector)):
				vector[i] -= normalized
				if np.abs(vector[i]) > max_for_mics[i] and (MOD_2_PI - np.abs(vector[i]) > max_for_mics[i]):
					print(i, (MOD_2_PI - np.abs(vector[i]), np.angle(vector)), " this vector was deleted")
					# skipped += 1
					break
			# print(vector)
			complex_vector = nprect(1, vector)
			# assert (vector - np.angle(complex_vector) < 0.0001).all()
			final_angle += complex_vector
			counter += 1
		if counter < THRESHOLD_FOR_MODE:
			return None
		final_angle /= counter
		results = []
		for phi in s_phi:
			results.append(np.vdot(phi, final_angle))
		final_angle = np.argmax(np.abs(results))
		to_return.append(final_angle * ANGLE_OF_DIRECTIONS)

	# print(to_return)
	return "MLE: " + str(to_return)


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
	plt.show()
