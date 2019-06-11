import numpy as np
import scipy.fftpack
from scipy import signal
from systemConstants import *
from cmath import rect


def extract_data(frames, results):
    """
    :param frames: deque of the raw data from the mics
    :param results: deque where the thread will save the data
    :return: None
    """
    is_still_empty = False
    thread_counter = 0
    avg_db = 30
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
            angle, db = calc_angle(list_of_data_sent_to_calc, avg_db, thread_counter)
            #print(angle)
            # avg_db -= (avg_db / LEN_OF_AVG)
            # avg_db += (db / LEN_OF_AVG)
            results.appendleft(angle)
            thread_counter += 1

            #     continue
        else:
            continue


def calc_angle(lst_of_data, avg_db, counter):
    """
    :param lst_of_data: list of NUM_OF_SNAPSHOTS_FOR_MUSIC arrays, for each
    array: n=4, in each cell the signal from the i-th mic
    :param avg_db: the average of the db in the area in the last RECORD_SECONDS seconds
    :param counter: for testing, counting how much into the signal the
    compute will go
    :return: the frequency and the angles of the signal. in case where there
    is more than one frequency - for each one
    """
    n = CHUNK
    t = 1.0 / SAMPLE_RATE
    xf = np.linspace(0.0, 1.0 / (2.0 * t), n / 2)

    # find the frequencies of a signal
    '''for snapshot in lst_of_data:
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
            location_of_real_peaks_in_data.append(index)'''
    fft_signal = scipy.fftpack.fft(lst_of_data)
    abs_of_yf = np.abs(fft_signal[:, :, :n // 2])

    magnitude_of_frequency = 2.0 / n * abs_of_yf
    db_of_yf = 20 * scipy.log10(magnitude_of_frequency)

    dB_for_find_peaks = np.sum(np.square(np.abs(db_of_yf)), axis=0)[0]

    peaks = find_peaks(dB_for_find_peaks, avg_db, counter)

    location_of_real_peaks_in_data = peaks[1]

    fft_signal = fft_signal[:, :, :n // 2]
    separated_vector_for_music = []
    for i in location_of_real_peaks_in_data:
        # each frequency in a special vector
        separated_vector_for_music.append(fft_signal[:, :, i])
    results = []

    for index, fft_vector in enumerate(separated_vector_for_music):
        freq = xf[location_of_real_peaks_in_data[index]]
        if not is_correct_frequency(freq):
            continue
        result = MUSIC_algorithm(fft_vector, freq, counter)
        results.append(result)

    final_vector = sum_vectors(np.asarray(results))

    try:
        indexes = (signal.argrelmax(np.asarray(final_vector), mode='warp')[0])
        angles = []
        for index in indexes:
            angles.append((index * ANGLE_OF_DIRECTIONS, final_vector[index]))
        #print(angles)
        return angles, peaks[2]
    except IndexError:
        #print([])
        return [], peaks[2]


def find_peaks(raw_signal, avg, counter):
    """
    :param raw_signal: raw signal from the mics
    :param avr: the db average of the signal for the last RECORD_SECONDS seconds
    :return: array [list of the freq peaks in the signal, the location in the 
    array of it, the fft of those locations, the average of the db of the signal]
    """
    n = CHUNK
    t = 1.0 / SAMPLE_RATE  # sample spacing
    xf = np.linspace(0.0, 1.0 / (2.0 * t), n / 2)
    yf = raw_signal
    abs_of_yf = np.abs(yf[:n // 2])
    magnitude_of_frequency = 2.0 / n * abs_of_yf
    db_of_yf = 20 * scipy.log10(magnitude_of_frequency)
    result = signal.find_peaks(db_of_yf, max(THRESHOLD_FOR_FIND_PEAKS,avg))

    real_db = result[1]['peak_heights']
    # print(result)
    if real_db.size == 0:
        real_db = np.append(real_db, [0])
    return [xf[result[0]], result[0], real_db.mean()]


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
    print("freq: ", freq)

    s_phi = potential_phi(freq)

    # MLE:
    sigma = []
    angle = (np.angle(vector_of_signals) % MOD_2_PI)
    for snapshot in angle:
        norm = snapshot[0]
        for i, mic in enumerate(snapshot):
            snapshot[i] -= norm
    # print(angle)
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
    phase = 2 * PI * freq / SPEED_OF_SOUND
    print((sigma[3] / phase) / (sigma[1] / phase))
    # print(sigma)
    MLE_complex = nprect(1, sigma)
    results = []
    for phi in s_phi:
        results.append(np.vdot(phi, MLE_complex))
    MLE = np.argmax(np.abs(results)) * ANGLE_OF_DIRECTIONS
    # print("MLE: ", MLE)
    # END MLE
    R = np.zeros([NUM_OF_MICS,NUM_OF_MICS], dtype=np.complex64)

    assert len(vector_of_signals) == NUM_OF_SNAPSHOTS_FOR_MUSIC

    # MUSIC algorithm
    for vector in vector_of_signals:
        angles = np.angle(vector)
        norm = angles[0]
        for i in range(len(angles)):
            angles[i] -= norm

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

    # determine how many signals, according to eigenvalues
    # large eigenvalue mean signal, the noise should be the eigenvalue 0.
    M = 1
    P_MUSIC_phi = []
    for index, angle in enumerate(s_phi):
        result = 0
        for i in range(len(eigenvalues) - M):
            result += np.square(np.abs(np.vdot(eigenvectors[i].T, angle)))
        P_MUSIC_phi.append(1 / result)

    return np.asarray(P_MUSIC_phi)


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
        delta_x = [0,
                   D * math.cos(rads),
                   math.sqrt(2) * D * math.cos((PI/4) - rads),
                   D * math.sin(rads)]
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


def average_db(vector): 
    return np.sum(np.square(np.abs(vector))) / len(vector)


def sum_vectors(vectors):
    return np.sum(vectors, axis=0)


def is_correct_frequency(frequency):
    return np.abs(frequency - FREQUENCY1) < FREQUENCY_BOUNDARY or \
           np.abs(frequency - FREQUENCY2) < FREQUENCY_BOUNDARY
