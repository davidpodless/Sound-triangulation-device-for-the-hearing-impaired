from systemConstants import *
import numpy as np


def run():
	N = CHUNK
	T = 1.0 / SAMPLE_RATE  # sample spacing
	x = np.linspace(0.0, N * T, N)

	xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
	# yf = scipy.fftpack.fft(raw_signal)

	print(xf)


if __name__=="__main__":
	run()