import numpy as np
import math

GAUSSIAN_DEFAULT_SIG = 4


def cart2pol(x, y):
	rho = np.sqrt(x ** 2 + y ** 2)
	phi = np.arctan2(y, x)
	return (rho, phi)


def pol2cart(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return (x, y)


def build_gaussian(miu, sig):
	# normalization = 1 / (2 * math.sqrt(sig * math.pi))
	normalization = 1

	def gaussian(x):
		return normalization * math.exp((-((x - miu) ** 2) / (2 * (sig ** 2))))

	return gaussian


def build_gaussian_for_circle(deg, amp, width=GAUSSIAN_DEFAULT_SIG):
	deg = deg % 360
	return lambda x: build_gaussian(deg, width)(x % 360) * amp

