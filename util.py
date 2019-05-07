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
	normalization = 1 / (2 * math.sqrt(sig * math.pi))
	# normalization = 1

	def gaussian(x):
		return normalization * math.exp((-((x - miu) ** 2) / (2 * (sig ** 2))))

	return gaussian


def build_gaussian_for_circle(deg, amp, width=GAUSSIAN_DEFAULT_SIG):
	deg = deg % 360
	return lambda x: build_gaussian(deg, width)(x % 360) * amp

def build_multiple_gausians(degs_and_amps, width=GAUSSIAN_DEFAULT_SIG):
	sum_function = lambda x: 1

	def add_func(func_1, func_2):
		return lambda x: func_1(x) + func_2(x)

	for deg, amp in degs_and_amps:
		sum_function = add_func(sum_function, build_gaussian_for_circle(deg, amp))

	return sum_function
