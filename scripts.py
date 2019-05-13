from systemConstants import *
import numpy as np
import matplotlib.pyplot as plt
import computing
import scipy.fftpack



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
		delta_x = [0, D * math.cos(rads), math.sqrt(
			2) * D * math.cos((PI/4) - rads), D * math.sin(rads)]
		# phase approach:
		phase_change = (2*PI*freq / SPEED_OF_SOUND)
		r = 1
		for dx in delta_x:
			results.append(complex(r*math.cos(dx * phase_change), r*math.sin(dx * phase_change)))
			# r -= 0.2

		lst_to_return.append(results)
	return lst_to_return


def run():
	results = potential_phi(350)
	mic1 = []
	mic2 = []
	mic3 = []
	for result in results:
		mic1.append(result[1])
		mic2.append(result[2])
		mic3.append(result[3])
	x = ANGLE_OF_DIRECTIONS * np.arange(0, NUM_OF_DIRECTIONS, 1)
	print("0: ", np.angle(results[0]))
	print("45: ", results[45])
	# print("90: ", results[90])
	# print("180: ", results[180])
	# print("270: ", results[270])
	# print("315: ", results[315])
	plt.plot(x, np.angle(mic1), label="mic1")
	plt.plot(x, np.angle(mic2), label="mic2")
	plt.plot(x, np.angle(mic3), label="mic3")
	plt.legend()
	title = "Expected result because math"
	plt.title(title)
	plt.show()
	# plt.savefig("angle graph.png", format="PNG", dpi=720)


def check_outer_product():
	a = np.zeros([4,4], dtype=complex)
	x = np.zeros(shape=(4,1), dtype=complex)
	x[0,0] = complex(0,1)
	x[1,0] = complex(1/2,math.sqrt(3)/2)
	x[2,0] = complex(1,0)
	x[3,0] = complex(math.sqrt(2)/2,math.sqrt(2)/2)
	print(x)
	print(x.conj())
	print(np.outer(x, x))
	print(np.outer(x, x.conj()))
	print(x @ x.conj().T)
	print(np.matmul(x, x.conj().T))
	print(computing.matrix_from_vector(x))
	# a +=

def check_sum_of_matrix():
	R = np.zeros([2,2], dtype=np.complex64)
	a = np.matrix([[complex(1,2),2],[3,4]])
	b = np.matrix([[1, 2], [3, 4]])
	print(R)
	R = computing.sum_of_matrix(a,b)
	print(R)
	R = a+b
	print(R)

def check_mean_of_angle():
	sigma = []
	angle = np.asarray([[ 0.,-6.01300704, -5.88468294, -6.20433656],
	                    [ 0.,          0.26343833,  0.25972342,  0.15630209],
 [ 0.,          0.37666975,  0.36675583, -0.04633747],
 [ 0.,          0.19767262,  0.28710173, -0.03393963],
 [ 0.,          0.19716212,  0.15120855, -0.0796047 ]])
	for snapshot in angle:
		norm = snapshot[0]
		for i, mic in enumerate(snapshot):
			snapshot[i] -= norm

	for mic in angle.T:
		x = np.mean(np.cos(mic))
		y = np.mean(np.sin(mic))
		sigma.append(math.atan2(y, x))
	print(angle, "\n", sigma, "\n\n\n\n")


def plot_sin():

	X = 1/20 * np.arange(-NUM_OF_DIRECTIONS, NUM_OF_DIRECTIONS, 1)
	sin_for_x = np.sin(np.radians(X))
	plt.plot(X, sin_for_x)
	plt.xlabel("degree")
	plt.ylabel("sin")
	plt.grid(visible = True)
	plt.savefig("./graphs/sins.png", dpi=720)


def fft_test():
	a = [1, 2, 3, 4]
	fft_a = scipy.fftpack.fft(a)
	fft_fft_a = scipy.fftpack.fft(fft_a)
	print(a, '\n', fft_a, '\n', fft_fft_a)


if __name__=="__main__":
	fft_test()