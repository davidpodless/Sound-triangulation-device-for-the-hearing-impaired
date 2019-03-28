from systemConstants import *
import numpy as np
import matplotlib.pyplot as plt



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


def run():
	results = potential_phi(600)
	mic1 = []
	mic2 = []
	mic3 = []
	for result in results:
		mic1.append(result[1])
		mic2.append(result[2])
		mic3.append(result[3])
	x = ANGLE_OF_DIRECTIONS * np.arange(0, NUM_OF_DIRECTIONS, 1)
	print("0: ", results[0])
	print("36: ", results[36])
	print("45: ", results[45])
	print("90: ", results[90])
	print("200: ", results[200])
	print("300: ", results[300])
	plt.plot(x, mic1, label="mic1")
	plt.plot(x, mic2, label="mic2")
	plt.plot(x, mic3, label="mic3")
	plt.legend()
	title = "Expected result because math"
	plt.title(title)
	plt.savefig("angle graph.png", format="PNG", dpi=720)


def check_outer_product():
	x = np.zeros(shape=(2,1), dtype=complex)
	x[0,0] = complex(0,1)
	x[1,0] = complex(1,2)
	print(x)
	print(x.conj())
	print(np.outer(x, x))
	print(np.outer(x.conj(), x))

def check_sum_of_matrix():
	R = np.zeros([2,2], dtype=np.complex64)
	a = np.matrix([[complex(1,2),2],[3,4]])
	b = np.matrix([[1, 2], [3, 4]])
	print(R)
	R = np.add(a,R)
	print(R)
	R += b
	print(R)


if __name__=="__main__":
	check_sum_of_matrix()