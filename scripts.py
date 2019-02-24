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
	# title = str(counter) +" " + str(freq)
	# plt.title(title)
	plt.show()


if __name__=="__main__":
	run()