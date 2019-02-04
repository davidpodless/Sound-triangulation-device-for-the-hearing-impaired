#!/usr/bin/python
# base code created by russellcardullo
# amplify.py: Amplifies a WAV file by the given scaling factor
#             and output to specified file

import sys
from audiofile import *

if len(sys.argv) < 4:
	print
	"Usage: amplify.py in_file out_file scale_factor"
	sys.exit(1)

in_filename = sys.argv[1]
out_filename = sys.argv[2]
scale_factor = float(sys.argv[3])

fin = AudioFile(in_filename, 'r')
fout = AudioFile(out_filename, 'w')
fout.setparams(fin.getnchannels(),
			   fin.getsampwidth(),
			   fin.getframerate(),
			   0,
			   'NONE',
			   'not compressed')

num_frames = fin.getnframes()
num_channels = fin.getnchannels()


def transform(data, scale_factor):
	sample = int(data * scale_factor)
	return sample


while (fin.tell() < num_frames):
	frame_data = fin.read(1)
	for data in frame_data:
		out_frame = transform(data, scale_factor)
		fout.write(out_frame)

fout.close()
fin.close()