from usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time
import pyaudio
import wave
import numpy as np
# from scipy.io import wavfile
import binascii
import main
from systemConstants import *
from collections import deque

RESPEAKER_CHANNELS = 6  # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 2  # refer to input device id
WAVE_OUTPUT_FILENAME = "output.wav"

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)


def record(frames, foo):
    print(type(frames))
    p = pyaudio.PyAudio()

    stream = p.open(
        rate=SAMPLE_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX, )
    # print("* recording")
    # for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
    if not main.CHUNK_RECORDING:
        while True:
            
            data = stream.read(CHUNK, exception_on_overflow = False)
            frames.appendleft(data)
    else:
        counter = 0
        lst = []
        while True:
            data = stream.read(CHUNK, exception_on_overflow = False)
            lst.append(data)
            counter = (counter + 1) % NUM_OF_SNAPSHOTS_FOR_MUSIC
            if counter == 0:
                frames.appendleft(lst.copy())
                lst.clear()

    print("* done recording")
    print(len(frames))
    print(frames[0])
    print(type(frames[0]))

    first_data = frames[0]
    channel1 = first_data[2::6]
    channel2 = first_data[3::6]
    channel3 = first_data[4::6]
    channel4 = first_data[5::6]
    print(binascii.hexlify(bytearray(first_data)))
    print(binascii.hexlify(bytearray(channel1)))
    print(binascii.hexlify(bytearray(channel2)))
    print(binascii.hexlify(bytearray(channel3)))
    print(binascii.hexlify(bytearray(channel4)))

    stream.stop_stream()
    stream.close()
    p.terminate()
    #
    # b = np.frombuffer(b''.join(frames), dtype='<f4')
    # print(len(b[0]))


    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(RESPEAKER_CHANNELS)
    # wf.setsampwidth(
    #     p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    # wf.setframerate(RESPEAKER_RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()
