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
from pixel_ring.pixel_ring import pixel_ring


RESPEAKER_CHANNELS = 6  # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
RESPEAKER_INDEX = 2   # refer to input device id

# for i in range(0, numdevices):
#         if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#             print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
#             RESPEAKER_INDEX = i

WAVE_OUTPUT_FILENAME = "output.wav"

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)


def record(frames, foo):
    # print(type(frames))
    p = pyaudio.PyAudio()
    pixel_ring.off()

    stream = p.open(
        rate=SAMPLE_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX, )
    print("* listening")

    if not main.CHUNK_RECORDING:
        while True:
            
            data = stream.read(CHUNK, exception_on_overflow = False)
            frames.appendleft(data)
    else:
        counter = 0
        lst = []
        while main.keepRecording:
            data = stream.read(CHUNK, exception_on_overflow = False)
            lst.append(data)
            counter = (counter + 1) % NUM_OF_SNAPSHOTS_FOR_MUSIC
            if counter == 0:
                frames.appendleft(lst.copy())
                lst.clear()

    print("* done recording")
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
