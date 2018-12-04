from usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time
import pyaudio
import wave
import numpy as np
# from scipy.io import wavfile
import binascii

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6  # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 2  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "output.wav"

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)


def read_audio():
    pass
    # wave.open()
    # fs, data = wavfile.read('output.wav')
    # return fs, data


def record():
    p = pyaudio.PyAudio()

    stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX, )

    print("* recording")

    frames = []

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

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


if __name__ == '__main__':
    record()