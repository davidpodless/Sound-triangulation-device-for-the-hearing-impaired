from usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time
import pyaudio

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

p = pyaudio.PyAudio()
# info = p.get_host_api_info_by_index(0)
# numdevices = info.get('deviceCount')

if __name__ == '__main__':
    if dev:
        Mic_tuning = Tuning(dev)
        while True:
            try:
                print(Mic_tuning.direction)
                time.sleep(0.5)
            except KeyboardInterrupt:
                break