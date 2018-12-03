from usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

if __name__ == '__main__':
    if dev:
        Mic_tuning = Tuning(dev)
        while True:
            try:
                print(Mic_tuning.direction)
                time.sleep(0.5)
            except KeyboardInterrupt:
                break