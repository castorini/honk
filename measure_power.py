import os
import threading
import time

import serial

class PowerAccumulator(object):
    def __init__(self, idle_watts=0):
        self.reset()
        self.idle_watts = idle_watts

    def reset(self):
        self.joules = 0.
        self.total_time = 0.
        self.peak_power = 0.
        self.last_time = None
        self.last_watts = None

    @property
    def mean_power(self):
        if self.total_time == 0:
            return self.last_watts
        return self.joules / self.total_time

    def __call__(self, watts):
        if self.last_time is None:
            self.last_time = time.time()
            self.last_watts = watts - self.idle_watts
            return
        curr_time = time.time()
        time_delta = curr_time - self.last_time
        self.total_time += time_delta

        self.joules += time_delta * ((watts + self.last_watts - self.idle_watts) / 2)
        self.last_watts = watts - self.idle_watts
        self.last_time = curr_time
        self.peak_power = max(self.peak_power, watts)
        return self.joules

class PowerMeter(object):
    def __init__(self, dev="ttyUSB"):
        self.refresh_device(dev)
        self.io_thread = None

    def refresh_device(self, dev):
        directory = "/dev"
        device = None
        for filename in os.listdir(directory):
            if dev in filename:
                device = os.path.join(directory, filename)
                break
        if device is None:
            raise ValueError("Couldn't find the serial device!")
        print("Found {}".format(device))
        self.device = serial.Serial(device, 115200)

    def flush_io(self):
        self.device.reset_input_buffer()
        self.device.reset_output_buffer()

    def start_logging(self, callback):
        def power_loop():
            while True:
                line = self.device.readline().decode()
                try:
                    watts = float(line.split(",")[3]) / 10
                except:
                    continue
                if watts is None:
                    continue
                callback(watts)
        self.device.write(b"#L,W,3,E,,1;")
        time.sleep(0.1)
        self.io_thread = threading.Thread(target=power_loop)
        self.io_thread.start()

def main():
    def power_cb(power):
        joules = acc(power)
        print(acc.joules)

    acc = PowerAccumulator(idle_watts=2.338)
    meter = PowerMeter()
    meter.start_logging(power_cb)
    meter.io_thread.join()

if __name__ == "__main__":
    main()
