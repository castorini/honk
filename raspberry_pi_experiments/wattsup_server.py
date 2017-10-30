#!/usr/bin/env python

import os, serial
import json

from flask import Flask
from threading import Thread

EXTERNAL_MODE = 'E'
INTERNAL_MODE = 'I'
TCPIP_MODE = 'T'
FULLHANDLING = 2

app = Flask(__name__)

class WattsUp(object):
    def __init__(self, port, interval=1.0):
        self.serial = serial.Serial(port, 115200)
        self.power_consumption = 0
        self.peak_watt = -1
        self.last_read = 0
        self.interval = interval
        self.running = True

    def start(self, verbose=False):
        if verbose:
            print "Logging..."

        line = self.serial.readline()
        n = 0

        # Reply with ready msg
        while self.running:
            if line.startswith('#d'):
                fields = line.split(',')
                fields = [d if d != '_' else 0 for d in fields]
                if len(fields)>5:
                    # the watt read
                    # print fields[3]
                    W = float(fields[3]) / 10

                    self.power_consumption += W
                    self.peak_watt = max(self.peak_watt, W)
                    self.last_read = W
                    n += self.interval
            line = self.serial.readline()

    def reset(self):
        self.power_consumption = 0
        self.peak_watt = -1

    def get_read(self):
        return json.dumps({"consumption": self.power_consumption, "peak": self.peak_watt})

    def get_last_read(self):
        return str(self.last_read)

    def stop(self):
        self.running = False

def start_wattsup_process(meter):
    meter.start(verbose=False)

@app.route('/get_read')
def app_get_read():
    return meter.get_read()

@app.route('/get_last_read')
def app_get_last_read():
    return meter.get_last_read()

@app.route('/reset_read')
def app_reset_read():
    meter.reset()
    return "OK"

if __name__ == '__main__':
    meter = WattsUp("/dev/tty.usbserial-A60144B9")
    # meter.log(verbose=False)
    thread = Thread(target=start_wattsup_process, args=(meter,))
    thread.start()
    app.run("0.0.0.0", port=5000)
    meter.stop()

