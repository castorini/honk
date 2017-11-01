#!/usr/bin/env python

import sys, socket
import os, serial
import json
import argparse

from flask import Flask
from threading import Thread

EXTERNAL_MODE = 'E'
INTERNAL_MODE = 'I'
TCPIP_MODE = 'T'
FULLHANDLING = 2

app = Flask(__name__)
meter = None

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
            print("Logging...")

        # setup the serial to get the watt read
        self.serial.write('#L,W,3,%s,,%d;' % (EXTERNAL_MODE, self.interval))
        line = self.serial.readline()
        n = 0

        # polling the watt reads
        while self.running:
            if line.startswith('#d'):
                fields = line.split(',')
                fields = [d if d != '_' else 0 for d in fields]
                if len(fields)>5:
                    # retrieve the watt read
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

def print_inet_ip():
    # print the inet ip, the ip address needed for rpi to call this server
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    print("wattsup_server ip:", s.getsockname()[0])
    s.close()

def main():
    # search for WattsUp usb serial
    serial_filepath = ""
    for filename in os.listdir("/dev/"):
        if "tty.usbserial" in filename:
            serial_filepath = "/dev/" + filename
            break

    if serial_filepath == "":
        print >>sys.stderr, "fatal error: cant find usb serial"
        exit(1)

    print_inet_ip()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip",
        type=str,
        default="0.0.0.0",
        help="The ip address to run this server on")
    parser.add_argument(
        "--port",
        type=str,
        default="5000",
        help="The port to run this server on")
    flags, _ = parser.parse_known_args()

    global meter
    meter = WattsUp(serial_filepath)

    thread = Thread(target=start_wattsup_process, args=(meter,))
    thread.start()
    app.run(flags.ip, port=int(flags.port))
    meter.stop()

if __name__ == '__main__':
    main()
