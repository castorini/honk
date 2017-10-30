import service
import requests
import time
import json
import numpy as np

WATTUP_SERVER_IP = "192.168.45.16"

non_keyword_set = set(["_silence_", "_unknown_"])
keyword_list = ["_silence_", "_unknown_", "yes", "no", "up", "down", \
    "left", "right", "on", "off", "stop", "go"]
model_list = ["cnn_one_fstride4.onnx", "cnn_one_fstride8.onnx", "cnn_tpool2.onnx", "cnn_tpool3.onnx", "cnn_trad_fpool3.onnx", "google-speech-dataset-full.onnx", "google-speech-dataset-compact.onnx"]

get_read_url = "http://%s:5000/get_read" % WATTUP_SERVER_IP
get_last_read_url = "http://%s:5000/get_last_read" % WATTUP_SERVER_IP
reset_read_url = "http://%s:5000/reset_read" % WATTUP_SERVER_IP

def wait_util_idle(idle_read=2.2):
    for _ in xrange(15):
        time.sleep(5)
        last_read = float(requests.get(get_last_read_url).json())
        if last_read <= idle_read:
            break

for model in model_list:
    serv = service.Caffe2LabelService("model/%s" % model, keyword_list)
    # print "model %s has loaded" % model
    model_accuracy = []
    model_consumption = []
    model_duration = []
    model_peak = -1

    for ind, keyword in enumerate(keyword_list):
        if keyword in non_keyword_set:
            continue

        wait_util_idle()
        # print "proceed with keyword", keyword
        requests.get(reset_read_url)
        start_time = time.time()

        accuracy = round(serv.evaluate([keyword], [ind]), 3)
        duration = round(time.time() - start_time, 1)
        read_dic = requests.get(get_read_url).json()
        consumption = round(read_dic["consumption"], 1)
        peak = read_dic["peak"]

        print model, keyword, accuracy, duration, consumption, peak

        model_accuracy.append(accuracy)
        model_duration.append(duration)
        model_consumption.append(consumption)
        model_peak = max(model_peak, peak)

    print "model:", model
    print "avg accuracy:", np.mean(model_accuracy)
    print "avg duration:", np.mean(model_duration)
    print "avg consumption:", np.mean(model_consumption)
    print "peak watt:", model_peak, "\n"
