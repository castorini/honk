import service
import requests
import time
import json
import numpy as np
import argparse

non_keyword_set = set(["_silence_", "_unknown_"])
keyword_list = ["_silence_", "_unknown_", "yes", "no", "up", "down", \
    "left", "right", "on", "off", "stop", "go"]
model_list = ["cnn_one_fstride4.onnx", "cnn_one_fstride8.onnx", "cnn_tpool2.onnx", "cnn_tpool3.onnx", "cnn_trad_fpool3.onnx", "google-speech-dataset-full.onnx", "google-speech-dataset-compact.onnx"]


def wait_util_idle(get_last_read_url, idle_read=2.2):
    for _ in xrange(15):
        time.sleep(5)
        last_read = float(requests.get(get_last_read_url).json())
        if last_read <= idle_read:
            break

def evaluate_model(get_read_url, get_last_read_url, reset_read_url):
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

            wait_util_idle(get_last_read_url)
            # print "proceed with keyword", keyword
            requests.get(reset_read_url)
            start_time = time.time()

            accuracy = round(serv.evaluate([keyword], [ind]), 3)
            duration = round(time.time() - start_time, 1)
            read_dic = requests.get(get_read_url).json()
            consumption = round(read_dic["consumption"], 1)
            peak = read_dic["peak"]

            print(model, keyword, accuracy, duration, consumption, peak)

            model_accuracy.append(accuracy)
            model_duration.append(duration)
            model_consumption.append(consumption)
            model_peak = max(model_peak, peak)

        print("model:", model)
        print("avg accuracy:", np.mean(model_accuracy))
        print("avg duration:", np.mean(model_duration))
        print("avg consumption:", np.mean(model_consumption))
        print("peak watt:", model_peak, "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip",
        type=str,
        default="",
        help="The ip address to run this server on")
    parser.add_argument(
        "--port",
        type=str,
        default="",
        help="The port to run this server on")
    flags, _ = parser.parse_known_args()

    if not flags.ip or not flags.port:
        print("the ip address and the port of the wattsup_server must be provided")
        exit(0)

    wattsup_server_ip, wattsup_server_port = flags.ip, flags.port

    get_read_url = "http://%s:%s/get_read" % (wattsup_server_ip, wattsup_server_port)
    get_last_read_url = "http://%s:%s/get_last_read" % (wattsup_server_ip, wattsup_server_port)
    reset_read_url = "http://%s:%s/reset_read" % (wattsup_server_ip, wattsup_server_port)

    evaluate_model(get_read_url, get_last_read_url, reset_read_url)

if __name__ == '__main__':
    main()
