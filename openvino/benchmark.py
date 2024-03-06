"""
Benchmark model performance with OpenVINO.
"""

import argparse as ap
import numpy as np
from openvino.inference_engine import IECore, IENetwork
import os
import pathlib
import sys
import time
from tqdm import tqdm
import utils

# create the arg parser
def get_args():

    # parse command line args
    print("\nParsing command line args...")

    parser = ap.ArgumentParser(
        prog="train.py",
        description="Train a convolutional neural network."
    )
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-d", "--device", type=str)

    return parser.parse_args()

def main(args):

    # start timer
    start = time.perf_counter()

    print("\nLoading model {args.model}")
    base_path ="/home/heathsmith/repos/neuromorphic-computing/openvino/out"
    model_dir = os.path.join(base_path, args.model)
    model_xml = os.path.join(model_dir, "inference_graph.xml")
    model_bin = os.path.join(model_dir, "inference_graph.bin")
    t = utils.op_timer(start)


    # read compressed dataset
    print(f"\nLoading test data...")
    x_data = os.path.join(model_dir, 'x_test.npz')
    y_data = os.path.join(model_dir, 'y_test.npz')
    with np.load(x_data) as x:
        x_test = x['arr_0']
    with np.load(y_data) as y:
        y_test = y['arr_0']

    print(f"X_Data shape ---> {x_test.shape}")
    print(f"Y_Data shape ---> {y_test.shape}")
    t = utils.op_timer(t)


    # Set up the core -> Finding your device and loading the model
    print("\nSetting up IE Core...")
    ie = IECore()

    device_list = ie.available_devices
    print(f"Found devices ---> {device_list}")
    print(f"Selected device ---> {args.device}")
    #if args.device not in device_list:
    #    raise ValueError(f"Device {args.device} not found! Available devices: [{device_list}]")
    t = utils.op_timer(t)


    print("\nReading network...")
    read_net = ie.read_network(model=model_xml, weights=model_bin)
    t = utils.op_timer(t)


    print(f"\nLoading network to device {args.device}...")
    load_net = ie.load_network(network=read_net, device_name=args.device)
    t = utils.op_timer(t)

    # extract input/output information
    input_key = list(load_net.input_info)[0]
    output_key = list(load_net.outputs.keys())[0]
    input_shape = load_net.input_info[input_key].tensor_desc.dims

    print(f"\nInput shape ---> {input_shape}")
    print(f"Input key ---> {input_key}")
    print(f"Output key ---> {output_key}")


    # allocate storage for sample results
    n_iters = 10
    n_samples = x_test.shape[0]
    t_samples = np.zeros(n_samples)
    predictions = np.zeros(n_samples)

    print(f"\nRunning inference on {n_samples} samples..")
    t = time.perf_counter()

    for i in tqdm(range(n_samples)):
        sum_j = 0

        # prepare the next sample, start timer
        #print(x_test[i])
        sample = np.expand_dims(np.squeeze(x_test[i]), 0)
        sample = np.expand_dims(sample, 0)
        #print(sample.shape)

        for j in range(n_iters):

            start_j = time.perf_counter()

            # execute the inference, end timer
            result = load_net.infer(inputs={input_key: sample})[output_key]
            end_j = time.perf_counter()

            # store the elapsed time
            elapsed_j = end_j - start_j
            sum_j += elapsed_j

        # calculate avg time for 1000 samples
        elapsed_i = sum_j / n_iters
        t_samples[i] = elapsed_i

        # store prediction result
        #print(result)
        predictions[i] = int(np.argmax(result) == np.argmax(y_test[i]))
        #print(np.argmax(result), np.argmax(y_test[i]))

    t = utils.op_timer(t)

    # calculate timing statistics
    t_avg = round((np.mean(t_samples)*1000), 5)
    t_stdev = round(np.std(t_samples, axis=0)*1000, 5)
    t_max = round((np.max(t_samples)*1000), 5)
    t_min = round((np.min(t_samples)*1000), 5)
    t_total_s = np.sum(t_samples)
    t_total = round(t_total_s*1000, 5)

    print(f"\nAvg. Inference time ---> {t_avg} +/- {t_stdev} ms")
    print(f"Min Inference time ---> {t_min} ms")
    print(f"Max Inference time ---> {t_max} ms")
    print(f"Total Inference time ---> {t_total} ms")
    print(f"Samples per second ---> {n_samples/t_total_s}")

    # calculate accuracy results
    acc = np.count_nonzero(predictions) / n_samples
    print(f"Accuracy ---> {acc}")


if __name__ == '__main__':

    print('\n' + '-'*80)

    args = get_args()

    main(args)

