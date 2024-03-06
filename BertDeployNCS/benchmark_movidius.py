try: from openvino.inference_engine import IECore, IENetwork
except ImportError: print('Make sure you activated setupvars.sh!')
import sys
import time
import numpy as np
import lightweight_dataframes as dataframes
# Prepare the dataset
############################################################################################
# Dataset

if len(sys.argv) < 3:
  print("File not specified")
  print("Usage:",sys.argv[0], "<model_dir> <model_name>")
  exit()

model_dir = sys.argv[1]
filename = sys.argv[2]
model_name = filename

val_x = np.loadtxt("data/mrpc_val_x.txt")
val_x = np.reshape(val_x, (val_x.shape[0], 3, 128))
val_y = np.loadtxt("data/mrpc_val_y.txt")

# Set up the core -> Finding your device and loading the model
############################################################################################
ie = IECore()
device_list = ie.available_devices
print("Found devices --> ", device_list)

currentBaseTime = time.time()
# Load any network from file
model_xml = model_dir + "/" + filename + ".xml"
model_bin = model_dir + "/" + filename + ".bin"
#net = IENetwork(model=model_xml, weights=model_bin)
net = ie.read_network(model=model_xml, weights=model_bin)

interpreterReadTime1 = (time.time() - currentBaseTime)*1000

# create some kind of blob
_ = next(iter(net.outputs))
_ = next(iter(net.outputs))
out_blob = next(iter(net.outputs))
inp1 = "input_ids"
inp2 = "attention_mask"
inp3 = "token_type_ids"

# Input Shape
#print('Inp1 Shape: ' + str(net.inputs[inp1].shape))
#print('Inp2 Shape: ' + str(net.inputs[inp2].shape))
#print('Inp3 Shape: ' + str(net.inputs[inp3].shape))

print("Out name:", out_blob)

# Load model to device
##############################################################################################

currentBaseTime = time.time()
exec_net = ie.load_network(network=net, device_name='MYRIAD')
allocationTime1 = (time.time() - currentBaseTime)*1000 # ms


# Run Inference
##############################################################################################

num_samples = 100
avg_time = 0
df = dataframes.createDataFrame(columnNames=["model_name", "avg inference time (ms)","alloc_time (ms)", "total_inference_time (s)", "load_time (ms)", "num_samples", "file_name"])
for i in range(num_samples):
    start_time = time.time()
    x_sample = val_x[i]
    word_ids = np.reshape(x_sample[0], (1,x_sample.shape[1]))
    mask = np.reshape(x_sample[1], (1,x_sample.shape[1]))
    type_ids = np.reshape(x_sample[2], (1,x_sample.shape[1]))
    res = exec_net.infer(inputs={
        inp1: word_ids,
        inp2: mask,
        inp3: type_ids
    })
    res = res[out_blob]
    infer_time = time.time() - start_time
    avg_time+=infer_time

avg_infer_time = (avg_time/num_samples)*1000
print(filename)
print("Model Load Time is", interpreterReadTime1, "(ms)")
print("Allocation Time is", allocationTime1, "(ms)")
print("Inference Time is", avg_time, "(secs)")
print("Total Time is", interpreterReadTime1/1000 + allocationTime1/1000 + avg_time, "(secs)")
print("Avg Time for one inference:", avg_infer_time, "(ms)")
df = dataframes.append_row(df, {"model_name":model_name,"avg inference time (ms)":avg_infer_time, "alloc_time (ms)":allocationTime1, "total_inference_time (s)":avg_time, "load_time (ms)":interpreterReadTime1, "num_samples":num_samples, "file_name":filename})
dataframes.to_csv(df, model_name + "_benchmark.csv")
