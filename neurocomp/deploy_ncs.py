"""
Convert model for deployment on Intel Movidius
Neural Compute Stick 2.
"""

# import dependencies
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm
from openvino.inference_engine import IECore

print("Loading X test data...")
with np.load('./tests/x_test.npz') as data:
    x_test = data['arr_0']
print(f"X Test Shape= {x_test.shape}")

print("Loading Y test data...")
with np.load('./tests/y_test.npz') as data:
    y_test = data['arr_0']
print(f"Y Test Shape= {y_test.shape}")

# The paths of the source and converted models
model_path = Path("./test_inference_graph.pb")

# Construct the command for Model Optimizer
#mo_command = f"""python3 /opt/intel/openvino_2021.4.752/deployment_tools/model_optimizer/mo_tf.py
mo_command = f"""mo --input_model "{model_path}" --input_shape "[1, 48, 48, 1]" --data_type FP16 --output_dir "." """
mo_command = " ".join(mo_command.split())
print('-'*80)
print('LOG --> Running Model Optimizer...')
os.system(str(mo_command))
print('-'*80)

ir_path = Path("./test_inference_graph.xml")
ie = IECore()
net = ie.read_network(model=ir_path, weights=Path("./test_inference_graph.bin"))
exec_net = ie.load_network(network=net, device_name="CPU")

input_key = list(exec_net.input_info)[0]
output_key = list(exec_net.outputs.keys())[0]
network_input_shape = exec_net.input_info[input_key].tensor_desc.dims

num_samples = x_test.shape[0]
total_time = 0
true_results = 0

# grab a sample
for i in tqdm(range(0, num_samples)):
    sample = np.expand_dims(np.squeeze(x_test[i]), 0)
    start_time = time.perf_counter()
    result = exec_net.infer(inputs={input_key: sample})[output_key]
    end_time = time.perf_counter()
    total_time += end_time - start_time
    if i % 500 == 0:
        time.sleep(3)
    if np.argmax(result) == np.argmax(y_test[i]):
        true_results +=1
    #print(i)

print('Average inference time: '+str(total_time/num_samples))
print('Accuracy results: '+str(true_results/num_samples))

