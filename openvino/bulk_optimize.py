

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
from tqdm import tqdm

model_path = r'/media/heathsmith/843E-918C/models'
out_path = r'/media/heathsmith/843E-918C/ir_models'

models = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
print(f"\nLOG ---> Found {len(models)} models.")

for model in tqdm(models):
    input_model = os.path.join(model_path, model)

    # Construct the command for Model Optimizer
    mo_command = f"""mo --input_model "{input_model}" --input_shape "[1, 48, 48, 1]" --data_type FP16 --output_dir "{out_path}" """
    mo_command = " ".join(mo_command.split())
    print('-'*80)
    print('\nRunning Model Optimizer...')
    print(f'{mo_command}')

    os.system(str(mo_command))
    print('-'*80)
