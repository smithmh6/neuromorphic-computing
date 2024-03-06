# import dependencies
import configparser
import json
import numpy as np
import os
import pathlib
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import shutil
import tensorflow as tf
import utils
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import snntoolbox.bin.run as snn

# get list of models
models = [str(x).replace('.h5', '') for x in os.listdir(r'/Volumes/USB/models') if x.endswith('h5')]
print(f"Found {len(models)} models.")

models = models[5:-1]
for m in models:

    print(f"Simulating model {m} :")

    os.remove('./config.ini')

    # create config file for snntoolbox
    config = configparser.ConfigParser()

    # set up data/output paths
    config['paths'] = {
        'path_wd': '/Volumes/USB/snntoolbox_output',
        'dataset_path': '/Volumes/USB/data',
        'runlabel': m,
        'filename_ann': m
    }

    # configure tools
    config['tools'] = {
        'evaluate_ann': False,
        'parse': True,
        'normalize': False,
        'simulate': True
    }

    # configure conversion parameters
    config['conversion'] = {
        'max2avg_pool': False
    }

    # configure simulation settings
    config['simulation'] = {
        'simulator': 'INI',
        'duration': 64,
        'batch_size': 24,
        'num_to_test': 144,
        'keras_backend': 'tensorflow'
    }

    # configure output parameters
    config['output'] = {
        'plot_vars': {
            # 'input_image',
            # 'spiketrains',
            # 'spikerates',
            # 'spikecounts',
            # 'operations',
            # 'normalization_activations',
            # 'activations',
            # 'correlation',
            # 'v_mem',
            # 'error_t'
        },
        'verbose': 0,
        'overwrite': True
    }

    with open('./config.ini', 'w') as configfile:
        config.write(configfile)

    # run snn conversion/simulation
    snn.main("./config.ini")