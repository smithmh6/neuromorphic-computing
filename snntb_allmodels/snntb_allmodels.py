# import dependencies
import configparser
import h5py
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

# create datasets
data_path = pathlib.Path(r'../datasets/ck_plus_48.npz')
with np.load(data_path) as data:
    images = data['images']
    labels = data['labels']
    classes = data['classes']
print(f"Found {images.shape[0]} examples in {labels.shape[1]} classes.")
print(f"Shape= {(images.shape[1], images.shape[2])}")
print(f"Class Names= {classes}")

tf.random.set_seed(1)

# constants
CLASS_LABELS = classes
NUM_CLASSES = len(classes)
BATCH_SIZE = 24
IMAGE_DIMS = (48, 48)
SEED = 123
_TRAIN = .75
_VAL = .1
_TEST = .15

# create a dataset object and shuffle
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(len(images) * BATCH_SIZE, seed=SEED)
shuffle_images = []
shuffle_labels = []
for im, l in dataset.as_numpy_iterator():
    shuffle_images.append(im)
    shuffle_labels.append(l)
shuffle_images = np.asarray(shuffle_images)
shuffle_labels = np.asarray(shuffle_labels)

# create the slicing indices
TRAIN = round(len(dataset) * _TRAIN)
VAL = TRAIN + round(len(dataset) * _VAL)
TEST = VAL + round(len(dataset) * _TEST)

train_ds = utils.data_pipeline(
    shuffle_images[0:TRAIN], shuffle_labels[0:TRAIN], IMAGE_DIMS,
    edges=True, batch_size=BATCH_SIZE, flip=False, haar=False)
val_ds = utils.data_pipeline(
    shuffle_images[TRAIN:VAL], shuffle_labels[TRAIN:VAL], IMAGE_DIMS,
    edges=True, batch_size=BATCH_SIZE, flip=False, haar=False)
test_ds = utils.data_pipeline(
    shuffle_images[VAL:TEST], shuffle_labels[VAL:TEST], IMAGE_DIMS,
    edges=True, batch_size=BATCH_SIZE, haar=False)

print(f"Using {len(train_ds)*BATCH_SIZE} samples for training.")
print(f"Using {len(val_ds)*BATCH_SIZE} samples for validation.")
print(f"Using {len(test_ds)*BATCH_SIZE} samples for testing.")

# store the data for SNN conversion/simulation
x_test = []
y_test = []
x_norm = []

# create the dataset for SNN simulation
for im, l in test_ds.as_numpy_iterator():
    for i in range(np.shape(im)[0]):
        #print(im[i, :, :, :].shape)
        x_test.append(im[i, :, :, :])
        y_test.append(l[i, :])

# create a dataset for normalization
for im, l in list(train_ds.as_numpy_iterator()):
    for i in range(np.shape(im)[0]):
        #print(im[i, :, :, :].shape)
        x_norm.append(im[i, :, :, :])

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_norm = np.asarray(x_norm)

np.savez_compressed(r'./data/x_test', x_test)
np.savez_compressed(r'./data/y_test', y_test)
np.savez_compressed(r'./data/x_norm', x_norm)

top_models = [
    'VGG_b2k16k224k344k452u195u2120p1_0.22p2_0.12',
    'VGG_b2k16k228k348k460u180u2115p1_0.11p2_0.11',
    'VGG_b2k16k232k344k460u190u2105p1_0.25p2_0.2',
    'VGG_b2k16k232k344k464u185u2115p1_0.21p2_0.17',
    'VGG_b2k18k224k336k460u195u2110p1_0.15p2_0.29',
    'VGG_b2k18k224k348k452u190u2105p1_0.15p2_0.15',
    'VGG_b2k18k232k348k460u190u2100p1_0.15p2_0.19',
    'VGG_b2k18k232k348k464u190u2115p1_0.15p2_0.11',
    'VGG_b2k110k228k340k456u195u2115p1_0.13p2_0.16',
    'VGG_b2k110k228k348k460u1100u2110p1_0.13p2_0.18',
]

model_name = top_models[8]
test_model = tf.keras.models.load_model(f'./models/{model_name}.h5')
test_model.summary()

for layer in test_model.layers[4:]:
    print(layer)

    # build model
model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=(48, 48, 1)),

    tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),


    tf.keras.layers.Conv2D(filters=28, kernel_size=(3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=28, kernel_size=(3, 3), padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

])
for layer in test_model.layers[4:]:
    print(f"Adding layer {layer.name}")
    model.add(layer)

model.summary()

# copy Sequential_1
for i in range(0, 3):
    print(f"Copying weights from {test_model.layers[0].layers[i].name} ---> {model.layers[i].name}")
    model.layers[i].set_weights(test_model.layers[0].layers[i].get_weights())

# copy max pooling 1
print(f"Copying weights from {test_model.layers[1].name} ---> {model.layers[3].name}")
model.layers[3].set_weights(test_model.layers[1].get_weights())

# copy Sequential_2
for i in range(4, 7):
    print(f"Copying weights from {test_model.layers[2].layers[i - 4].name} ---> {model.layers[i].name}")
    model.layers[i].set_weights(test_model.layers[2].layers[i - 4].get_weights())

# copy max pooling 2
print(f"Copying weights from {test_model.layers[3].name} ---> {model.layers[7].name}")
model.layers[7].set_weights(test_model.layers[3].get_weights())

# copy remaining layers
for i in range(8, len(test_model.layers)):
    print(f"Copying weights from {test_model.layers[i - 4].name} ---> {model.layers[i].name}")
    model.layers[i].set_weights(test_model.layers[i - 4].get_weights())

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

if os.path.exists(f'./out/{model_name}.h5'): os.system(f"rm ./out/{model_name}.h5")

# add checkpoint callback
checkpoint = tf.keras.callbacks.ModelCheckpoint(
                f'./out/{model_name}.h5',
                monitor='val_accuracy',
                verbose=2,
                save_best=True,
                save_weights_only=False,
                mode='max')

# add early stopping callback
print("\nConfiguring early stopping parameters.")
early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=.005,
                patience=5,
                verbose=2,
                mode='auto')

hist = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=25,
                verbose=1,
                callbacks=[checkpoint, early_stop])

loss, acc = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"loss ---> {loss}\naccuracy ---> {acc}")

# create config file for snntoolbox
config = configparser.ConfigParser()

# set up data/output paths
config['paths'] = {
    'path_wd': './out',
    'dataset_path': './data',
    'runlabel': model_name,
    'filename_ann': model_name
}

# configure tools
config['tools'] = {
    'evaluate_ann': False,
    'parse': True,
    'normalize': True,
    'simulate': True
}

# configure conversion parameters
config['conversion'] = {
    'max2avg_pool': False,
    'maxpool_type': 'avg_max'
}

# configure simulation settings
config['simulation'] = {
    'simulator': 'INI',
    'duration': 72,
    'batch_size': 24,
    'num_to_test': 144,
    'keras_backend': 'tensorflow'
}

# configure the cell parameters
config['cell'] = {
    'reset': """'Reset by subtraction'""",
    'v_thresh': 1.0
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

# clean up resources
if os.path.exists(f'./out/{model_name}_INI.h5'): os.system(f"rm ./out/{model_name}_INI.h5")
if os.path.exists(f'./out/{model_name}_parsed.h5'): os.system(f"rm ./out/{model_name}_parsed.h5")
