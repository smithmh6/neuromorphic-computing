import warnings

import matplotlib.pyplot as plt
import nengo
from nengo.utils.matplotlib import rasterplot
import numpy as np
import tensorflow as tf

import nengo_dl
import nengo_loihi
import sys
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

warnings.simplefilter("ignore")
tf.get_logger().addFilter(lambda rec: "Tracing is expensive" not in rec.msg)
nengo_loihi.set_defaults()


# read compressed dataset
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
    edges=False, batch_size=BATCH_SIZE, flip=True)
val_ds = utils.data_pipeline(
    shuffle_images[TRAIN:VAL], shuffle_labels[TRAIN:VAL], IMAGE_DIMS,
    edges=False, batch_size=BATCH_SIZE, flip=True)
test_ds = utils.data_pipeline(
    shuffle_images[VAL:TEST], shuffle_labels[VAL:TEST], IMAGE_DIMS,
    edges=False, batch_size=BATCH_SIZE)

print(f"Using {len(train_ds)*BATCH_SIZE} samples for training.")
print(f"Using {len(val_ds)*BATCH_SIZE} samples for validation.")
print(f"Using {len(test_ds)*BATCH_SIZE} samples for testing.")

# construct the model
k_size = (3, 3)
p_size = (2, 2)
stride = None #(1, 1)


input_lyr = tf.keras.layers.Input(shape=IMAGE_DIMS + (1, ))

conv0 = tf.keras.layers.Conv2D(
        filters=12, kernel_size=k_size, padding="same", activation="relu")(input_lyr)
avg0 = tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride)(conv0)

conv1 = tf.keras.layers.Conv2D(
        filters=24, kernel_size=k_size, padding="same", activation="relu")(avg0)
avg1 = tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride)(conv1)

conv2 = tf.keras.layers.Conv2D(
        filters=32, kernel_size=k_size, padding="same", activation="relu")(avg1)
avg2 = tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride)(conv2)

conv3 = tf.keras.layers.Conv2D(
        filters=48, kernel_size=k_size, padding="same", activation="relu")(avg2)
batch_norm = tf.keras.layers.BatchNormalization()(conv3)
avg3 = tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride)(batch_norm)

flat = tf.keras.layers.Flatten()(avg3)
dense0 = tf.keras.layers.Dense(units=48,activation="relu")(flat)
drop0 = tf.keras.layers.Dropout(0.2)(dense0)
dense1 = tf.keras.layers.Dense(units=24,activation="relu")(drop0)
drop1 = tf.keras.layers.Dropout(0.2)(dense1)
output_lyr = tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax")(drop1)

model = tf.keras.Model(inputs=input_lyr, outputs=output_lyr)

### # display the model summary
model.summary()

# set epochs
NUM_EPOCHS = 5
DECAY_STEPS = (len(train_ds)) * 10
print(f"Decay steps per epoch= {DECAY_STEPS}")

# define learning rate decay
lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.03, decay_steps=DECAY_STEPS, decay_rate=1, staircase=False
)

# plot the learning rate
steps = np.arange(NUM_EPOCHS)
lr = lr_decay(steps)


fig = go.Figure()
fig.add_trace(go.Scatter(x=steps, y=lr,
                    mode='lines',
                    name='lines'))
fig.update_layout(
    template='plotly_dark',
    title={'text': "Learning Rate Decay", 'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
    xaxis_title="Epochs",
    yaxis_title="LR Value",)

# save and display the plot image
## fig.write_image("./tests/learning_rate_decay.png", width=1000)
## fig.show()


converter = nengo_dl.Converter(model) #, scale_firing_rates=2)

# separate images and labels for Nengo
train_images = shuffle_images[0:TRAIN]
train_labels = shuffle_labels[0:TRAIN]
train_images = np.asarray(train_images)
train_images = train_images.reshape((train_images.shape[0], 1, -1))
train_labels = np.asarray(train_labels)
train_labels = train_labels.reshape((train_labels.shape[0], 1, -1))

test_images = shuffle_images[VAL:TEST]
test_labels = shuffle_labels[VAL:TEST]
test_images = np.asarray(test_images)
test_images = test_images.reshape((test_images.shape[0], 1, -1))
test_labels = np.asarray(test_labels)
test_labels = test_labels.reshape((test_labels.shape[0], 1, -1))

TRAIN_NEW = True
if TRAIN_NEW:
    with nengo_loihi.Simulator(converter.net, minibatch_size=BATCH_SIZE) as sim:
        # run training
        sim.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr_decay),
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=['accuracy'],#tf.metrics.sparse_categorical_accuracy],
        )
        sim.fit(
            {converter.inputs[input_lyr]: train_images},
            {converter.outputs[output_lyr]: train_labels},
            validation_data=(
                {converter.inputs[input_lyr]: test_images},
                {converter.outputs[output_lyr]: test_labels},
            ),
            epochs=NUM_EPOCHS,
        )

        # save the parameters to file
        sim.save_params("./keras_to_snn_params")

# activation = nengo.RectifiedLinear()
activation = nengo.SpikingRectifiedLinear()
scale_firing_rates = 1
n_steps = 20
n_test = 144 #test_labels.shape[0] # 400

# convert the keras model to a nengo network
nengo_converter = nengo_dl.Converter(
    model,
    swap_activations={tf.nn.relu: activation},
    scale_firing_rates=scale_firing_rates,
    synapse=0.001,
)

# get input/output objects
nengo_input = nengo_converter.inputs[input_lyr]
nengo_output = nengo_converter.outputs[output_lyr]

# add a probe to the first convolutional layer to record activity.
# we'll only record from a subset of neurons, to save memory.
sample_neurons = np.linspace(
    0,
    np.prod(conv0.shape[1:]),
    1000,
    endpoint=False,
    dtype=np.int32,
)
sample_neurons_2 = np.linspace(
    0,
    np.prod(conv3.shape[1:]),
    1000,
    endpoint=False,
    dtype=np.int32,
)

with nengo_converter.net:
    conv0_probe = nengo.Probe(nengo_converter.layers[conv0][sample_neurons])
    conv3_probe = nengo.Probe(nengo_converter.layers[conv3][sample_neurons_2])

# repeat inputs for some number of timesteps
tiled_test_images = np.tile(test_images[:n_test], (1, n_steps, 1))

# set some options to speed up simulation
with nengo_converter.net:
    nengo_dl.configure_settings(stateful=False)

# build network, load in trained weights, run inference on test images
print('-'*80)
print('Running inference on test data...')
with nengo_dl.Simulator(
    nengo_converter.net, minibatch_size=BATCH_SIZE, progress_bar=True
) as nengo_sim:
    nengo_sim.load_params("keras_to_snn_params")
    data = nengo_sim.predict({nengo_input: tiled_test_images})

print('-'*80)
# compute accuracy on test data, using output of network on
# last timestep
predictions = np.argmax(data[nengo_output][:, -1], axis=-1)

## print(type(predictions))
## print(np.shape(predictions))
## print(predictions)
##
## print(type(test_labels[:n_test, 0, 0]))
## print(np.shape(test_labels[:n_test, 0, 0]))
## print(test_labels[:n_test, 0, 0])
##
## print(type(predictions == test_labels[:n_test, 0, 0]))
## print(np.shape(predictions == test_labels[:n_test, 0, 0]))
## print(predictions == test_labels[:n_test, 0, 0])

accuracy = (predictions == test_labels[:n_test, 0, 0]).mean()
print(f"Test accuracy: {100 * accuracy:.2f}%")


print('-'*80)
print("Plotting test results...")
# plot the results
for ii in range(3):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Input image")
    plt.imshow(test_images[ii, 0].reshape((48, 48)), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    scaled_data = data[conv0_probe][ii] * scale_firing_rates
    if isinstance(activation, nengo.SpikingRectifiedLinear):
        scaled_data *= 0.001
        rates = np.sum(scaled_data, axis=0) / (n_steps * nengo_sim.dt)
        plt.ylabel("Number of spikes")
    else:
        rates = scaled_data
        plt.ylabel("Firing rates (Hz)")
    plt.xlabel("Timestep")
    plt.title(
        f"Neural activities (conv0 mean={rates.mean():.1f} Hz, "
        f"max={rates.max():.1f} Hz)"
    )
    plt.plot(scaled_data)

    plt.subplot(1, 4, 3)
    conv3_scaled_data = data[conv3_probe][ii] * scale_firing_rates
    if isinstance(activation, nengo.SpikingRectifiedLinear):
        conv3_scaled_data *= 0.001
        rates = np.sum(conv3_scaled_data, axis=0) / (n_steps * nengo_sim.dt)
        plt.ylabel("Number of spikes")
    else:
        rates = conv3_scaled_data
        plt.ylabel("Firing rates (Hz)")
    plt.xlabel("Timestep")
    plt.title(
        f"Neural activities (conv3 mean={rates.mean():.1f} Hz, "
        f"max={rates.max():.1f} Hz)"
    )
    plt.plot(conv3_scaled_data)

    plt.subplot(1, 4, 4)
    plt.title("Output predictions")
    plt.plot(tf.nn.softmax(data[nengo_output][ii]))
    plt.legend([str(j) for j in range(10)], loc="upper left")
    plt.xlabel("Timestep")
    plt.ylabel("Probability")

    plt.tight_layout()
    #plt.show()

