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
tf.get_logger().addFilter(lambda rec: "This TensorFlow binary" not in rec.msg)
nengo_loihi.set_defaults()

def line_break():
    print('-'*80)

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
EPOCHS = 25
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

line_break()
print(f"Using {len(train_ds)*BATCH_SIZE} samples for training.")
print(f"Using {len(val_ds)*BATCH_SIZE} samples for validation.")
print(f"Using {len(test_ds)*BATCH_SIZE} samples for testing.")

line_break()



# separate images and labels for Nengo
train_images = shuffle_images[0:TRAIN]
train_labels = shuffle_labels[0:TRAIN]
train_images = np.asarray(train_images)
train_images = train_images.reshape((train_images.shape[0], -1))
train_labels = np.asarray(train_labels)
train_labels = train_labels.reshape((train_labels.shape[0], -1))
train_labels = np.asarray(
    [np.argmax(a) for a in train_labels])

test_images = shuffle_images[VAL:TEST]
test_labels = shuffle_labels[VAL:TEST]
test_images = np.asarray(test_images)
test_images = test_images.reshape((test_images.shape[0], -1))
test_labels = np.asarray(test_labels)
test_labels = test_labels.reshape((test_labels.shape[0], -1))
test_labels = np.asarray(
    [np.argmax(a) for a in test_labels])

line_break()
print(np.shape(train_images), train_labels.shape)
print(np.shape(test_images), test_labels.shape)

def conv_layer(x, *args, activation=True, **kwargs):
    # create a Conv2D transform with the given arguments
    conv = nengo.Convolution(*args, channels_last=False, **kwargs)

    if activation:
        # add an ensemble to implement the activation function
        layer = nengo.Ensemble(conv.output_shape.size, 1).neurons
    else:
        # no nonlinearity, so we just use a node
        layer = nengo.Node(size_in=conv.output_shape.size)

    # connect up the input object to the new layer
    nengo.Connection(x, layer, transform=conv)

    # print out the shape information for our new layer
    print("LAYER")
    print(conv.input_shape.shape, "->", conv.output_shape.shape)

    return layer, conv

dt = 0.001  # simulation timestep
presentation_time = 0.02  # input presentation time
max_rate = 100  # neuron firing rates
# neuron spike amplitude (scaled so that the overall output is ~1)
amp = 1 / max_rate
# input image shape
input_shape = (1, 48, 48)
n_parallel = 2  # number of parallel network repetitions

with nengo.Network(seed=0) as net:
    # set up the default parameters for ensembles/connections
    nengo_loihi.add_params(net)
    net.config[nengo.Ensemble].neuron_type = nengo.SpikingRectifiedLinear(amplitude=amp)
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    # the input node that will be used to feed in input images
    inp = nengo.Node(
        nengo.processes.PresentInput(test_images, presentation_time), size_out=48 * 48
    )

    # the output node provides the N-dimensional classification
    out = nengo.Node(size_in=NUM_CLASSES)

    # build parallel copies of the network
    for _ in range(n_parallel):
        layer, conv = conv_layer(
            inp, 1, input_shape, kernel_size=(1, 1), init=np.ones((1, 1, 1, 1))
        )
        # first layer is off-chip to translate the images into spikes
        net.config[layer.ensemble].on_chip = False
        layer, conv = conv_layer(layer, 12, conv.output_shape, strides=(2, 2))
        layer, conv = conv_layer(layer, 24, conv.output_shape, strides=(2, 2))
        layer, conv = conv_layer(layer, 32, conv.output_shape, strides=(2, 2))
        layer, conv = conv_layer(layer, 48, conv.output_shape, strides=(2, 2))

        nengo.Connection(layer, out, transform=nengo_dl.dists.Glorot())

    out_p = nengo.Probe(out, label="out_p")
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01), label="out_p_filt")


# set up training data, adding the time dimension (with size 1)
minibatch_size = 10
train_images = train_images[:, None, :]
train_labels = train_labels[:, None, None]

# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
n_steps = int(presentation_time / dt)
test_images = np.tile(test_images[: minibatch_size * 2, None, :], (1, n_steps, 1))
test_labels = np.tile(test_labels[: minibatch_size * 2, None, None], (1, n_steps, 1))

line_break()
print(np.shape(train_images), train_labels.shape)
print(np.shape(test_images), test_labels.shape)
line_break()


def classification_accuracy(y_true, y_pred):
    return 100 * tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])


with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=0) as sim:
    sim.compile(loss={out_p_filt: classification_accuracy})
    print(
        "accuracy before training: %.2f%%"
        % sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"]
    )

    # run training
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
    )
    sim.fit(train_images, train_labels, epochs=EPOCHS)

    sim.compile(loss={out_p_filt: classification_accuracy})
    print(
        "accuracy after training: %.2f%%"
        % sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"]
    )

    sim.save_params("./ckplus_params")

    # store trained parameters back into the network
    sim.freeze_params(net)

for conn in net.all_connections:
    conn.synapse = 0.005

with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
    sim.compile(loss={out_p_filt: classification_accuracy})
    print(
        "accuracy w/ synapse: %.2f%%"
        % sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"]
    )

n_presentations = 5

# if running on Loihi, increase the max input spikes per step
hw_opts = dict(snip_max_spikes_per_step=120)
with nengo_loihi.Simulator(
    net,
    dt=dt,
    precompute=True,
    hardware_options=hw_opts,
) as sim:
    print("Running simulation on Loihi...")

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    print("Checking classification accuracy...")

    # check classification accuracy
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1 :: step]

    correct = 100 * np.mean(
        np.argmax(output, axis=-1) == test_labels[:n_presentations, -1, 0]
    )
    print("loihi accuracy: %.2f%%" % correct)

show_plot = False
if show_plot:
    line_break()
    print("Plotting test results...")
    n_plots = 5
    plt.figure()

    plt.subplot(2, 1, 1)
    images = test_images.reshape(-1, 48, 48, 1)[::step]
    ni, nj, nc = images[0].shape
    allimage = np.zeros((ni, nj * n_plots, nc), dtype=images.dtype)
    for i, image in enumerate(images[:n_plots]):
        allimage[:, i * nj : (i + 1) * nj] = image
    if allimage.shape[-1] == 1:
        allimage = allimage[:, :, 0]
    plt.imshow(allimage, aspect="auto", interpolation="none", cmap="gray")

    plt.subplot(2, 1, 2)
    plt.plot(sim.trange()[: n_plots * step], sim.data[out_p_filt][: n_plots * step])
    plt.legend(["%d" % i for i in range(10)], loc="best")

print("Done!")
