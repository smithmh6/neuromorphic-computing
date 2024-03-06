"""
Convert network to nengo loihi.
"""

import collections
import warnings
import json
import matplotlib.pyplot as plt
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import pathlib
import nengo_loihi
import utils
from nxsdk.graph.monitor.probes import PerformanceProbeCondition
from nxsdk.api.n2a import ProbeParameter

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ignore NengoDL warning about no GPU
warnings.filterwarnings("ignore", message="No GPU", module="nengo_dl")

# The results in this notebook should be reproducible across many random seeds.
# However, some seed values may cause problems, particularly in the `to-spikes` layer
# where poor initialization can result in no information being sent to the chip. We set
# the seed to ensure that good results are reproducible without having to re-train.
np.random.seed(1)
tf.random.set_seed(1)

# create datasets
data_path = pathlib.Path(r'../datasets/ck_plus_48.npz')
with np.load(data_path) as data:
    images = data['images']
    labels = data['labels']
    classes = data['classes']
print(f"Found {images.shape[0]} examples in {labels.shape[1]} classes.")
print(f"Shape= {(images.shape[1], images.shape[2])}")
print(f"Class Names= {classes}")


# constants
CLASS_LABELS = classes
NUM_CLASSES = len(classes)
BATCH_SIZE = 24
IMAGE_DIMS = (48, 48)
SEED = 123
_TRAIN = .8
_VAL = 0
_TEST = .2

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

train_images, train_labels = utils.data_pipeline(
    shuffle_images[0:TRAIN], shuffle_labels[0:TRAIN], IMAGE_DIMS,
    edges=True, batch_size=1, flip=False)
train_images, train_labels = np.asarray(train_images), np.asarray(train_labels)
train_labels = np.asarray([np.argmax(a) for a in train_labels])

## val_ds = utils.data_pipeline(
##     shuffle_images[TRAIN:VAL], shuffle_labels[TRAIN:VAL], IMAGE_DIMS,
##     edges=False, batch_size=BATCH_SIZE, flip=True)

test_images, test_labels = utils.data_pipeline(
    shuffle_images[VAL:TEST], shuffle_labels[VAL:TEST], IMAGE_DIMS,
    edges=True, batch_size=1)
test_images, test_labels = np.asarray(test_images), np.asarray(test_labels)
test_labels = np.asarray([np.argmax(a) for a in test_labels])

# flatten images and add time dimension
train_images = train_images.reshape((train_images.shape[0], 1, -1))
train_labels = train_labels.reshape((train_labels.shape[0], 1, -1))
test_images = test_images.reshape((test_images.shape[0], 1, -1))
test_labels = test_labels.reshape((test_labels.shape[0], 1, -1))

print(f"Train images shape ---> {train_images.shape}")
print(f"Train labels shape ---> {train_labels.shape}")

print(f"Test images shape ---> {test_images.shape}")
print(f"Test labels shape ---> {test_labels.shape}")


inp = tf.keras.layers.Input(shape=(48, 48, 1))

# transform input signal to spikes using trainable 1x1 convolutional layer
to_spikes_layer = tf.keras.layers.Conv2D(
    filters=3,  # 3 neurons per pixel
    kernel_size=1,
    strides=1,
    activation=tf.nn.relu,
    use_bias=False,
    name="to-spikes",
)
to_spikes = to_spikes_layer(inp)

# on-chip convolutional block 1
conv0_layer = tf.keras.layers.Conv2D(
    filters=12, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name="conv0", use_bias=False)
conv0 = conv0_layer(to_spikes)
conv1_layer = tf.keras.layers.Conv2D(
    filters=12, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=tf.nn.relu, name="conv1", use_bias=False)
conv1 = conv1_layer(conv0)

# on-chip convolutional block 2
conv2_layer = tf.keras.layers.Conv2D(
    filters=24, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name="conv2", use_bias=False)
conv2 = conv2_layer(conv1)
conv3_layer = tf.keras.layers.Conv2D(
    filters=24, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=tf.nn.relu, name="conv3", use_bias=False)
conv3 = conv3_layer(conv2)

# on-chip convolutional block 3
conv4_layer = tf.keras.layers.Conv2D(
    filters=44, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name="conv4", use_bias=False)
conv4 = conv4_layer(conv3)
conv5_layer = tf.keras.layers.Conv2D(
    filters=44, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=tf.nn.relu, name="conv5", use_bias=False)
conv5 = conv5_layer(conv4)

flatten = tf.keras.layers.Flatten(name="flatten")(conv5)

dense0_layer = tf.keras.layers.Dense(units=85, activation=tf.nn.relu, name="dense0")
dense0 = dense0_layer(flatten)

dense1_layer = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, name="dense1")
dense1 = dense1_layer(dense0)

# since this final output layer has no activation function,
# it will be converted to a `nengo.Node` and run off-chip
dense2 = tf.keras.layers.Dense(units=7)(dense1)

model = tf.keras.Model(inputs=inp, outputs=dense2)
model.summary()


def train(params_file="./out/keras_to_loihi_params", epochs=1, do_training=False, **kwargs):
    converter = nengo_dl.Converter(model, **kwargs)

    with nengo_dl.Simulator(converter.net, seed=0, minibatch_size=BATCH_SIZE) as sim:
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss={
                converter.outputs[dense2]: tf.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                )
            },
            metrics={converter.outputs[dense2]: tf.metrics.sparse_categorical_accuracy},
        )

        if do_training == True:
            sim.fit(
                {converter.inputs[inp]: train_images},
                {converter.outputs[dense2]: train_labels},
                epochs=epochs,
            )

            # save the parameters to file
            sim.save_params(params_file)
        else:
            sim.load_params(params_file)

# train this network with normal ReLU neurons
### print('-'*80)
### print("Training with normal ReLu Neurons..")
### train(
###     params_file="./out/keras_to_loihi_params",
###     epochs=10,
###     swap_activations={tf.nn.relu: nengo.RectifiedLinear()},
###     do_training=True
### )


def run_network(
    activation,
    params_file="./out/keras_to_loihi_params",
    n_steps=25,
    scale_firing_rates=1,
    synapse=None,
    n_test=192,
    n_plots=3,
    outfile=''
):
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        scale_firing_rates=scale_firing_rates,
        swap_activations={tf.nn.relu: activation},
        synapse=synapse,
    )

    # get input/output objects
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[dense2]

    # add probes to layers to record activity
    with nengo_converter.net:
        probes = collections.OrderedDict(
            [
                [to_spikes_layer, nengo.Probe(nengo_converter.layers[to_spikes])],
                [conv0_layer, nengo.Probe(nengo_converter.layers[conv0])],
                [conv1_layer, nengo.Probe(nengo_converter.layers[conv1])],
                [conv2_layer, nengo.Probe(nengo_converter.layers[conv2])],
                [conv3_layer, nengo.Probe(nengo_converter.layers[conv3])],
                [conv4_layer, nengo.Probe(nengo_converter.layers[conv4])],
                [conv5_layer, nengo.Probe(nengo_converter.layers[conv5])],
                [dense0_layer, nengo.Probe(nengo_converter.layers[dense0])],
                [dense1_layer, nengo.Probe(nengo_converter.layers[dense1])],
            ]
        )

    # repeat inputs for some number of timesteps
    tiled_test_images = np.tile(test_images[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=BATCH_SIZE, progress_bar=True
    ) as nengo_sim:
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_test_images})

    # compute accuracy on test data, using output of network on
    # last timestep
    test_predictions = np.argmax(data[nengo_output][:, -1], axis=-1)

    print(
        "Test accuracy: %.2f%%"
        % (100 * np.mean(test_predictions == test_labels[:n_test, 0, 0]))
    )

    # plot the results
    print("Plotting results..")
    mean_rates = []
    for i in range(n_plots):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title("Input image")
        plt.imshow(test_images[i, 0].reshape((48, 48)), cmap="gray")
        plt.axis("off")

        n_layers = len(probes)
        mean_rates_i = []
        for j, layer in enumerate(probes.keys()):
            probe = probes[layer]
            plt.subplot(n_layers, 3, (j * 3) + 2)
            plt.suptitle("Neural activities")

            outputs = data[probe][i]

            # look at only at non-zero outputs
            nonzero = (outputs > 0).any(axis=0)
            outputs = outputs[:, nonzero] if sum(nonzero) > 0 else outputs

            # undo neuron amplitude to get real firing rates
            outputs /= nengo_converter.layers[layer].ensemble.neuron_type.amplitude

            rates = outputs.mean(axis=0)
            mean_rate = rates.mean()
            mean_rates_i.append(mean_rate)
            print(
                '"%s" mean firing rate (example %d): %0.1f' % (layer.name, i, mean_rate)
            )

            if is_spiking_type(activation):
                outputs *= 0.001
                if j == 4:
                    plt.ylabel("# of Spikes")
            else:
                if j == 4:
                    plt.ylabel("Firing rates (Hz)")

            # plot outputs of first 100 neurons
            plt.plot(outputs[:, :100])

        mean_rates.append(mean_rates_i)

        plt.xlabel("Timestep")

        plt.subplot(1, 3, 3)
        plt.title("Output predictions")
        plt.plot(tf.nn.softmax(data[nengo_output][i]))
        plt.legend(classes, loc="upper left")
        plt.xlabel("Timestep")
        plt.ylabel("Probability")

        plt.tight_layout()
        plt.savefig(f"./out/{outfile}_{i}.png")
    # take mean rates across all plotted examples
    mean_rates = np.array(mean_rates).mean(axis=0)

    return mean_rates


def is_spiking_type(neuron_type):
    return isinstance(neuron_type, (nengo.LIF, nengo.SpikingRectifiedLinear))


# test the trained networks on test set
### print('-'*80)
### print('Evaluating with nengo.RectifiedLiner()..')
### mean_rates = run_network(
###     activation=nengo.RectifiedLinear(),
###     n_steps=10,
###     outfile='nengo_test_output')
###
### # test the trained networks using spiking neurons
### print('-'*80)
### print('Evaluating with nengo.SpikingRectifiedLinear()..')
### mean_rates_snn = run_network(
###     activation=nengo.SpikingRectifiedLinear(),
###     scale_firing_rates=100,
###     synapse=0.005,
###     outfile='nengo_spiking_output')

# test the trained networks using spiking neurons
### print('-'*80)
### print('Evaluating with nengo_loihi.neurons.LoihiSpikingRectifiedLinear()..')
### target_mean = 220
### scale_firing_rates = {
###     to_spikes_layer: target_mean / mean_rates[0],
###     conv0_layer: target_mean / mean_rates[1],
###     conv1_layer: target_mean / mean_rates[2],
###     conv2_layer: target_mean / mean_rates[3],
###     conv3_layer: target_mean / mean_rates[4],
###     conv4_layer: target_mean / mean_rates[5],
###     conv5_layer: target_mean / mean_rates[6],
###     dense0_layer: target_mean / mean_rates[7],
###     dense1_layer: target_mean / mean_rates[8]
### }
### mean_rates_loihi = run_network(
###     activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
###     scale_firing_rates=scale_firing_rates,
###     synapse=0.005,
###     n_steps=50,
###     outfile='loihi_spiking_output')

# train this network with loihi neurons
print('-'*80)
print('Training with nengo_loihi.neurons.LoihiSpikingRectifiedLinear()..')
train(
    params_file="./out/keras_to_loihi_loihineuron_params",
    epochs=10,
    swap_activations={tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()},
    scale_firing_rates=100,
    do_training=True
)


# test the trained networks using spiking neurons
print('-'*80)
print('Evaluating model trained with nengo_loihi.neurons.LoihiSpikingRectifiedLinear()..')
run_network(
    activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    scale_firing_rates=100,
    params_file="./out/keras_to_loihi_loihineuron_params",
    n_steps=50,
    synapse=0.005,
    outfile='loihineuron_training_output'
)

######## now we'll test on Loihi ###########


pres_time = 0.05  # how long to present each input, in seconds
n_test = 5  # how many images to test

# convert the keras model to a nengo network
nengo_converter = nengo_dl.Converter(
    model,
    scale_firing_rates=1000,
    swap_activations={tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()},
    synapse=0.005,
)
net = nengo_converter.net

# get input/output objects
nengo_input = nengo_converter.inputs[inp]
nengo_output = nengo_converter.outputs[dense2]

# build network, load in trained weights, save to network
with nengo_dl.Simulator(net) as nengo_sim:
    nengo_sim.load_params("./out/keras_to_loihi_loihineuron_params")
    nengo_sim.freeze_params(net)

# The input Node needs to altered to generate our test images as output:
with net:
    nengo_input.output = nengo.processes.PresentInput(
        test_images, presentation_time=pres_time
    )

## We specify that the to_spikes layer should run off-chip:
with net:
    nengo_loihi.add_params(net)  # allow on_chip to be set
    net.config[nengo_converter.layers[to_spikes].ensemble].on_chip = False

## the conv layers will be too big
## break them across multiple Loihi cores
with net:
    conv0_shape = conv0_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[conv0].ensemble
    ].block_shape = nengo_loihi.BlockShape((16, 16, 4), conv0_shape)

    conv1_shape = conv1_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[conv1].ensemble
    ].block_shape = nengo_loihi.BlockShape((12, 12, 6), conv1_shape)

    conv2_shape = conv2_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[conv2].ensemble
    ].block_shape = nengo_loihi.BlockShape((8, 8, 16), conv2_shape)

    conv3_shape = conv3_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[conv3].ensemble
    ].block_shape = nengo_loihi.BlockShape((12, 12, 4), conv3_shape)

    conv4_shape = conv4_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[conv4].ensemble
    ].block_shape = nengo_loihi.BlockShape((12, 12, 4), conv4_shape)

    conv5_shape = conv5_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[conv5].ensemble
    ].block_shape = nengo_loihi.BlockShape((12, 12, 4), conv5_shape)

    dense0_shape = dense0_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[dense0].ensemble
    ].block_shape = nengo_loihi.BlockShape((20,), dense0_shape)

    dense1_shape = dense1_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[dense1].ensemble
    ].block_shape = nengo_loihi.BlockShape((20,), dense1_shape)


# set the default interval as a variable
dt = 0.001

# configure the loihi simuilator with an energy probe
loihi_sim = nengo_loihi.Simulator(net, dt=dt)
board = loihi_sim.sims["loihi"].nxsdk_board
probe_cond = PerformanceProbeCondition(
    tStart=1,
    tEnd=int(n_test * pres_time / dt),
    bufferSize=1000,
    binSize=4
)
e_probe = board.probe(ProbeParameter.ENERGY, probe_cond)

probe_cond2 = PerformanceProbeCondition(
    tStart=1,
    tEnd=int(n_test * pres_time / dt),
    bufferSize=1000,
    binSize=4
)
t_probe = board.probe(ProbeParameter.EXECUTION_TIME, probe_cond2)

# build and run the simulator
with loihi_sim:
    # print information about how cores are being utilized
    print("\n".join(loihi_sim.model.utilization_summary()))

    loihi_sim.run(n_test * pres_time)

    # get output (last timestep of each presentation period)
    pres_steps = int(round(pres_time / loihi_sim.dt))
    loihi_output = loihi_sim.data[nengo_output][pres_steps - 1 :: pres_steps]

    # compute the Loihi accuracy
    loihi_predictions = np.argmax(loihi_output, axis=-1)
    correct = 100 * np.mean(loihi_predictions == test_labels[:n_test, 0, 0])
    print("Loihi accuracy: %.2f%%" % correct)

print('\n', '-'*80)
print('Writing probe results to file..')
print(json.dumps(board.energyTimeMonitor.powerProfileStats, cls=NpEncoder, indent=4))
with open("./out/loihi_energy_time_monitor.json", "w") as f:
    json.dump(board.energyTimeMonitor.powerProfileStats, f, cls=NpEncoder, indent=4)

print('-'*80)

## print(dir(board))
## print(dir(board.energyTimeMonitor))
## print(dir(board.energyTimeMonitor.energyProbe))
## print(dir(board.energyTimeMonitor.executionTimeProbe))

# plot the neural activity of the convnet layers
plt.figure(figsize=(12, 4))

timesteps = loihi_sim.trange() / loihi_sim.dt

# plot the presented MNIST digits
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
images = test_images.reshape(-1, 48, 48, 1)[:n_test]
ni, nj, nc = images[0].shape
allimage = np.zeros((ni, nj * n_test, nc), dtype=images.dtype)
for i, image in enumerate(images[:n_test]):
    allimage[:, i * nj : (i + 1) * nj] = image
if allimage.shape[-1] == 1:
    allimage = allimage[:, :, 0]
plt.imshow(allimage, aspect="auto", interpolation="none", cmap="gray")
plt.xticks([])
plt.yticks([])

# plot the network predictions
plt.subplot(2, 1, 2)
plt.plot(timesteps, loihi_sim.data[nengo_output])
plt.legend(classes, loc="lower left",  prop={'size': 6})
plt.suptitle("Output Predictions")
plt.xlabel("Timestep (ms)")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig('./out/loihi_result.png')
