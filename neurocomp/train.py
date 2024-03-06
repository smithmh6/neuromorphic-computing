"""
Trains a VGG-inspired small-scale CNN on CK+ dataset.
"""


# import dependencies
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
## model = tf.keras.models.Sequential([
##
##     tf.keras.layers.Input(shape=IMAGE_DIMS + (1, )),
##
##     tf.keras.layers.Conv2D(
##         filters=12, kernel_size=k_size, padding="same", activation="relu"),
##     tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride),
##
##     tf.keras.layers.Conv2D(
##         filters=24, kernel_size=k_size, padding="same", activation="relu"),
##     tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride),
##
##     tf.keras.layers.Conv2D(
##         filters=32, kernel_size=k_size, padding="same", activation="relu"),
##     tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride),
##
##     tf.keras.layers.Conv2D(
##         filters=48, kernel_size=k_size, padding="same", activation="relu"),
##     tf.keras.layers.BatchNormalization(),
##     tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride),
##
##     tf.keras.layers.Flatten(),
##     tf.keras.layers.Dense(units=48,activation="relu"),
##     tf.keras.layers.Dropout(0.2),
##     tf.keras.layers.Dense(units=24,activation="relu"),
##     tf.keras.layers.Dropout(0.2),
##     tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax")
##
## ])
model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=IMAGE_DIMS + (1, )),

    tf.keras.layers.Conv2D(
        filters=12, kernel_size=k_size, padding="same", activation="relu"),
    tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride),

    tf.keras.layers.Conv2D(
        filters=24, kernel_size=k_size, padding="same", activation="relu"),
    tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride),

    tf.keras.layers.Conv2D(
        filters=32, kernel_size=k_size, padding="same", activation="relu"),
    tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride),

    tf.keras.layers.Conv2D(
        filters=48, kernel_size=k_size, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.AveragePooling2D(pool_size=p_size, strides=stride),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=48,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=24,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax")

])

### # display the model summary
model.summary()

# plot the model architecture
## tf.keras.utils.plot_model(
##     model, to_file='./tests/architecture.png', show_shapes=True, show_dtype=False,
##     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=150,
##     layer_range=None, show_layer_activations=False
## )

# set epochs
NUM_EPOCHS = 150
DECAY_STEPS = (len(train_ds)) * 10
print(f"Decay steps per epoch= {DECAY_STEPS}")

# define learning rate decay
lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.03, decay_steps=DECAY_STEPS, decay_rate=1, staircase=False
)
## boundaries = [40, 80]
## values = [0.01, 0.008, 0.006]
## lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
##     boundaries, values
## )
## step = tf.Variable(initial_value=0, trainable=False)


# plot the learning rate
steps = np.arange(NUM_EPOCHS)
lr = lr_decay(steps)
## for i in range(NUM_EPOCHS):
##     if i < boundaries[0]:
##         lr.append(values[0])
##     elif boundaries[0] <= i < boundaries[1]:
##         lr.append(values[1])
##     elif boundaries[1] <= i:
##         lr.append(values[2])

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
fig.write_image("./tests/learning_rate_decay.png", width=1000)
#fig.show()


# compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr_decay),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

#input("Press ENTER to continue")

# remove old model file and training history
try:
    os.remove('./tests/test_model.h5')
except:
    pass
try:
    os.remove('./tests/test_history.json')
except:
    pass

# add checkpoint callback
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    './tests/test_model.h5',
    monitor='val_loss',
    verbose=1,
    save_best=True,
    save_weights_only=False,
    save_best_only=True,
    mode='min'
)

# add early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=.005,
    patience=10,
    verbose=1,
    mode='auto'
)

# Train the model
history = model.fit(
    train_ds, validation_data=val_ds, epochs=NUM_EPOCHS,
    verbose=2, callbacks=[checkpoint, early_stop])
json.dump(history.history, open('./tests/test_history.json', 'w'))


# plot the results
epochs_range = np.arange(len(history.history['accuracy']))
fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))

fig.add_trace(
    go.Scatter(
        x=epochs_range, y=history.history['accuracy'],
        name='Training', legendgroup='accuracy', mode='lines',  line=dict(color='blue'),
        legendgrouptitle_text='Accuracy'), row=1, col=1)
fig.add_trace(go.Scatter(x=epochs_range, y=history.history['val_accuracy'], mode='lines',  line=dict(color='green'),
    name='Validation', legendgroup='accuracy'), row=1, col=1)

fig.add_trace(
    go.Scatter(x=epochs_range, y=history.history['loss'], line=dict(color='blue'),
    mode='lines', showlegend=False), row=1, col=2)
fig.add_trace(
    go.Scatter(x=epochs_range, y=history.history['val_loss'], line=dict(color='green'),
    mode='lines', showlegend=False), row=1, col=2)


fig.update_layout(
    template='plotly_dark',
    title="Test Results",
    title_x=0.5
    ## xaxis_title="Epochs"
    ## yaxis_title="LR Value",)
)

# save and display the plot image
pio.write_image(fig, "./tests/history.png", scale=6, width=1000, height=600)
#fig.show()


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
for im, l in list(train_ds.as_numpy_iterator())[:2500]: # 4 for JAFFE, 30+ for FER2013
    for i in range(np.shape(im)[0]):
        #print(im[i, :, :, :].shape)
        x_norm.append(im[i, :, :, :])



x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_norm = np.asarray(x_norm)

np.savez_compressed(r'./tests/x_test', x_test)
np.savez_compressed(r'./tests/y_test', y_test)
np.savez_compressed(r'./tests/x_norm', x_norm)


# evaluate the model
print('\n' + "-" * 60 + '\n')
_loss, _acc = model.evaluate(test_ds, verbose=2)
print('\n' + "-" * 60 + '\n')

# create confusion matrix
y_pred = np.argmax(model.predict(x_test, verbose=2), axis=1)
y_true = np.argmax(y_test, axis=1)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred).numpy()
con_mat_norm = np.around(
    confusion_mtx.astype('float') /
    confusion_mtx.sum(axis=1)[:, np.newaxis],
    decimals=3
)

fig = ff.create_annotated_heatmap(
    con_mat_norm, x=[str(x) for x in CLASS_LABELS], y=[str(x) for x in CLASS_LABELS])#, colorscale='deep')
fig.update_layout(
    template='plotly_dark',
    title={'text': "Confusion Matrix", 'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
    xaxis_title="Prediction",
    yaxis_title="Actual"
)
fig['data'][0]['showscale'] = True
fig['layout']['yaxis']['autorange'] = "reversed"
fig['layout']['xaxis']['side'] = 'bottom'
pio.write_image(fig, './tests/confusion_matrix.png', width=1000, height=800, scale=6)
#fig.show()


print("-" * 60)
# convert h5 file to a saved model object
saved_model = tf.keras.models.load_model('./tests/test_model.h5')
tf.saved_model.save(saved_model, 'test_saved_model')


#path of the directory where you want to save your model
frozen_out_path = './'

# model --> graph filename
frozen_graph_filename = "test_inference_graph"

# convert to keras ConcreteFunction object
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# get the concrete function
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]

print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)

try:
    os.remove('./tests/test_model_parsed.h5')
except:
    pass
try:
    os.remove('./tests/test_model_INI.h5')
except:
    pass
try:
    shutil.rmtree('./tests/test_snn')
except:
    pass


# run snn conversion/simulation
print('\n' + "-" * 60 + '\n')
print("Converting ANN --> SNN and running simulation...\n")
snn.main("./snn_config.ini")

print('\n' + "-" * 60 + '\n')
print(f"ANN Loss= {round(_loss, 5)}   Acc.= {round(_acc*100, 4)} %")
print('\n' + "-" * 60 + '\n')

