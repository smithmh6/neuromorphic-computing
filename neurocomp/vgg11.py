"""
This module contains a script that trains
a pre-trained VGG-11 model for facial
expression recognition.
"""
# import dependencies
import argparse as ap
import json
import numpy as np
import os
import pathlib
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import tensorflow as tf
import utils
import time

# create the arg parser
ARG_PARSER = ap.ArgumentParser(
    prog="train.py",
    description="Train a convolutional neural network."
)
ARG_PARSER.add_argument("-d", "--data", type=int)
ARG_PARSER.add_argument("-n", "--name", type=str)

def main():
    """
    Main training script.
    """

    # start timer
    START_TIME = time.perf_counter()

    # get cmd line args
    args = ARG_PARSER.parse_args()

    if not os.path.exists(f"./models/vgg11/{args.name}"):
        os.mkdir(f"./models/vgg11/{args.name}")

    OUTPATH = f"./models/vgg11/{args.name}"
    print(f"Writing output to {OUTPATH}")


    # read compressed dataset
    DS_OPTS = ('ck_plus_48', 'fer_2013', 'jaffe_nofear', 'jaffe')
    data_path = pathlib.Path(f"../datasets/{DS_OPTS[args.data]}.npz")
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
    BATCH_SIZE = 32
    IMAGE_DIMS = (48, 48)
    SEED = 123
    _TRAIN = .8
    _VAL = .1
    _TEST = 1 - (_TRAIN + _VAL)

    EDGE = False

    # create a dataset object and shuffle
    print("Preprocessing datasets.")
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
        edges=EDGE, batch_size=BATCH_SIZE, flip=True, shuffle=True)
    val_ds = utils.data_pipeline(
        shuffle_images[TRAIN:VAL], shuffle_labels[TRAIN:VAL], IMAGE_DIMS,
        edges=EDGE, batch_size=BATCH_SIZE, flip=False, shuffle=False)
    test_ds = utils.data_pipeline(
        shuffle_images[VAL:TEST], shuffle_labels[VAL:TEST], IMAGE_DIMS,
        edges=EDGE, batch_size=BATCH_SIZE, flip=False, shuffle=False)
    print(f"Using {len(train_ds)*BATCH_SIZE} samples for training.")
    print(f"Using {len(val_ds)*BATCH_SIZE} samples for validation.")
    print(f"Using {len(test_ds)*BATCH_SIZE} samples for testing.")

    # construct the model
    print("Building VGG-11 model.")
    k_size = (7, 7)
    p_size = (2, 2)
    stride = (2, 2)
    model = tf.keras.models.Sequential([

        # Convolutional Block 1
        tf.keras.layers.Conv2D(
            input_shape=IMAGE_DIMS + (1, ), filters=32, kernel_size=k_size,padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=p_size,strides=stride),

        # Convolutional Block 2
        tf.keras.layers.Conv2D(filters=64, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=p_size,strides=stride),

        # convolutional Block 3
        tf.keras.layers.Conv2D(filters=128, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=p_size,strides=stride),

        # Convolutional Block 4
        tf.keras.layers.Conv2D(filters=256, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=p_size,strides=stride),

        # Convolutional Block 5
        tf.keras.layers.Conv2D(filters=512, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=k_size, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=p_size,strides=stride),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4096,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=4096,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax")
    ])


    ### # display the model summary
    print(model.summary())

    # plot the model architecture
    print("Plotting the model architecture.")
    tf.keras.utils.plot_model(
        model, to_file=os.path.join(OUTPATH, 'architecture.png'),
        show_shapes=True, show_dtype=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=150,
        layer_range=None, show_layer_activations=True
    )

    # set epochs
    NUM_EPOCHS = 50
    DECAY_STEPS = (len(train_ds)) * 4
    DECAY_RATE = 100
    LEARNING_RATE = 0.01
    print(f"Learning Rate Decay steps per epoch= {DECAY_STEPS}")

    # define learning rate decay
    lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
        LEARNING_RATE, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE, staircase=False
    )

    # plot the learning rate
    print("Plotting the learning rate decay.")
    step = np.linspace(0, NUM_EPOCHS)
    lr = lr_decay(step)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step, y=lr,
                        mode='lines',
                        name='lines'))
    fig.update_layout(
        template='plotly',
        title={'text': "Learning Rate Decay", 'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        xaxis_title="Epochs",
        yaxis_title="LR Value",)

    # save and display the plot image
    fig.write_image(os.path.join(OUTPATH, "learning_rate_decay.png"), width=1000)


    # compile the model
    print("Compiling the model.")
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_decay),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # remove old model file and training history
    print("Cleaning up resources.")
    try:
        os.remove(os.path.join(OUTPATH, 'model.h5'))
    except:
        pass
    try:
        os.path.join(OUTPATH, "history.json")
    except:
        pass


    print("Setting up model checkpoints.")
    # add checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPATH, 'model.h5'),
        monitor='val_accuracy',
        verbose=2,
        save_best=True,
        save_weights_only=False,
        mode='max'
    )

    # add early stopping callback
    print("Configuring early stopping parameters.")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=.001,
        patience=5,
        verbose=2,
        mode='auto'
    )

    # Train the model
    print("Beginning model training.")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=NUM_EPOCHS,
        verbose=1, callbacks=[checkpoint, early_stop])
    json.dump(history.history, open(os.path.join(OUTPATH, 'history.json'), 'w'))


    # plot the results
    print("Plotting the training history.")
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
        template='plotly',
        title="Training History",
        title_x=0.5
    )

    # save and display the plot image
    pio.write_image(fig, os.path.join(OUTPATH, "history.png"), scale=6, width=1000, height=600)


    # store the data for SNN conversion/simulation
    x_test = []
    y_test = []
    x_norm = []

    # create the dataset for SNN simulation
    print("Compressing data for SNN conversion.")
    for im, l in test_ds.as_numpy_iterator():
        for i in range(np.shape(im)[0]):
            x_test.append(im[i, :, :, :])
            y_test.append(l[i, :])

    # create a dataset for normalization
    MAX_SAMPLES = 2500
    print(f"Extracting {MAX_SAMPLES} samples for normalization.")
    for im, l in train_ds.as_numpy_iterator():
        for i in range(np.shape(im)[0]):
            if len(x_norm) < MAX_SAMPLES:
                x_norm.append(im[i, :, :, :])

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x_norm = np.asarray(x_norm)

    if not os.path.exists(os.path.join(OUTPATH, 'data')):
        os.mkdir(os.path.join(OUTPATH, 'data'))
    np.savez_compressed(os.path.join(OUTPATH, 'data/x_test'), x_test)
    np.savez_compressed(os.path.join(OUTPATH, 'data/y_test'), y_test)
    np.savez_compressed(os.path.join(OUTPATH, 'data/x_norm'), x_norm)

    # evaluate the model
    print("Evaluating the model.")
    _loss, _acc = model.evaluate(test_ds, verbose=1)
    print(f"-----> Loss= {round(_loss, 4)}  Acc={round(_acc, 3)} %")

    # create confusion matrix
    print("Creating confusion matrix.")
    y_pred = np.argmax(model.predict(x_test, verbose=1), axis=1)
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
        template='plotly',
        title={'text': "Confusion Matrix", 'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        xaxis_title="Prediction",
        yaxis_title="Actual"
    )
    fig['data'][0]['showscale'] = True
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['xaxis']['side'] = 'bottom'
    pio.write_image(fig, os.path.join(OUTPATH, 'confusion_matrix.png'), width=1000, height=800, scale=6)

    # write parameters to file
    with open(os.path.join(OUTPATH, 'info.csv'), 'w') as f:
        f.write("ds_name,acc,loss,classes,batch_size,image_dims,seed,train,val,test,edges,learning_rate,decay_steps,decay_rate\n")
        f.write(
            f"{DS_OPTS[args.data]},{round(_acc, 4)},{round(_loss, 4)},{NUM_CLASSES},{BATCH_SIZE},{IMAGE_DIMS[0]}x{IMAGE_DIMS[1]},"\
            f"{SEED},{_TRAIN},{_VAL},{round(_TEST, 1)},{EDGE},{LEARNING_RATE},{DECAY_STEPS},{DECAY_RATE}")

    # calculate elapsed time
    END_TIME = time.perf_counter()
    ELAPSED = round((END_TIME - START_TIME) / 60.0, 3)

    print(f"Elapsed Time= {ELAPSED} minutes")


if __name__ == "__main__":

    # execute main function
    main()

