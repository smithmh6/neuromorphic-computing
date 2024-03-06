"""
This module contains a script that trains
a pre-trained VGG-16 model for facial
expression recognition.
"""
# import dependencies
import argparse as ap
import ast
import configparser as cp
import json
import numpy as np
import os
import pathlib
import time
import shutil
from tqdm import tqdm
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import tensorflow as tf
import models
import utils
#from openvino.inference_engine import IECore

# create the arg parser
def get_args():

    # parse command line args
    print('\n', '-'*80, '\n')
    print("\nParsing command line args...")

    parser = ap.ArgumentParser(
        prog="train.py",
        description="Train a convolutional neural network."
    )
    parser.add_argument("-c", "--config", type=pathlib.Path)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--pretrain", type=str)

    return parser.parse_args()


def main(args):
    """
    Main training script.
    """

    # start timer
    start = time.perf_counter()

    # parse config
    print("\nReading config from path ---> ", args.config)
    cfg = cp.ConfigParser()
    cfg.read(args.config)

    # read compressed dataset
    print(f"\nLoading dataset ---> ", args.dataset)
    ds_path = os.path.join(cfg['data']['ds_path'], args.dataset + '.npz')
    with np.load(ds_path) as ds:
        images, labels, classes = (ds['images'], ds['labels'], ds['classes'])

    print(f"\nFound {images.shape[0]} examples in {labels.shape[1]} classes.")
    print(f"Shape= {(images.shape[1], images.shape[2])}")
    print(f"Class Names= {classes}")

    # determine image shape
    im_dims = ast.literal_eval(cfg.get('processing', 'img_dims'))
    im_color = cfg.getboolean('processing', 'rgb')
    im_depth = (3,) if im_color == True else (1,)

    # get model params as dict
    model_params = dict(cfg['model'])
    if args.pretrain: model_params['pretrain'] = args.pretrain

    # construct the model
    model = models.get_model(
        im_dims + im_depth, # input shape
        len(classes),       # number of classes
        args.model,         # model names
        **model_params
    )

    #train = input("Continue? (y/n)")
    #if train == 'n':
    #    sys.exit()

    # plot the model architecture
    print("\nConfiguring output directory...")
    out_path = os.path.join(cfg['output']['path'], args.output)
    if not os.path.exists(out_path): os.mkdir(out_path)
    else: shutil.rmtree(out_path, ignore_errors=True), os.mkdir(out_path)

    if cfg.get('output', 'plot_model', fallback=False) == True:
        print("\nPlotting the model architecture.")
        tf.keras.utils.plot_model(
            model, to_file=os.path.join(out_path, 'architecture.png'),
            show_shapes=True, show_dtype=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=150,
            layer_range=None, show_layer_activations=True)

    # configure learning rate
    print("\nConfiguring learning rate...")
    learning_rate = cfg.getfloat('training', 'learning_rate')

    if cfg.get('training', 'decay_rate', fallback=None):
        DECAY_STEPS = (len(x_train)) * 10
        print(f"\nLearning Rate Decay steps per epoch= {DECAY_STEPS}")

        # define learning rate decay
        learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
            cfg.getint('training', 'learning_rate'),
            decay_steps=cfg.getint('training', 'decay_steps'),
            decay_rate=cfg.getint('training', 'decay_rate'),
            staircase=cfg.getboolean('training', 'staircase')
        )

        # plot the learning rate
        print("\nPlotting the learning rate decay.")
        step = np.linspace(0, cfg.getint('training', 'epochs'))
        lr = learning_rate(step)

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
        fig.write_image(
            os.path.join(out_path, "learning_rate_decay.png"), width=1000)


    # compile the model
    print("\nCompiling the model.")
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # remove old model file and training history
    print("\nCleaning up resources.")
    model_path = os.path.join(out_path, 'model.h5')
    history_path = os.path.join(out_path, 'history.json')

    if os.path.exists(model_path): os.remove(model_path)
    if os.path.exists(history_path): os.remove(history_path)


    print("\nSetting up model checkpoints.")
    # add checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    model_path,
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




    # create a dataset object and shuffle
    print("\nShuffling dataset...")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(
        len(images) * cfg.getint('data', 'batch_size'),
        seed=cfg.getint('data', 'seed'))
    shuffle_images = []
    shuffle_labels = []
    for im, l in dataset.as_numpy_iterator():
        shuffle_images.append(im)
        shuffle_labels.append(l)
    shuffle_images = np.asarray(shuffle_images)
    shuffle_labels = np.asarray(shuffle_labels)



    # create the slicing indices and set image dims
    TRAIN = round(len(dataset) * cfg.getfloat('data', 'train'))
    VAL = TRAIN + round(len(dataset) *  cfg.getfloat('data', 'val'))
    TEST = VAL + round(len(dataset) *  cfg.getfloat('data', 'test'))
    img_dims = ast.literal_eval(cfg.get('processing', 'img_dims'))

    print(f"\nCreating training dataset...")
    x_train, y_train = utils.data_pipeline(
        shuffle_images[0:TRAIN], shuffle_labels[0:TRAIN], img_dims,
        rgb=cfg.getboolean('processing', 'rgb'),
        edges=cfg.getboolean('processing', 'edges'),
        batch_size=cfg.getint('data', 'batch_size'),
        flip=cfg.getboolean('processing', 'flip'),
        haar=cfg.getboolean('processing', 'haar'))

    print(f"\nCreating validation dataset...")
    x_val, y_val = utils.data_pipeline(
        shuffle_images[TRAIN:VAL], shuffle_labels[TRAIN:VAL], img_dims,
        rgb=cfg.getboolean('processing', 'rgb'),
        edges=cfg.getboolean('processing', 'edges'),
        batch_size=cfg.getint('data', 'batch_size'),
        flip=False,
        haar=cfg.getboolean('processing', 'haar'))

    print(f"\nCreating testing dataset...")
    x_test, y_test = utils.data_pipeline(
        shuffle_images[VAL:TEST], shuffle_labels[VAL:TEST], img_dims,
        rgb=cfg.getboolean('processing', 'rgb'),
        edges=cfg.getboolean('processing', 'edges'),
        batch_size=cfg.getint('data', 'batch_size'),
        flip=False,
        haar=cfg.getboolean('processing', 'haar'))

    print(f"\nUsing {len(x_train)*cfg.getint('data', 'batch_size')} samples for training.")
    print(f"Using {len(x_val)*cfg.getint('data', 'batch_size')} samples for validation.")
    print(f"Using {len(x_test)*cfg.getint('data', 'batch_size')} samples for testing.")

    # Train the model
    print("\nBeginning model training..")
    n_epochs = cfg.getint('training', 'epochs')
    n_train = np.shape(x_train)[0]
    n_val = np.shape(x_val)[0]

    history = {
        'loss':[],
        'accuracy':[],
        'val_loss':[],
        'val_accuracy':[]
    }


    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1} of {n_epochs} ...")
        with tqdm(total=(n_train + n_val)) as pbar:
            for batch in range(0, n_train):
                train_out = model.train_on_batch(
                        np.asarray(x_train[batch]),
                        np.asarray(y_train[batch]),
                        reset_metrics=False,
                        return_dict=True)
                pbar.update(1)
            for batch in range(0, n_val):
                val_out = model.test_on_batch(
                            np.asarray(x_val[batch]),
                            np.asarray(y_val[batch]),
                            reset_metrics=False,
                            return_dict=True)
                pbar.update(1)
        history['loss'].append(train_out['loss'])
        history['accuracy'].append(train_out['accuracy'])
        history['val_loss'].append(val_out['loss'])
        history['val_accuracy'].append(val_out['accuracy'])
        print(f"training ---> {train_out}")
        print(f"validation ---> {val_out}")


    ## train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ## train_ds = train_ds.batch(16)
    ## val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ## val_ds = val_ds.batch(16)
    ## hist = model.fit(
    ##                 train_ds,
    ##                 validation_data=val_ds,
    ##                 epochs=n_epochs,
    ##                 verbose=1,
    ##                 callbacks=[checkpoint, early_stop])
    ## history = hist.history

    print("\nSaving model to file...")
    model_path = os.path.join(out_path, 'model.h5')
    model.save(model_path)
    json.dump(history, open(history_path, 'w'))


    # plot the results
    print("\nPlotting the training history.")
    epochs_range = np.arange(len(history['accuracy']))
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))

    fig.add_trace(
        go.Scatter(
            x=epochs_range, y=history['accuracy'],
            name='Training', legendgroup='accuracy', mode='lines',  line=dict(color='blue'),
            legendgrouptitle_text='Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs_range, y=history['val_accuracy'], mode='lines',  line=dict(color='green'),
        name='Validation', legendgroup='accuracy'), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=epochs_range, y=history['loss'], line=dict(color='blue'),
        mode='lines', showlegend=False), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=epochs_range, y=history['val_loss'], line=dict(color='green'),
        mode='lines', showlegend=False), row=1, col=2)

    fig.update_layout(
        template='plotly',
        title="Training History",
        title_x=0.5
    )

    # save and display the plot image
    history_img = os.path.join(out_path, 'history.png')
    if os.path.exists(history_img): os.remove(history_img)
    pio.write_image(fig, history_img, scale=6, width=1000, height=600)


    # extract x/y test data for simulation
    x_test_samples = []
    y_test_samples = []

    # create the dataset for SNN simulation
    print("\nCompressing the test data...")
    x_test_path = os.path.join(out_path, 'x_test')
    y_test_path = os.path.join(out_path, 'y_test')

    if os.path.exists(x_test_path): os.remove(x_test_path)
    if os.path.exists(y_test_path): os.remove(y_test_path)

    for im, l in zip(x_test, y_test):
        for i in range(np.shape(im)[0]):
            x_test_samples.append(im[i])
            y_test_samples.append(l[i])
    x_test_samples = np.asarray(x_test_samples)
    y_test_samples = np.asarray(y_test_samples)
    print(np.shape(x_test_samples), np.shape(y_test_samples))

    np.savez_compressed(x_test_path, x_test_samples)
    np.savez_compressed(y_test_path, y_test_samples)

    # evaluate the model
    print("\nEvaluating the model.")
    model.load_weights(model_path)
    _loss, _acc = model.evaluate(x=x_test_samples, y=y_test_samples, verbose=1)
    print(f"-----> Loss= {round(_loss, 4)}  Acc={round(_acc*100, 3)} %")

    # create confusion matrix
    print("\nGenerating the confusion matrix.")
    y_pred = np.argmax(model.predict(x_test_samples, verbose=1), axis=1)
    y_true = np.argmax(y_test_samples, axis=1)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred).numpy()
    con_mat_norm = np.around(
        confusion_mtx.astype('float') /
        confusion_mtx.sum(axis=1)[:, np.newaxis],
        decimals=3
    )

    fig = ff.create_annotated_heatmap(
            con_mat_norm,
            x=[str(x) for x in classes],
            y=[str(x) for x in classes])
    fig.update_layout(
            template='plotly',
            title={
                'text': "Confusion Matrix",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Prediction",
            yaxis_title="Actual")
    fig['data'][0]['showscale'] = True
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['xaxis']['side'] = 'bottom'
    pio.write_image(
            fig, os.path.join(out_path, 'confusion_matrix.png'),
            width=1000, height=800, scale=6)


    # convert h5 file to a saved model object
    print("\nConverting to SavedModel object...")
    saved_path = os.path.join(out_path, 'saved_model')
    saved_model = tf.keras.models.load_model(model_path)
    tf.saved_model.save(saved_model, saved_path)


    # convert to keras ConcreteFunction object
    print("\nConverting to Concrete Function object...")
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    print("\nFrozen model layers: ")
    layers = [op.name for op in frozen_func.graph.get_operations()]
    for layer in layers:
        print("  --->", layer)
    print("Frozen model inputs: \n", frozen_func.inputs)
    print("Frozen model outputs: \n", frozen_func.outputs)

    # Save frozen graph to disk
    print("\nSaving frozen graph to disk...")
    frozen_graph_filename = "inference_graph"
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=out_path,
                    name=f"{frozen_graph_filename}.pb",
                    as_text=False)


    # The paths of the source and converted models
    inference_path = os.path.join(out_path, "inference_graph.pb")

    # Construct the command for Model Optimizer
    #mo_command = f"""python3 /opt/intel/openvino_2021.4.752/deployment_tools/model_optimizer/mo_tf.py
    mo_command = f"""mo --input_model "{inference_path}" --input_shape "[1, 48, 48, 1]" --data_type FP16 --output_dir "{out_path}" """
    mo_command = " ".join(mo_command.split())
    print('-'*80)
    print('\nRunning Model Optimizer...')
    print(f'{mo_command}')
    #os.system(str(mo_command))
    print('-'*80)

    # calculate elapsed time
    print(f"\n-----> Loss= {round(_loss, 4)}  Acc={round(_acc*100, 3)} %")
    end = time.perf_counter()
    elapsed = round((end - start) / 60.0, 3)
    print(f"\nElapsed Time= {elapsed} minutes")



    print('-'*80)


if __name__ == "__main__":

    # execute main function
    args = get_args()
    main(args)

    #ds_path = '/Users/heathsmith/repos/neuromorphic-computing/datasets/fer_2013.npz'
    #out_dir = '/Users/heathsmith/repos/neuromorphic-computing/openvino/out/vgg_mini_jaffe_48_haar_flip_edges'
    #convert(ds_path, out_dir)

    #optimize(out_dir)
