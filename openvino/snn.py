"""
Run snntoolbox.
"""

import os
import shutil
from snntoolbox.bin.run import main
import numpy as np
import tensorflow as tf
import utils
import ast

if __name__ == "__main__":

    ##try:
    ##    os.remove("./models/vgg16/jaffe_nofear_run01/model_parsed.h5")
    ##except:
    ##    pass
    ##try:
    ##    os.remove("./models/vgg16/jaffe_nofear_run01/model_INI.h5")
    ##except:
    ##    pass
    ##try:
    ##    shutil.rmtree('./models/vgg16/jaffe_nofear_run01/log/gui/snn01')
    ##except:
    ##    pass

    # read compressed dataset
    ds_path = '/Users/heathsmith/repos/github/neuromorphic-computing/datasets/ck_plus_48_v2.npz'
    with np.load(ds_path) as ds:
        images, labels, classes = (ds['images'], ds['labels'], ds['classes'])

    # create a dataset object and shuffle
    print("\nShuffling dataset...")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(
        len(images) * 128,
        seed=123)
    shuffle_images = []
    shuffle_labels = []
    for im, l in dataset.as_numpy_iterator():
        shuffle_images.append(im)
        shuffle_labels.append(l)
    shuffle_images = np.asarray(shuffle_images)
    shuffle_labels = np.asarray(shuffle_labels)



    # create the slicing indices and set image dims
    TRAIN = round(len(dataset) * 1.0)


    print(f"\nCreating training dataset...")
    x_train, y_train = utils.data_pipeline(
        shuffle_images[0:TRAIN], shuffle_labels[0:TRAIN], (48,48),
        rgb=False,
        edges=False,
        batch_size=16,
        flip=True,
        haar=True)

    # create a dataset for normalization
    model_name = 'vgg_mini_jaffe_48_haar_flip_edges'
    x_norm = []
    for im in x_train: # 4 for JAFFE, 30+ for FER2013
        for i in range(np.shape(im)[0]):
            x_norm.append(im[i])

    x_norm = np.asarray(x_norm)

    np.savez_compressed(f'./out/{model_name}/x_norm', x_norm)


    model = f'./out/{model_name}'

    main(f"./snn_config.ini")

