"""
Helper functions for neural network training/conversion.
"""

import cv2 as cv
import numpy as np
import os
import pathlib
import tensorflow as tf

def create_JAFFE():
    """
    Organize the JAFFE image dataset.
    """

    ds = pathlib.Path(r'/Users/heathsmith/repos/github/neuromorphic-computing/datasets/JAFFE/')

    # parse the text file
    with open(os.path.join(ds, 'A_README_FIRST.txt'), 'r') as f:

        # jump to classes list
        f.seek(11572) #  2112
        classes1 = f.readline().strip().strip('\n').split(' ')

        # jump to semantic ratings section
        f.seek(11652) #  2204

        # store image ratings in dict
        imgs = {}
        for i in range(187):  #  219
            row = f.readline().strip('\n').split(' ')
            imgs[row[6]] = (row[0], classes1[row[1:6].index(max(row[1:6]))])

        # copy images to class folders
        for k, v in imgs.items():
            fname = str(k).replace('-', '.')

            try:
                im = cv.imread(os.path.join(ds, f"{fname}.{v[0]}.tiff"))  ##, -1)
                    # 'classes' for first set or 'classes_nofear' for second set
                cv.imwrite(os.path.join(ds, 'classes_rgb_nofear', v[1], f"{k}.png"), im)

            except Exception as e:
                #print(os.path.join(ds, f"{fname}.{v[0]}.tiff"), 'not found.')
                print(e)
                pass

def compress_dataset(ds_path:pathlib.Path, out_path:pathlib.Path):
    """
    Compress dataset into numpy .npz file.
    """

    # get classes
    classes = [str(x) for x in os.listdir(ds_path) if not 'DS_' in x]

    # define images and labels
    images = []
    labels = []

    for i, cl in enumerate(classes):
        label = tf.keras.utils.to_categorical(i, len(classes))
        for f in os.listdir(os.path.join(ds_path, cl)):
            try:

                im = cv.imread(os.path.join(ds_path, cl, f), -1)
                if len(im.shape) > 2 and im.shape[2] == 3:
                    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

                images.append(im)
                labels.append(label)

                print(label, im.shape, type(im), im.dtype, np.max(im), np.min(im))

            except:
                print(f"Unable to read {os.path.join(cl, f)}")

    np.savez_compressed(
        os.path.join(ds_path, out_path), images=images, labels=labels, classes=classes)

def data_pipeline(
    images:np.ndarray, labels:np.ndarray, im_size:tuple,
    edges:bool=False, batch_size:int=32, seed=None,
    drop_rem:bool=False, flip:bool=False, shuffle:bool=False) -> tf.data.Dataset:
    """"""

    # modified copy of images and labels
    _images = []

    # reshape images if image dims not equal to input dims
    if im_size != images[0].shape:
        _images = list(map(lambda x: cv.resize(x, im_size), images))
    else:
        _images = images


    # canny edge detection if requested
    if edges:

        # run a gaussian blur on the images
        _images = list(map(lambda x: cv.GaussianBlur(x, (3, 3), sigmaX=0, sigmaY=0), _images))

        # convert to absolute scale
        _images = list(map(lambda x: cv.convertScaleAbs(x), _images))

        # canny edge detection
        _images = list(map(
            lambda x: cv.Canny(x, threshold1=50, threshold2=150), _images))

    # scale the images
    _images = list(map(lambda x: np.array(x/255.0).astype(np.float32), _images))

    # downsample to 48 x 48
    #_images = list(map(lambda x: cv.resize(x, (48, 48)), _images))

    # expand dimensions to H x W x Depth
    _images = list(map(
        lambda x: np.expand_dims(x, axis=2), _images
    ))
    ## _images = np.asarray(list(map(lambda x: cv.cvtColor(x, cv.COLOR_GRAY2RGB), _images)))

    if flip:
        with tf.device('/CPU:0'):
            _images = tf.image.random_flip_left_right(_images, seed=seed).numpy()
    ## else:
    ##     # convert to numpy
    ##     _images = np.asarray(_images)

    # create dataset object and optimize performance
    ds = tf.data.Dataset.from_tensor_slices((_images, labels))
    if shuffle:
        ds = ds.shuffle(len(_images) * batch_size, seed=seed)
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    ds = ds.batch(
        batch_size=batch_size,
        #num_parallel_calls=tf.data.AUTOTUNE,
        drop_remainder=drop_rem)

    # return the dataset object
    return ds

if __name__ == "__main__":
    print('No functions active.')

    # run compress_dataset()
    ## compress_dataset(
    ##     pathlib.Path(r"/Users/heathsmith/repos/github/neuromorphic-computing/datasets/FER-2013"),
    ##     pathlib.Path(r"/Users/heathsmith/repos/github/neuromorphic-computing/datasets/fer_2013"))

