"""
Helper functions for neural network training/conversion.
"""

import cv2 as cv
import numpy as np
import os
import pathlib
import tensorflow as tf
import time


def op_timer(start):
    """
    input start time, print out elapsed time, restart timer
    """
    now = time.perf_counter()
    print(f"---> elapsed time: {round((now - start)*1000, 5)} ms")
    timer = time.perf_counter()
    return timer

def get_size(arr):
    arr = np.asarray(arr)
    print(f"---> size: {round((arr.size*arr.itemsize)*1e-6, 5)} mb")

def data_pipeline(
    images:np.ndarray, labels:np.ndarray, im_size:tuple,
    edges:bool=False, batch_size:int=1, seed=None, rgb:bool=False,
    drop_rem:bool=False, flip:bool=False, shuffle:bool=False,
    haar:bool=False) -> tf.data.Dataset:
    """"""

    t = time.perf_counter()
    start = time.perf_counter()

    print('-'*80)
    print("\nDataset Info:")
    get_size(images)

    print(f"\n---> dimensions: {np.shape(images[0])}")
    print(f"---> samples: {np.shape(images)[0]}")

    n_batches = np.shape(images)[0] // batch_size
    print(f"---> batches: {n_batches}")

    if haar:
        print("\nExtracting faces from image data...")
        face_haar_cascade = cv.CascadeClassifier('/Users/heathsmith/repos/github/neuromorphic-computing/openvino/haarcascade_frontalface_defaults.xml')
        extracted_faces, extracted_labels = [], []
        # iterate images and labels
        for im, lbl in zip(images, labels):
            # detect any faces present
            #im = cv.resize(im, (256, 256))
            faces_detected = face_haar_cascade.detectMultiScale(im)
            # iterate through faces detected
            for (x,y,w,h) in faces_detected:
                try:
                    roi_gray = im[y:y + h, x:x + w]
                    image_pixels = cv.resize(roi_gray, im_size)
                    # append to extracted list
                    extracted_faces.append(image_pixels)
                    extracted_labels.append(lbl)
                except:
                    pass
        print(f"\nExtracted {len(extracted_faces)} faces from {len(images)} images.")
        t = op_timer(t)

        # overwrite dataset with face extracted images
        images, labels = extracted_faces, extracted_labels

    else:
        # reshape images if image dims not equal to input dims
        if im_size != images[0].shape:
            print("\nResizing image data...")
            images = list(map(lambda x: (cv.resize(x, im_size)), images))
            t = op_timer(t)

    # canny edge detection if requested
    # this must occur on 8-bit ints
    if edges:
        # upsample
        print("\nUpsampling image data...")
        factor = 2
        upscale = (im_size[0]*factor, im_size[1]*factor)
        images = list(map(lambda x: (cv.resize(x, upscale)), images))
        t = op_timer(t)

        # gaussian blur
        print("\nGaussian blurring..")
        images = list(map(lambda x: cv.GaussianBlur(x, (3, 3), sigmaX=0, sigmaY=0), images))
        t = op_timer(t)

        # convert to absolute scale
        print("\nConverting to absolute scale..")
        images = list(map(lambda x: cv.convertScaleAbs(x), images))
        t = op_timer(t)

        # canny edge detection
        print("\nExecuting canny edge detection..")
        images = list(map(
            lambda x: cv.Canny(x, threshold1=50, threshold2=150), images))
        t = op_timer(t)

        # downsample
        print("\nRestoring image dimensions...")
        images = list(map(lambda x: (cv.resize(x, im_size)), images))
        t = op_timer(t)

    # set the correct image dimensions
    if rgb == True:
        print("\nConverting to RGB images... ")
        images = list(map(lambda x: cv.cvtColor(x, cv.COLOR_GRAY2RGB), images))
        t = op_timer(t)
    else:
        print("\nExpanding image dimensions...")
        images = list(map(lambda x: np.expand_dims(x, axis=2), images))
        t = op_timer(t)

    print("\nDataset size after reshaping:")
    get_size(images)

    print("\nBatching image data...")
    if batch_size > 1:
        batched_images, batched_labels = [], []
        a = 0
        b = batch_size
        for batch in range(0, n_batches):
            batched_images.append(images[a:b])
            batched_labels.append(labels[a:b])
            a += batch_size
            b += batch_size
        print("\nBatch shapes: ")
        print(np.shape(batched_images), np.shape(batched_labels))
        t = op_timer(t)
        images, labels = batched_images, batched_labels

    # create dataset object and optimize performance
    print("\nCreating dataset object..")
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    print(f"\nSetting batch size to {batch_size}")
    ds = ds.batch(batch_size=batch_size)
    t = op_timer(t)

    print("\nConfiguring prefetching...")
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    t = op_timer(t)

    # scale the images
    print("\nScaling images...")
    ds = ds.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    t = op_timer(t)

    if flip:
        print("\nAugmenting with random L/R flip...")
        with tf.device('/CPU:0'):
            ds = ds.map(
                lambda x,y: (tf.image.random_flip_left_right(x, seed=seed), y),
                num_parallel_calls=tf.data.AUTOTUNE)
        t = op_timer(t)

    if shuffle:
        print("\nShuffling dataset..")
        ds = ds.shuffle(len(images) * batch_size, seed=seed)
        t = op_timer(t)

    print("\nConfiguring caching...")
    ds = ds.cache()
    t = op_timer(t)


    print(f"\nTotal time ===> {(t - start)} s")
    print("-"*80)

    # return the dataset object
    #return ds
    return images, labels


if __name__ == "__main__":
    print('No functions active.')

    # run compress_dataset()
    ## compress_dataset(
    ##     pathlib.Path(r"/Users/heathsmith/repos/github/neuromorphic-computing/datasets/FER-2013"),
    ##     pathlib.Path(r"/Users/heathsmith/repos/github/neuromorphic-computing/datasets/fer_2013"))

