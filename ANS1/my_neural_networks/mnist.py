"""© 2018 Jianfei Gao All Rights Reserved

- Original Version
    
    Author: I-Ta Lee
    Date: 10/1/2017

- Modified Version

    Author: Jianfei Gao
    Date: 08/28/2018

"""
import os
import gzip
from struct import unpack
import numpy as np
import logging


TRAIN_IMAGE_FILE_NAME='train-images-idx3-ubyte.gz'
TRAIN_LABEL_FILE_NAME='train-labels-idx1-ubyte.gz'
TEST_IMAGE_FILE_NAME='t10k-images-idx3-ubyte.gz'
TEST_LABEL_FILE_NAME='t10k-labels-idx1-ubyte.gz'
N_CLASSES = 10


def _load_data(image_fpath, label_fpath, max_n_examples=-1):
    logging.info('load {}...'.format(image_fpath))
    X = read_images(image_fpath, max_n_examples=max_n_examples)
    logging.info('load {}...'.format(label_fpath))
    y = read_labels(label_fpath, max_n_examples=max_n_examples)
    return X, y


def load_train_data(folder,
                    train_image_fname=TRAIN_IMAGE_FILE_NAME,
                    train_label_fname=TRAIN_LABEL_FILE_NAME,
                    max_n_examples=-1):
    image_fpath = os.path.join(folder, train_image_fname)
    label_fpath = os.path.join(folder, train_label_fname)
    return _load_data(image_fpath, label_fpath, max_n_examples)


def load_test_data(folder,
                   test_image_fname=TEST_IMAGE_FILE_NAME,
                   test_label_fname=TEST_LABEL_FILE_NAME,
                   max_n_examples=-1):
    image_fpath = os.path.join(folder, test_image_fname)
    label_fpath = os.path.join(folder, test_label_fname)
    return _load_data(image_fpath, label_fpath, max_n_examples)


def read_images(fpath, max_n_examples=-1):
    with gzip.open(fpath, 'rb') as images:
        images.read(4)
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]
        if max_n_examples != -1:
            number_of_images = max_n_examples

        x = np.zeros((number_of_images, rows, cols), dtype=np.uint8)
        for i in range(number_of_images):
            for row in range(rows):
                for col in range(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    x[i][row][col] = tmp_pixel
    return x


def read_labels(fpath, n_classes=10, max_n_examples=-1):
    with gzip.open(fpath, 'rb') as labels:
        labels.read(4)
        number_of_labels = labels.read(4)
        number_of_labels = unpack('>I', number_of_labels)[0]
        if max_n_examples != -1:
            number_of_labels = max_n_examples
        
        y = np.zeros((number_of_labels, 1), dtype=np.int64)
        for i in range(number_of_labels):
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]
    return y
