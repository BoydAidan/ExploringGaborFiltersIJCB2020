from __future__ import division

import sys

import numpy as np
from mpl_toolkits import mplot3d


from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, concatenate, Reshape
from keras.models import Model, Sequential
from keras.regularizers import l2, l1
from keras import backend as K
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from imutils import paths
import cv2 as cv
import numpy.random as rng
from tqdm import tqdm
import tensorflow as tf
sys.path.insert(0, '../Custom_Layers/')
from bitsample_layer import BitSampling
from matplotlib import pyplot

def convert_to_osiris(all_weights, zero):
    print "Creating filters file"
    fifteen1 = all_weights[0][:, :, :, 0]
    fifteen2 = all_weights[0][:, :, :, 1]

    twentyseven1 = all_weights[2][:, :, :, 0]
    twentyseven2 = all_weights[2][:, :, :, 1]

    fiftyone1 = all_weights[4][:, :, :, 0]
    fiftyone2 = all_weights[4][:, :, :, 1]

    filter_set = {}
    filter_set["fifteen1"] = np.squeeze(fifteen1)
    filter_set["fifteen2"] = np.squeeze(fifteen2)
    filter_set["twentyseven1"] = np.squeeze(twentyseven1)
    filter_set["twentyseven2"] = np.squeeze(twentyseven2)
    filter_set["fiftyone1"] = np.squeeze(fiftyone1)
    filter_set["fiftyone2"] = np.squeeze(fiftyone2)

    averages = {}
    if zero:
        averages["fifteen1"] = np.mean(fifteen1)
        averages["fifteen2"] = np.mean(fifteen2)
        averages["twentyseven1"] = np.mean(twentyseven1)
        averages["twentyseven2"] = np.mean(twentyseven2)
        averages["fiftyone1"] = np.mean(fiftyone1)
        averages["fiftyone2"] = np.mean(fiftyone2)

    filter_file = open("path_to_save/filterts.txt", "w+")
    filter_file.write("6")
    for filter in filter_set:
        if "fifteen" in filter:
            filter_file.write("\n\n9\t15\n")
        elif "twentyseven" in filter:
            filter_file.write("\n\n9\t27\n")
        elif "fiftyone" in filter:
            filter_file.write("\n\n9\t51\n")
        else:
            print "ERROR writing file"

        values = filter_set[filter]
        if zero:
            average = averages[filter]

            values = values - average

        print str(np.sum(values))

        for row in values:
            filter_file.write("\n")
            for column in row:
                filter_file.write(str(np.round(column, 3)) + "\t")


def wrap_pad(inputs, size=[0,0]):
   """
   Cylindrical wrap the image on both axis
   """
   wrapped = tf.concat([inputs[:,:, -size[1]:,:], inputs, inputs[:,:, 0:size[1],:]], 2)
   wrapped = tf.concat([wrapped[:,-size[0]:, :,:], wrapped, wrapped[:,0:size[0], :,:]], 1)
   return wrapped
def create_base_network(in_dims):
   """
   Base network to be shared.
   """
   inputs = Input(shape=in_dims)
   x1_pad = Lambda(lambda x: wrap_pad(x, [4, 7]))(inputs)
   x1 = Conv2D(2, (9, 15), activation='sigmoid', input_shape=in_dims,
               padding="valid",
               name="fifteen"
               )(x1_pad)
   x2_pad = Lambda(lambda x: wrap_pad(x, [4, 13]))(inputs)
   x2 = Conv2D(2, (9, 27), activation='sigmoid', input_shape=in_dims,
               padding="valid",
               name="twentyseven"
               )(x2_pad)
   x3_pad = Lambda(lambda x: wrap_pad(x, [4, 25]))(inputs)
   x3 = Conv2D(2, (9, 51), activation='sigmoid', input_shape=in_dims,
               padding="valid",
               name="fiftyone"
               )(x3_pad)
   conc = concatenate([x1, x2, x3], axis=-1, name='conv_merge')
   x = BitSampling(1536)(conc)
   model = Model(inputs, x)
   return model

gabor = True
weight_location = "where weights are stored"
if not gabor:
    input_shape = (64, 512, 1, )
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    # Shared embedding layer for positive and negative items
    Shared_DNN = create_base_network(input_shape)

    encoded_anchor = Shared_DNN(anchor_input)

    trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)
    trained_model.load_weights(weight_location + "name_of_weight_folder/batch_number.h5py")

    all_weights = trained_model.get_weights()
    fifteen1 = all_weights[0][:, :, :, 0]
    print "Sum of fifteen1 weights: " + str(np.sum(np.abs(fifteen1)))
    fifteen2 = all_weights[0][:, :, :, 1]
    print "Sum of fifteen2 weights: " + str(np.sum(np.abs(fifteen2)))

    twentyseven1 = all_weights[2][:, :, :, 0]
    print "Sum of twentyseven1 weights: " + str(np.sum(np.abs(twentyseven1)))
    twentyseven2 = all_weights[2][:, :, :, 1]
    print "Sum of twentyseven2 weights: " + str(np.sum(np.abs(twentyseven2)))

    fiftyone1 = all_weights[4][:, :, :, 0]
    print "Sum of fiftyone1 weights: " + str(np.sum(np.abs(fiftyone1)))
    fiftyone2 = all_weights[4][:, :, :, 1]
    print "Sum of fiftyone2 weights: " + str(np.sum(np.abs(fiftyone2)))

	# sets whether filters are set to sum to zero or just to be written raw
    zero = True

    convert_to_osiris(all_weights, zero)
    if zero:
        fifteen1 = fifteen1 - np.mean(fifteen1)
        fifteen2 = fifteen2 - np.mean(fifteen2)
        twentyseven1 = twentyseven1 - np.mean(twentyseven1)
        twentyseven2 = twentyseven2 - np.mean(twentyseven2)
        fiftyone1 = fiftyone1 - np.mean(fiftyone1)
        fiftyone2 = fiftyone2 - np.mean(fiftyone2)
    filter_set = [fifteen1, fifteen2, twentyseven1, twentyseven2,
                  fiftyone1, fiftyone2]

    plt.clf()
    for idx, filter in enumerate(filter_set):
        filter = np.squeeze(filter, axis=-1)
        plt.subplot(6, 2, idx + 1)
        plt.imshow(filter, interpolation="nearest", cmap="gray")
    plt.tight_layout()
    plt.savefig("plots/name_2d.png")
    # plt.show()

    plt.clf()
    fig = plt.figure(dpi=600)
    for idx, filter in enumerate(filter_set):
        filter = np.squeeze(filter, axis=-1)
        Y, X = filter.shape
        X = range(0, X)
        Y = range(0, Y)

        X, Y = np.meshgrid(X, Y)
        Z = filter
        ax = fig.add_subplot(3, 2, idx+1, projection='3d')
        ax.contourf3D(X, Y, Z, 200, cmap='gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.invert_yaxis()
    plt.savefig("plots/name_3d.png")
    # fig.show()


else:
    fifteen1 = np.genfromtxt('../Filters/9_15_1.txt', delimiter=',')
    fifteen2 = np.genfromtxt('../Filters/9_15_2.txt', delimiter=',')
    twentyseven1 = np.genfromtxt('../Filters/9_27_1.txt', delimiter=',')
    twentyseven2 = np.genfromtxt('../Filters/9_27_2.txt', delimiter=',')
    fiftyone1 = np.genfromtxt('../Filters/9_51_1.txt', delimiter=',')
    fiftyone2 = np.genfromtxt('../Filters/9_51_2.txt', delimiter=',')

    filter_set = [fifteen1, fifteen2, twentyseven1, twentyseven2, fiftyone1, fiftyone2]
    plt.clf()
    for idx, filter in enumerate(filter_set):
        plt.subplot(6, 2, idx + 1)
        plt.imshow(filter, interpolation="nearest", cmap="gray")
    plt.tight_layout()
    plt.savefig("plots/raw_2d.png")
    #plt.show()

    plt.clf()
    fig = plt.figure(dpi=600)
    for idx, filter in enumerate(filter_set):
        Y, X = filter.shape
        X = range(0, X)
        Y = range(0, Y)

        X, Y = np.meshgrid(X, Y)
        Z = filter
        ax = fig.add_subplot(3, 2, idx+1, projection='3d')
        ax.contourf3D(X, Y, Z, 100, cmap='gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.invert_yaxis()
    plt.savefig("plots/raw_3d.png")
    #fig.show()
