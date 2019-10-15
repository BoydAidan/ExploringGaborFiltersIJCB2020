#!/usr/bin/env python
from __future__ import division

import time
from os import path

import numpy as np
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import plot_model
from scipy.spatial.distance import hamming, euclidean
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, concatenate, Reshape
from keras.models import Model, Sequential
from keras.regularizers import l2, l1
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import os
import pickle
from itertools import permutations
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from imutils import paths
import cv2 as cv
import numpy.random as rng
import sys
from tqdm import tqdm
import tensorflow as tf
import sys

sys.path.insert(0, '../Custom_Layers/')
from bitsample_layer import BitSampling


#######################################
####### NETWORK STUFF##################
#######################################

####  NOTES:
# - This loads in images by the batch, if you want to increase speed you could modify code to bulk load in database
# - I did it this way as it meant I didnt run into RAM problems using large databases
# - For masks, 1 represents iris region, 0 represents occluded region
# - ***NB*** The database is expected to be in the folder format of "Database -> subject name -> images for that subject"
# - NB: The folder containing masks should be in the format "Database -> All mask images", no subject information is needed as we use the actual image to get that info

def occlude_masks(inputs):
    # Input in the form of the masks and the output of the network
    sample_1 = tf.reshape(inputs[0][0], (1536,))
    sample_2 = tf.reshape(inputs[0][1], (1536,))
    mask = tf.reshape(inputs[1], (1536,))

    # mask out bits that are 0 from the mask
    masked_sample_1 = tf.boolean_mask(sample_1, mask)
    masked_sample_2 = tf.boolean_mask(sample_2, mask)
    # Get the mean distance difference for each value in the feature vector
    score = K.mean(K.abs(masked_sample_1 - masked_sample_2))
    print K.mean(score)

    return [[1, 2], score]


# This is for a visual check and not indicative of performance, does not include masks so no decisions should be made on this
def validate_model(model, val_inputs):
    # calculate embeddings on the validation set
    predictions = model.predict_on_batch(val_inputs)

    # 3x1536
    total_length = 4608
    genuine_scores = []
    impostor_scores = []
    for pred in predictions:
        # extract out the different vectors
        anchor = pred[0:int(total_length * 1 / 3)]
        positive = pred[int(total_length * 1 / 3):int(total_length * 2 / 3)]
        negative = pred[int(total_length * 2 / 3):int(total_length * 3 / 3)]

        # Use euclidean for just a test
        ap = euclidean(anchor, positive)
        an = euclidean(anchor, negative)

        genuine_scores.append(ap)
        impostor_scores.append(an)

    d_prime1 = (abs(np.mean(genuine_scores) - np.mean(impostor_scores)) /
                np.sqrt(0.5 * (np.var(genuine_scores) + np.var(impostor_scores))))
    return d_prime1


num_layers = 1 # Used for naming convention

# Add wrap padding to image of a specific size
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

# Load in the mask for the current image
def load_image_mask(filename, label, path_to_data, path_to_masks):
    label = str(label)
    image = cv.imread(path_to_data + label + "/" + filename, cv.IMREAD_GRAYSCALE)
    image = image / 255
    maskname = filename.replace("_imno.bmp", "_mano.bmp")
    mask_path = path_to_masks + label + "/" + maskname
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    mask = mask/255
    reduced_mask = mask[6:49:6, 7::16]
    reduced_mask = reduced_mask.flatten()
    reduced_mask = np.array(reduced_mask.tolist() * 6)
    return image, reduced_mask


def get_batch(batch_size, data, labels_encoded, unique, counts, model, path_to_data, path_to_masks):
    # All unique labels
    n_classes = unique
    height = 64
    width = 512

    # initialize 2 empty arrays for the input image batch
    triplets = [np.zeros((batch_size, height, width, 1)) for asdf in range(3)]

    # # initialize vector for storing the targets
    targets = np.zeros((batch_size, 4608))

    classes = rng.choice(n_classes, size=(batch_size,), replace=False)
    for i in range(batch_size):
        # pick the first class
        class_1 = classes[i]
        # determine the number of available examples
        available_examples_1 = counts[unique == class_1]
        # randsample two examples from this class
        idx_1 = rng.choice(available_examples_1[0] - 1, size=2, replace=False)

        # now get the image data from the data array
        # all samples from this class at first
        class_1_indices = np.where(labels_encoded == class_1)
        # rand_class_1 = random.randint(0, len(class_1_indices[0] - 1))

        anchor = data[class_1_indices[0][idx_1[0]]]
        anchor, anchor_mask = load_image_mask(anchor, class_1, path_to_data, path_to_masks)
        positive = data[class_1_indices[0][idx_1[1]]]
        positive, positive_mask = load_image_mask(positive, class_1, path_to_data, path_to_masks)

        # Anchor sample
        triplets[0][i, :, :, :] = anchor.reshape(height, width, 1)

        # Positive Sample
        triplets[1][i, :, :, :] = positive.reshape(height, width, 1)

	# Generate combined mask for anchor positive pair
        combined_mask_ap = np.zeros(len(anchor_mask))

        for bit_loc, bit in enumerate(positive_mask):
            anc_bit = anchor_mask[bit_loc]

            if bit and anc_bit == 1:
                combined_mask_ap[bit_loc] = 1
            else:
                combined_mask_ap[bit_loc] = 0

        ## BEGIN TRIPLET MINING...
        batch_samples = [np.zeros((len(classes), height, width, 1)) for asdf in range(3)]
        batch_masks = np.zeros((len(classes), 1536,))
        for l in range(batch_size):
            # select random class
            rando = random.randint(0, len(unique) - 1)
            rand_class = unique[rando]
            # make sure that this random class is a class outside of the batch
            while rand_class in classes:
                rando = random.randint(0, len(unique) - 1)
                rand_class = unique[rando]
            # Indices of this second class
            indices = np.where(labels_encoded == rand_class)
            # Select random sample from this class
            rand = random.randint(0, len(indices[0]) - 1)
            rand_index = indices[0][rand]
            sample = data[rand_index]
            sample, imp_mask = load_image_mask(sample, rand_class, path_to_data, path_to_masks)

            batch_samples[0][l, :, :, :] = anchor.reshape(1,64,512,1)
            batch_samples[1][l, :, :, :] = positive.reshape(1,64,512,1)
            batch_samples[2][l, :, :, :] = sample.reshape(1, height, width, 1)

            # Generate combined mask for anchor and potential negative
            combined_mask_an = np.zeros(len(imp_mask))

            for bit_loc, bit in enumerate(imp_mask):
                anc_bit = anchor_mask[bit_loc]

                if bit and anc_bit == 1:
                    combined_mask_an[bit_loc] = 1
                else:
                    combined_mask_an[bit_loc] = 0
            batch_masks[l, :] = combined_mask_an.reshape(1536)

        embeddings = model.predict_on_batch([batch_samples[0], batch_samples[1], batch_samples[2]])
        total_length = 4608
        anc_emb = embeddings[:, 0:int(total_length * 1 / 3)][0]
        pos_emb = embeddings[:, int(total_length * 1 / 3):int(total_length * 2 / 3)][0]

        unmasked_anchor1 = anc_emb[combined_mask_ap == 1]
        unmasked_positive = pos_emb[combined_mask_ap == 1]

        if len(unmasked_anchor1) == 0 or len(unmasked_positive) == 0:
            print "Fully masked"
            continue

        distance_ap = np.mean(np.abs(unmasked_anchor1 - unmasked_positive))

        min_distance = float("inf")
        best_mask = np.zeros(1536)
        num_oob = 0
        for pos, embedding in enumerate(embeddings):
            combined_mask_an = batch_masks[pos]

            neg_emb = embedding[int(total_length * 2 / 3):int(total_length * 3 / 3)].reshape(1536)

            unmasked_anchor = anc_emb[combined_mask_an == 1]
            unmasked_negative = neg_emb[combined_mask_an == 1]

            # make sure not using fully masked out samples
            if len(unmasked_anchor) == 0 or len(unmasked_negative) == 0:
                print "Fully masked"
                continue

            distance_an = np.mean(np.abs(unmasked_anchor - unmasked_negative))

            if distance_an < min_distance: # and distance_an > distance_ap:
                min_distance = distance_an
                samp = batch_samples[2][pos, :, :, :].reshape(height, width, 1)
                triplets[2][i, :, :, :] = samp
                best_mask = combined_mask_an.reshape(1536)

        targets[i, :] = np.concatenate((combined_mask_ap.reshape(1536), best_mask, np.zeros(1536)))

    return triplets, targets


# Create a validation set, this does not do any mining just generates random triplets
def get_val(batch_size, data, labels_encoded, unique, counts, path_to_data, path_to_masks):
    height = 64
    width = 512
    classes = unique
    n_examples = counts

    # initialize triplet arrays
    triplets = [np.zeros((batch_size, height, width, 1)) for asdf in range(3)]

    # initialize vector for storing the targets
    targets = np.zeros((batch_size, 4608))

    # now iterate over batch and pick respective pairs
    for ij in range(batch_size):

        # pick the first class
        rand_class1 = random.randint(0, len(classes) - 1)
        class_1 = classes[rand_class1]
        available_examples_1 = n_examples[rand_class1]

        # randsample two examples from this class
        idx_1 = rng.choice(available_examples_1 - 1, size=2, replace=False)

        # now get the image data from the data array
        # all samples from this class at first
        class_1_indices = np.where(labels_encoded == class_1)
        # idx_1[0] = index of anchor

        anchor = data[class_1_indices[0][idx_1[0]]]
        anchor, anchor_mask = load_image_mask(anchor, class_1, path_to_data, path_to_masks)
        # idx_1[1] = index of positive
        positive = data[class_1_indices[0][idx_1[1]]]
        positive, positive_mask = load_image_mask(positive, class_1, path_to_data, path_to_masks)

        # Anchor sample
        triplets[0][ij, :, :, :] = anchor.reshape(height, width, 1)

        # Positive Sample
        triplets[1][ij, :, :, :] = positive.reshape(height, width, 1)

        combined_mask_ap = np.zeros(len(anchor_mask))

        for bit_loc, bit in enumerate(positive_mask):
            anc_bit = anchor_mask[bit_loc]

            if bit and anc_bit == 1:
                combined_mask_ap[bit_loc] = 1
            else:
                combined_mask_ap[bit_loc] = 0

        # Select second random class
        rand_class2 = random.randint(0, len(classes) - 1)
        class_2 = classes[rand_class2]
        # Make sure class1 and class2 are different
        while class_2 == class_1:
            rand_class2 = random.randint(0, len(classes) - 1)
            class_2 = classes[rand_class2]

        # do the same stuff as for the genuines
        available_examples_2 = n_examples[rand_class2]
        idx_2 = rng.choice(available_examples_2 - 1, size=1, replace=False)
        class_2_indices = np.where(labels_encoded == class_2)
        # idx_1[0] = index of anchor

        negative = data[class_2_indices[0][idx_2[0]]]
        negative, negative_mask = load_image_mask(negative, class_2, path_to_data, path_to_masks)
        triplets[2][ij, :, :, :] = negative.reshape(height, width, 1)

        combined_mask_an = np.zeros(len(negative_mask))

        for bit_loc, bit in enumerate(negative_mask):
            anc_bit = anchor_mask[bit_loc]

            if bit and anc_bit == 1:
                combined_mask_an[bit_loc] = 1
            else:
                combined_mask_an[bit_loc] = 0

        targets[ij, :] = np.concatenate((combined_mask_ap.reshape(1536), combined_mask_an.reshape(1536), np.zeros(1536)))

    return triplets, targets


# ## Triplet NN
def triplet_loss(y_true, y_pred, alpha=0.5):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)
    # y_true = K.print_tensor(y_true)
    total_length = y_pred.shape.as_list()[-1]
	# Extract anchor, positive and negative embeddings
    anchor = y_pred[:, 0:int(total_length * 1 / 3)]
    positive = y_pred[:, int(total_length * 1 / 3):int(total_length * 2 / 3)]
    negative = y_pred[:, int(total_length * 2 / 3):int(total_length * 3 / 3)]
	# Get anchor/positive combined mask and anchor/negative combined mask
    ap_mask = y_true[:, 0:int(total_length * 1 / 3)]
    an_mask = y_true[:, int(total_length * 1 / 3):int(total_length * 2 / 3)]
    # Occlude masks using occlude_masks function
    out_ap = tf.map_fn(occlude_masks, elems=[[anchor, positive], ap_mask], infer_shape=False)
    out_an = tf.map_fn(occlude_masks, elems=[[anchor, negative], an_mask], infer_shape=False)
    pos_dist = out_ap[1]
    neg_dist = out_an[1]

    # pos_dist = K.print_tensor(pos_dist)
    # neg_dist = K.print_tensor(neg_dist)

    # compute loss
    loss = K.mean(K.log(1 + (K.exp(pos_dist - neg_dist))))

    return loss


def load_data_from_path(path_to_dataset):
    # initialize containers for data and labels
    data = []
    labels = []

    imagePaths = list(paths.list_images(path_to_dataset))

    data = np.empty(len(imagePaths), dtype=object)
    labels = np.empty(len(imagePaths), dtype=object)

    # load data from patches directory into a numpy array
    print "[INFO] describing images in the dataset"
    for i, imagePath in enumerate(tqdm(imagePaths)):
        # read the i-th image


        # extract the label
        path = imagePaths[i].split(os.path.sep)
        label = path[-2]

        filename = path[-1]

        data[i] = filename
        labels[i] = label

    [unique, counts] = np.unique(labels, return_counts=True)

    return data, labels, unique, counts


### HARD CODED PARAMS
solver = "adam"
learning_rate = 0.01
gabor = False # if you want to start from Gabor
fine_tuned = False # If you want to restart training from current weights
input_shape = (64, 512, 1,)
training_data = "" # path to all data
training_masks = "" # path to all masks
out_loc = "" # where to save weights

####################################
# Initialize Network

if gabor:
    g = "gabor"
else:
    g = "random"

ft = ""
if fine_tuned:
    ft = "_ft"

folder = g + "_" + solver + "_" + str(learning_rate) + ft + "_exp_pad_" + str(num_layers)
out_loc = out_loc + folder + "/"

if not os.path.exists(out_loc):
    os.makedirs(out_loc)

if solver == 'adam':
    adam_optim = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
elif solver == 'sgd':
    adam_optim = SGD(lr=learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
else:
    raise Exception('Solver not recognized!')

anchor_input = Input(input_shape, name='anchor_input')
positive_input = Input(input_shape, name='positive_input')
negative_input = Input(input_shape, name='negative_input')

# Shared embedding layer for positive and negative items
Shared_DNN = create_base_network(input_shape)

# To switch between random weight initialization and gabor initialization
if gabor:
	
	# Load in OSIRIS Gabor kernels from text files
    weights = Shared_DNN.get_weights()
    fifteen1 = np.genfromtxt('../Filters/9_15_1.txt', delimiter=',')
    fifteen2 = np.genfromtxt('../Filters/9_15_2.txt', delimiter=',')
    twoseven1 = np.genfromtxt('../Filters/9_27_1.txt', delimiter=',')
    twoseven2 = np.genfromtxt('../Filters/9_27_2.txt', delimiter=',')
    fiftyone1 = np.genfromtxt('../Filters/9_51_1.txt', delimiter=',')
    fiftyone2 = np.genfromtxt('../Filters/9_51_2.txt', delimiter=',')

    for id1, row in enumerate(weights[0]):
        for id2, col in enumerate(row):
            value = np.array([fifteen1[id1][id2], fifteen2[id1][id2]])
            weights[0][id1][id2] = [value]

    for id1, row in enumerate(weights[2]):
        for id2, col in enumerate(row):
            value = np.array([twoseven1[id1][id2], twoseven2[id1][id2]])
            weights[2][id1][id2] = [value]

    for id1, row in enumerate(weights[4]):
        for id2, col in enumerate(row):
            value = np.array([fiftyone1[id1][id2], fiftyone2[id1][id2]])
            weights[4][id1][id2] = [value]

    Shared_DNN.set_weights(weights)

    weights2 = Shared_DNN.get_weights()

    for idx, row in enumerate(weights2):
		# make sure weights correctly loaded
        if np.array_equal(weights[idx], weights2[idx]):
            print "Layer weights correct"
        else:
            print "Layer weight incorrect"

encoded_anchor = Shared_DNN(anchor_input)
encoded_positive = Shared_DNN(positive_input)
encoded_negative = Shared_DNN(negative_input)

merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector, name="Triplet Model")

model.compile(loss=triplet_loss, optimizer=adam_optim)

if fine_tuned:
    model.load_weights('path to weights you want to start from')
    print "Weights loaded"

model.save_weights(out_loc + 'before_' + folder + ".h5py")

model.summary()

# Plot the model
# plot_model(Shared_DNN, to_file=out_loc + 'encoded_anchor_' + str(num_layers) + '.png', show_shapes=True, show_layer_names=True)
# plot_model(model, to_file=out_loc + 'full_model_' + str(num_layers) + '.png', show_layer_names=True, show_shapes=True)


#######################################
# IRIS DATA LOADING


print "Loading in data..."
x, y, unique, counts = load_data_from_path(training_data)

print "Creating train/test splits"
training_proportion = 0.8
num_training = int(training_proportion * len(unique))
random.seed(42)
rand_train_indices = random.sample(unique, num_training)
x_train = []
masks_train = []
x_validation = []
masks_test = []
y_train = []
y_validation = []

# Generate subject disjoint train/test split
for sub_id in unique:
    sub_id = str(sub_id)
    indices = np.where(y == sub_id)
    vals = x[y == sub_id]

    if sub_id in rand_train_indices:
        for qwerty, im in enumerate(vals):
            x_train.append(im)
            y_train.append(sub_id)
    else:
        for qwerty, im in enumerate(vals):
            x_validation.append(im)
            y_validation.append(sub_id)

x_train = np.array(x_train)
y = np.array(y_train)
x_test = np.array(x_validation)
y_validation = np.array(y_validation)

print "Length of training subset: " + str(len(x_train))
print "Length of training subset: " + str(len(x_test))

[unique, counts] = np.unique(y, return_counts=True)
[unique_test, counts_test] = np.unique(y_validation, return_counts=True)

# Free up some memory
x = []
x_validation = []
masks = []


################################################
# NETWORK TRAINING
batch_size = 64
# Max training time
n_iterations = 200000

# variable to store accuracies
accuracies = []
accuracies_test = []
losses = []
losses_test = []

# #######
# create a fixed batch of testing data
print "[INFO] Generating validation set..."
val_before = time.time()
inputs_test, targets_test = get_val(2048, x_test, y_validation, unique_test, counts_test, training_data, training_masks)
val_creation = time.time() - val_before
x_test = []
print "Validation creation time: " + str(val_creation)
print "[INFO] Training begins..."

loss_list = []
previous_loss_1 = 100
previous_loss_2 = 100
for i in range(1, n_iterations):
    before_batch = time.time()
    (inputs, targets) = get_batch(batch_size, x_train, y, unique, counts, model, training_data, training_masks)
    targets = targets.reshape(-1, 4608)

    loss = model.train_on_batch([inputs[0], inputs[1], inputs[2]], targets)

    loss_list.append(loss)
    batch_time = time.time() - before_batch

    print "Iteration " + str(i) + " loss: " + str(loss) + ", time: " + str(batch_time)

    if i % 100 == 0:
        print("[INFO] Validating and saving model: {}/{}".format(i, n_iterations))
        # and evaluate the testing accuracy while we are here
        before = time.time()
        loss_test = model.test_on_batch([inputs_test[0], inputs_test[1], inputs_test[2]], targets_test)
        d_prime = validate_model(model, [inputs_test[0], inputs_test[1], inputs_test[2]])
        time_taken = time.time() - before
        # print "Loss on validation set: " + str(loss_test)
        losses_test.append(loss_test)
        losses.append(loss)
        model.save(out_loc + str(i) + ".h5py")

        print "Loss on the testing set is:" + str(loss_test) + " and ran in: " + str(time_taken)
        print "d prime: " + str(d_prime)

        mean_loss = round(loss_test, 4)
        loss_list = []

        if mean_loss < previous_loss_1:
            previous_loss_2 = previous_loss_1
            previous_loss_1 = mean_loss
        elif mean_loss >= previous_loss_1 and mean_loss >= previous_loss_2:
            print "Training complete, loss did not decrease over two validation steps."
            break


print "[INFO] Saving the model weights..."
model.save(out_loc + folder + ".h5py")
