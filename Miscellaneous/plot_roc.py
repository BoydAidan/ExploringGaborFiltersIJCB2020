from __future__ import print_function,division

import argparse
import glob, os
import time
from multiprocessing import Pool

import numpy as np
from sklearn import svm
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import multiclass
from sklearn import metrics


def compute_metrics():

    authentic_scores = []
    impostor_scores = []

    genuine_file = open('path to genuine scores', 'r')
    impostor_file = open('path to impostor scores', 'r')

    for line in genuine_file:
        tokens = line.split(" ")
        authentic_scores.append(float(tokens[2]))

    for line in impostor_file:
        tokens = line.split(" ")
        impostor_scores.append(float(tokens[2]))

    authentic_scores = np.array(authentic_scores).ravel()
    impostor_scores = np.array(impostor_scores).ravel()

    fpr1, tpr1, thr1 = compute_roc(authentic_scores, impostor_scores)

    np.save("score_files/fpr_name.npy", fpr1)
    np.save("score_files/tpr_name.npy", tpr1)
    np.save("score_files/thr_name.npy", thr1)
    print('ROC Calculated')



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx


def compute_roc(authentic_score, impostor_score, ignore_aut=-1, ignore_imp=-1):
    if ignore_aut != -1:
        authentic_score = authentic_score[authentic_score[:, 0].astype(int) < ignore_aut, 1].astype(float)
    elif np.ndim(authentic_score) == 1:
        authentic_score = authentic_score.astype(float)
    else:
        authentic_score = authentic_score[:, 2].astype(float)
    authentic_y = np.ones(authentic_score.shape[0])
    print(authentic_score.shape)


    if ignore_imp != -1:
        impostor_score = impostor_score[impostor_score[:, 0].astype(int) < ignore_imp, 1].astype(float)

    elif np.ndim(impostor_score) == 1:
        impostor_score = impostor_score.astype(float)
    else:
        impostor_score = impostor_score[:, 2].astype(float)
    impostor_y = np.zeros(impostor_score.shape[0])
    print(impostor_score.shape)

    y = np.concatenate([authentic_y, impostor_y])
    scores = np.concatenate([authentic_score, impostor_score])

    # invert scores in case of distance instead of similarity
    scores *= -1

    print(y.shape)

    return metrics.roc_curve(y, scores, drop_intermediate=True)


def plot(title, fpr1, tpr1, thr1, l1):
    label_kwargs1 = {}
    label_kwargs1['bbox'] = dict(
        boxstyle='round,pad=0.5', alpha=0.5,
    )

    plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams['font.size'] = 12

    plt.grid(True, zorder=0, linestyle='dashed')

    if title is not None:
        plt.title(title)

    plt.gca().set_xscale('log')

    begin_x = 1e-5
    end_x = 1e0
    print(begin_x, end_x)

    range = end_x - begin_x

    auc1 = metrics.auc(fpr1[(fpr1 >= begin_x) & (fpr1 <= end_x)], tpr1[(fpr1 >= begin_x) & (fpr1 <= end_x)])
    auc1_per = auc1 / range
    print(auc1_per)

    plt.plot(fpr1, tpr1, label=l1)
    # + ' - AUC ({:0.5f})'.format(auc1_per))

    thrs = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1]

    offset = 2

    invert = -1

    for i in thrs:
        y_pos = -30

        if i != 0.00001:
            offset = 0

        k = find_nearest(fpr1, i)
        t1 = str(np.round(thr1[k], 4) * invert)
        x1 = fpr1[k + offset]
        y1 = tpr1[k]

        print('X: ' + str(x1) + " , Y: " + str(y1))

        # y_pos = -55

        plt.annotate(t1, (x1, y1), xycoords='data',
                     xytext=(15, y_pos), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->"), fontsize=10,
                     **label_kwargs1)

        y_pos = 30
        inv = 1
        t1_1 = str(np.round(tpr1[k], 4) * inv)

        plt.annotate(t1_1, (x1, y1), xycoords='data',
                     xytext=(15, y_pos), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->"), fontsize=10,
                     **label_kwargs1)

    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([begin_x, end_x])
    # plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Match Rate')

    plt.tight_layout(pad=0)
    plt.savefig('test_roc.png', dpi=300)
    plt.show()
    return plt


compute_metrics()

