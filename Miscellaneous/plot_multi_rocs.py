from __future__ import print_function,division

import argparse
import glob, os
import time
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

### USED TO PLOT MULTIPLE ROCS ON ONE GRAPH


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx



def plot(title, figure_title, color, comb, fpr1, tpr1, thr1, l1, fpr2, tpr2, thr2, l2,
         fpr3, tpr3, thr3, l3, fpr4, tpr4, thr4, l4, fpr5, tpr5, thr5, l5):
    label_kwargs1 = {}
    label_kwargs1['bbox'] = dict(
        boxstyle='round,pad=0.5', fc=color, alpha=0.5,
    )

    label_kwargs2 = {}
    label_kwargs2['bbox'] = dict(
        boxstyle='round,pad=0.5', fc='#c6231b', alpha=0.5,
    )

    label_kwargs3 = {}
    label_kwargs3['bbox'] = dict(
        boxstyle='round,pad=0.5', fc='#4d6579', alpha=0.5,
    )

    label_kwargs4 = {}
    label_kwargs4['bbox'] = dict(
        boxstyle='round,pad=0.5', fc='#79508d', alpha=0.5,
    )

    label_kwargs5 = {}
    label_kwargs5['bbox'] = dict(
        boxstyle='round,pad=0.5', fc='#cc6600', alpha=0.5,
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

    if comb:
        plt.plot(fpr1, tpr1, color, label=l1)
    else:
        plt.plot(fpr1, tpr1, color)
    # + ' - AUC ({:0.5f})'.format(auc1_per))

    if l2 is not None:
        auc2 = metrics.auc(fpr2[(fpr2 >= begin_x) & (fpr2 <= end_x)], tpr2[(fpr2 >= begin_x) & (fpr2 <= end_x)])
        auc2_per = auc2 / range
        print(auc2_per)

        plt.plot(fpr2, tpr2, '#c6231b', label=l2)
        # + ' - AUC ({:0.5f})'.format(auc2_per))

    if l3 is not None:
        auc3 = metrics.auc(fpr3[(fpr3 >= begin_x) & (fpr3 <= end_x)], tpr3[(fpr3 >= begin_x) & (fpr3 <= end_x)])
        auc3_per = auc3 / range
        print(auc3_per)

        plt.plot(fpr3, tpr3, '#4d6579', label=l3)
        # + ' - AUC ({:0.5f})'.format(auc3_per))

    if l4 is not None:
        auc4 = metrics.auc(fpr4[(fpr4 >= begin_x) & (fpr4 <= end_x)], tpr4[(fpr4 >= begin_x) & (fpr4 <= end_x)])
        auc4_per = auc4 / range
        print(auc4_per)

        plt.plot(fpr4, tpr4, '#79508d', label=l4)

    if l5 is not None:
        auc5 = metrics.auc(fpr5[(fpr5 >= begin_x) & (fpr5 <= end_x)], tpr5[(fpr5 >= begin_x) & (fpr5 <= end_x)])
        auc5_per = auc5 / range
        print(auc5_per)

        plt.plot(fpr5, tpr5, '#cc6600', label=l5)

    # thrs = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]

    thrs = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    offset = 2

    invert = 1

    for i in thrs:
        y_pos = -30

        if i != 0.00001:
            offset = 0

        k = find_nearest(fpr1, i)
        t1 = str(np.round(tpr1[k], 4) * invert)
        x1 = fpr1[k + offset]
        y1 = tpr1[k]

        print('X: ' + str(x1) + " , Y: " + str(y1))

        if l2 is not None:
            k = find_nearest(fpr2, i)
            t2 = str(np.round(tpr2[k], 4) * invert)
            x2 = fpr2[k + offset]
            y2 = tpr2[k]

        if l3 is not None:
            k = find_nearest(fpr3, i)
            t3 = str(np.round(tpr3[k], 4) * invert)
            x3 = fpr3[k + offset]
            y3 = tpr3[k]

        if l4 is not None:
            k = find_nearest(fpr4, i)
            t4 = str(np.round(tpr4[k], 4) * invert)
            x4 = fpr4[k + offset]
            y4 = tpr4[k]

        if l5 is not None:
            k = find_nearest(fpr5, i)
            t5 = str(np.round(tpr5[k], 4) * invert)
            x5 = fpr5[k + offset]
            y5 = tpr5[k]

        y_pos = -30

        if comb == False:
            plt.annotate(t1, (x1, y1), xycoords='data',
                         xytext=(15, y_pos), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"), fontsize=10,
                         **label_kwargs1)

        if l2 is not None:
            if abs(y1 - y2) < 0.02:
                y_pos -= 30

            y_pos = -20

            # plt.annotate(t2, (x2, y2), xycoords='data',
            #              xytext=(15, y_pos), textcoords='offset points',
            #              arrowprops=dict(arrowstyle="->"), fontsize=10,
            #              **label_kwargs2)

        if l3 is not None:
            y_pos = -30

            if (abs(y1 - y2) < 0.02) and (abs(y2 - y3) < 0.05):
                y_pos -= 60
            elif (abs(y2 - y3) < 0.02):
                y_pos -= 30

            if i == 0.00001:
               y_pos = -25
            else:
               y_pos = -40

            # plt.annotate(t3, (x3, y3), xycoords='data',
            #              xytext=(15, y_pos), textcoords='offset points',
            #              arrowprops=dict(arrowstyle="->"), fontsize=10,
            #              **label_kwargs3)

        if l4 is not None:
            y_pos = -30

            if (abs(y1 - y2) < 0.02) and (abs(y2 - y3) < 0.05):
                y_pos -= 60
            elif (abs(y2 - y3) < 0.02):
                y_pos -= 30
            elif (abs(y3 - y4) < 0.02):
                y_pos -= 30

            if i == 0.00001:
               y_pos = -25
            else:
               y_pos = -40

            # plt.annotate(t4, (x4, y4), xycoords='data',
            #              xytext=(15, y_pos), textcoords='offset points',
            #              arrowprops=dict(arrowstyle="->"), fontsize=10,
            #              **label_kwargs4)

        if l5 is not None:
            y_pos = -30

            if (abs(y1 - y2) < 0.02) and (abs(y2 - y3) < 0.05):
                y_pos -= 60
            elif (abs(y2 - y3) < 0.02):
                y_pos -= 30
            elif (abs(y3 - y4) < 0.02):
                y_pos -= 30
            elif (abs(y4 - y5) < 0.02):
                y_pos -= 30

            # if i == 0.00001:
            #    y_pos = -25
            # else:
            #    y_pos = -40

            # plt.annotate(t5, (x5, y5), xycoords='data',
            #              xytext=(15, y_pos), textcoords='offset points',
            #              arrowprops=dict(arrowstyle="->"), fontsize=10,
            #              **label_kwargs5)

    plt.legend(loc='lower right', fontsize=18)
    plt.xlim([begin_x, end_x])
    # plt.ylim([0.95, 1])
    # plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Match Rate', fontsize=18)
    # plt.plot(0.001, 0.988, 'kx', markersize=20)
    # plt.annotate("Nguyen et al.", (0.0013, 0.988))

    plt.tight_layout(pad=0.2)
    fig_path = figure_title
    plt.savefig(fig_path, dpi=600)
    plt.show()
    return plt

base_path = 'score_files/'

db = 'name'

fpr1 = np.load(base_path + 'fpr_raw_' + db + '.npy')
tpr1 = np.load(base_path + 'tpr_raw_' + db + '.npy')
thr1 = np.load(base_path + 'thr_raw_' + db + '.npy')

fpr2 = np.load(base_path + 'fpr_rnd_' + db + '.npy')
tpr2 = np.load(base_path + 'tpr_rnd_' + db + '.npy')
thr2 = np.load(base_path + 'thr_rnd_' + db + '.npy')

fpr3 = np.load(base_path + 'fpr_gab_' + db + '.npy')
tpr3 = np.load(base_path + 'tpr_gab_' + db + '.npy')
thr3 = np.load(base_path + 'thr_gab_' + db + '.npy')

# fpr4 = np.load(base_path + 'fpr_    .npy')
# tpr4 = np.load(base_path + 'tpr_    .npy')
# thr4 = np.load(base_path + 'thr_    .npy')

# fpr5 = np.load(base_path + 'fpr_    .npy')
# tpr5 = np.load(base_path + 'tpr_    .npy')
# thr5 = np.load(base_path + 'thr_    .npy')

label1 = "Original Gabor"
label2 = "Randomly Initialized"
label3 = "Gabor Initialized"

title = "Name of database"
# title = None
figure_title = "score_files/" + db + "_roc.png"
color = "#30949a"
comb = True
#

# comb = False
# fpr2, tpr2, thr2, label2 = (None, None, None, None)
# fpr3, tpr3, thr3, label3 = (None, None, None, None)
fpr4, tpr4, thr4, label4 = (None, None, None, None)
fpr5, tpr5, thr5, label5 = (None, None, None, None)

plot(title, figure_title, color, comb,fpr1, tpr1, thr1, label1,
         fpr2, tpr2, thr2, label2, fpr3, tpr3, thr3, label3,
     fpr4, tpr4, thr4, label4, fpr5, tpr5, thr5, label5)

