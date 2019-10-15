import os
from multiprocessing import Pool

from PIL import Image
from scipy import mean, stats
import numpy as np
from skimage import io
# import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def align(subject):
# for subject in tqdm(classes):
#     print subject

    images = source_classes + subject + "/"
    correlations = {}
    for image1 in os.listdir(images):
        img_path1 = images + image1
        img1 = io.imread(img_path1, as_gray=True)
        individual_correlations = []
        for image2 in os.listdir(images):
            if image1 == image2:
                continue

            img_path2 = images + image2
            img2 = io.imread(img_path2, as_gray=True)
            img1 = np.array(img1).flatten()
            img2 = np.array(img2).flatten()
            image_correlation = stats.pearsonr(img1, img2)
            individual_correlations.append(abs(image_correlation[0]))
        mean_correlation = mean(individual_correlations)
        correlations[image1] = mean_correlation
    max_corr = max(correlations, key=correlations.get)
    # print max_corr
    # print correlations[max_corr]
    # reference_image[subject] = max_corr

    ref_img = io.imread(images + max_corr, as_gray=True)
    ref_flat = np.array(ref_img).flatten()

    height, width = ref_img.shape
    # print height, width

    for image in os.listdir(images):

        sub_dst = destination + subject
        if not os.path.exists(sub_dst):
            os.makedirs(sub_dst)
        dst = sub_dst + "/" + image
        mask_src = mask_source + image.replace("_imno.bmp", "_mano.bmp")
        mask_dst = mask_destination + image.replace("_imno.bmp", "_mano.bmp")
        max_correlation = 0
        shift_needed = 0
        if image == max_corr:
            im = Image.fromarray(ref_img)
            mask = io.imread(mask_src, as_gray=True)
            mask = Image.fromarray(mask)
            mask.save(mask_dst)
            im.save(dst)
            continue

        img_path = images + image
        img_to_shift = io.imread(img_path, as_gray=True)
        img_to_shift = np.array(img_to_shift).flatten()

        mask_to_shift = io.imread(mask_src, as_gray=True)
        mask_to_shift = np.array(mask_to_shift).flatten()

        for i in range(width):
            shifted = np.roll(img_to_shift, i)
            corr = stats.pearsonr(ref_flat, shifted)
            plot_points[i].append(corr)

            if corr > max_correlation:
                max_correlation = corr
                shift_needed = i

        # print "Shift needed: " + str(shift_needed)
        corrected_image = np.roll(img_to_shift, shift_needed)
        corrected_image = corrected_image.reshape((height, width))
        im = Image.fromarray(corrected_image)
        im.save(dst)

        corrected_mask = np.roll(mask_to_shift, shift_needed)
        corrected_mask = corrected_mask.reshape((height, width))
        msk = Image.fromarray(corrected_mask)
        msk.save(mask_dst)

# The source classes must be in folder format "Database -> subject ID -> images for that subject"
# Masks should be in format "Database -> all images"
source_classes = "path to unaligned data"
destination = "where to save aligned data"
mask_source = "unaligned masks"
mask_destination = "where to save aligned masks"
# reference_image = {}
plot_points = []
for j in range(512):
    plot_points.append([])
classes = os.listdir(source_classes)
pool = Pool(30)
before = time.time()
for alignment in tqdm(pool.imap_unordered(align, classes)):
    pass
# pool.map(align, classes)
total_exec = time.time() - before
print "Execution time: " + str(total_exec)
points = []
for vals in plot_points:
    points.append(mean(vals))

# plt.plot(points)
# plt.savefig("mean_corrs.png")
# plt.show()
