
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import skimage.transform as tf
import scipy.signal as signal
import re
import sys
from utils.aligner import Aligner

aln =  Aligner()
def atoi(text):
    return int(text) if text.isdigit() else text.lower()
    
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

# retrieve and scale files
filelist = glob.glob('figs/A_RAW/*.txt')
filelist.sort(key=natural_keys)
A_array = np.array([np.array(np.loadtxt(fname)) for fname in filelist])

for i, A in enumerate(A_array):
    print("Image " + str(i))
    print("Title: " + filelist[i])
    print("Min before scaling: " + str(A.min()))
    print("Max before scaling: " + str(A.max()))
    # convert to uint8 range
    A_array[i] = np.interp(A, (A.min(), A.max()), (0, (2**16 - 1)))

A_array = A_array.astype(np.uint16)
select = input("Find alignment for which images? (i.e \"0,1\" or \"3,1\")\n")
select = np.fromstring(select, dtype=int, sep=",")
print(select)
[source, target] = A_array[select]
source = aln.sharpen(aln.upsample(source))
target = aln.sharpen(aln.upsample(target))

im_reg, trans = aln.alignImages(source, target)
xmin, xmax, ymin, ymax = aln.autoCropper(im_reg)

plt.figure(figsize=(12,12))
plt.imshow(im_reg[ymin:ymax,xmin:xmax])
plt.title("Registered source")

plt.figure(figsize=(12,12))
plt.imshow(target[ymin:ymax,xmin:xmax])
plt.title("Target")


plt.figure(figsize=(12,12))
plt.imshow(cv2.addWeighted(im_reg, -1, target, 1, 0))
plt.title("Difference image")

plt.show()

plt.imsave("reg", im_reg[ymin:ymax,xmin:xmax])
plt.imsave("target", target[ymin:ymax,xmin:xmax])