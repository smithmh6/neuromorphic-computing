import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import os
import sys

im_path = os.path.normpath(r'/Users/heathsmith/repos/github/CSCE790/CSCE790-Neuromorphic-Computing/datasets/FER-2013-Original/train/happy/Training_1866804.jpg')

fig = plt.figure(figsize=(10, 6))
plt_rows = 1
plt_cols = 4

im = cv.imread(im_path, -1)
im = cv.resize(im, (48, 48))
im_rgb = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

# extract faces
face_haar_cascade = cv.CascadeClassifier('/Users/heathsmith/repos/github/neuromorphic-computing/openvino/haarcascade_frontalface_defaults.xml')
extracted_faces = []

faces = face_haar_cascade.detectMultiScale(im)

for (x, y, w, h) in faces:
    cv.rectangle(im_rgb, (x, y),(x + w, y + h),(0,255,0), 1)

ax1 = plt.subplot(plt_rows, plt_cols, 1)
plt.imshow(im_rgb)
plt.title("Detection")
plt.axis('off')

ax2 = plt.subplot(plt_rows, plt_cols, 2)
cropped = im[y:y + h, x:x + w]
cropped = cv.resize(cropped, (48, 48))
plt.imshow(cropped, cmap='gray')
plt.title("Cropping")
plt.axis('off')

ax3 = plt.subplot(plt_rows, plt_cols, 3)
k = 3
kernel = np.ones((k, k)) / k ** 2
f = 2
im_up = cv.resize(cropped, (48*f, 48*f))
im_smooth = cv.GaussianBlur(im_up, (k, k), sigmaX=0, sigmaY=0)
plt.imshow(im_smooth, cmap='gray')
plt.title("Smoothed")
plt.axis('off')

ax4 = plt.subplot(plt_rows, plt_cols, 4)
conv = cv.convertScaleAbs(im_smooth)
edge = cv.Canny(image=conv, threshold1=25, threshold2=125)
#edge = cv.resize(edge, (48, 48))
plt.imshow(edge, cmap='gray')
plt.title("Edges")
plt.axis('off')

plt.tight_layout()
plt.show()

# Display the resulting frame
## cv.imshow('Face Detection', im_rgb)

# wait for 'c' to close the application
## cv.waitKey(0)
## cv.destroyAllWindows()