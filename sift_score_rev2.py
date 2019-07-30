import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
import cv2
import argparse





# filter good matches
    
    # filter good matches by relative location (must be in same vacinity)


# read two images from arg.parse

parser = argparse.ArgumentParser(description='sift match scorer')
parser.add_argument('--img1')
parser.add_argument('--img2')
args = parser.parse_args()

filename1 = args.img1
filename2 = args.img2

img1=cv2.imread(filename1)
img2=cv2.imread(filename2)

# take two images and enforce same size by resizing larger to smaller

h1, w1, c1 = img1.shape
h2, w2, c2 = img2.shape

# h1 is bigger so resize h1 to h2
if h1 > h2:
    # calculate ratio
    ratio = h2 / h1
    img1 = cv2.resize(img1, (0,0), fx=ratio, fy=ratio)
    
# resize h2 to h1
else:
    # calculate ratio
    ratio = h1 / h2
    img2 = cv2.resize(img2, (0,0), fx=ratio, fy=ratio)

# convert to grayscale

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# find the range of pixel values
min_pixel_1 = np.min(gray1)
max_pixel_1 = np.max(gray1)

min_pixel_2 = np.min(gray2)
max_pixel_2 = np.max(gray2)


# binary threshold between the range of pixels
_, thresh1 = cv2.threshold(gray1, 150, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(gray2, 150, 255, cv2.THRESH_BINARY)

cv2.imwrite('thresh1.jpg', thresh1)
cv2.imwrite('thresh2.jpg', thresh2)


# detect sift features

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(thresh1, None)
kp2, des2 = sift.detectAndCompute(thresh2, None)

# img1 = cv2.drawKeypoints(img1, kp1, None)
# img2 = cv2.drawKeypoints(img2, kp2, None)


# cv2.imwrite('sift_keypoints1.jpg',img1)
# cv2.imwrite('sift_keypoints2.jpg',img2)

# calc the total number of keypoints for both images
num_keypoints = 0
if len(kp1) <= len(kp2):
    num_keypoints = len(kp1)
else:
    num_keypoints = len(kp2)
print("Keypoints 1ST Image: " + str(len(kp1)))
print("Keypoints 2ND Image: " + str(len(kp2)))


# match sift features
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)


region_threshold = 0.15

# filter good matches by distance (L2 from discriptors), using ratio test

good_dist_count = 0

good_matches = []  
for m, n in matches:

    good_dist_count += 1

    # n is the second closest match, so this measures the uniqueness of the m match
    if m.distance < 0.75 * n.distance:

        # now extract the indexes of the match on both images
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx

        # Get the coordinates of matches on both images
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # get relative region diff in x and y directions
        x_diff = abs(x1-x2) / w1
        y_diff = abs(y1-y2) / h1

        # filter good_dist_points by searching within within a proximinity only
        if x_diff < region_threshold and y_diff < region_threshold:

            # print('x1, y1', x1, y1)
            # print('x2, y2', x2, y2)

            # print('x_diff', x_diff)
            # print('y_diff', y_diff)

            # add the match points
            good_matches.append([m])

print("GOOD Matches:", len(good_matches))
print("% of good feature matches: ", len(good_matches) / good_dist_count * 100, "%")

img3 = cv2.drawMatchesKnn(thresh1, kp1, thresh2, kp2, good_matches, None, flags=2)
plt.imshow(img3),plt.show()



