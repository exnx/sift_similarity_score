import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
import cv2
import argparse



# read two images from arg.parse

parser = argparse.ArgumentParser(description='sift match scorer')
parser.add_argument('--img1')
parser.add_argument('--img2')
parser.add_argument('--use_gray')
args = parser.parse_args()

filename1 = args.img1
filename2 = args.img2

img1=cv2.imread(filename1)
img2=cv2.imread(filename2)

# flip to rgb ordering
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

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

# calculate new shapes
h1, w1, c1 = img1.shape
h2, w2, c2 = img2.shape

print('img1 shape', h1, w1, c1)
print('img2 shape', h2, w2, c2)

# convert to grayscale

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# # binary threshold
# _, thresh1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
# _, thresh2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)


cv2.imwrite('thresh1.jpg', gray1)
cv2.imwrite('thresh2.jpg', gray2)


# detect sift features
sift = cv2.xfeatures2d.SIFT_create()

if args.use_gray:
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

else:
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

# calc the total number of keypoints for both images
num_keypoints = 0
if len(kp1) >= len(kp2):
    num_keypoints = len(kp1)
else:
    num_keypoints = len(kp2)
print("Keypoints 1ST Image: " + str(len(kp1)))
print("Keypoints 2ND Image: " + str(len(kp2)))


# match sift features
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# threshold for what is considered local in each direction of a feature match 
x_local = 0.3
y_local = 0.3

# filter good matches by distance (L2 from discriptors), using ratio test

good_dist_count = 0

good_matches = []  
for m, n in matches:

    # n is the second closest match, so this measures the uniqueness of the m match
    if m.distance < 0.8 * n.distance:

        good_dist_count += 1

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
        if x_diff < x_local and y_diff < y_local:

            # print('x1, y1', x1, y1)
            # print('x2, y2', x2, y2)

            # print('x_diff', x_diff)
            # print('y_diff', y_diff)

            # add the match points
            good_matches.append([m])

if good_dist_count == 0:
    print('no good matches')
    exit()

print("GOOD Matches:", len(good_matches))
print("% of good feature matches: ", len(good_matches) / good_dist_count * 100, "%")

if args.use_gray:
    img3 = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good_matches, None, flags=2)
else:
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

plt.imshow(img3),plt.show()



