import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
import cv2

filename1 = 'ca_c1.png'
filename2 = 'ca_c2.png'
filename3 = 'ca_c3.png'

img1=cv2.imread(filename1)
img2=cv2.imread(filename2)
img3=cv2.imread(filename3)


# resize img 1
img1 = cv2.resize(img1, (0,0), fx=0.25, fy=0.25)

# resize img3
img3 = cv2.resize(img3, (50,50))




# now all the same size


# conver to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)



sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray3,None)


img1 = cv2.drawKeypoints(img1, kp1, None)
img2 = cv2.drawKeypoints(img2, kp2, None)


cv2.imwrite('sift_keypoints1.jpg',img1)
cv2.imwrite('sift_keypoints2.jpg',img2)



# calc num of min keypoints between the images
num_keypoints = 0
if len(kp1) <= len(kp2):
    num_keypoints = len(kp1)
else:
    num_keypoints = len(kp2)
print("Keypoints 1ST Image: " + str(len(kp1)))
print("Keypoints 2ND Image: " + str(len(kp2)))



# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

print('kp1', kp1[0].pt)
exit()

# Apply ratio test
good_points = []
for m,n in matches:

    if m.distance < 0.75 * n.distance:
        good_points.append([m])
        a=len(good_points)
        percent=(a*100)/len(kp2)
        print("{} % similarity".format(percent))
        if percent >= 75.00:
            print('Match Found')
        if percent < 75.00:
            print('Match not Found')

print("GOOD Matches:", len(good_points))
print("% of good feature matches: ", len(good_points) / num_keypoints * 100, "%")

img3 = cv2.drawMatchesKnn(gray1, kp1, gray3, kp2, good_points, None, flags=2)
plt.imshow(img3),plt.show()



