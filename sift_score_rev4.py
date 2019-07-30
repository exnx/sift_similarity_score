import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
import cv2
import argparse
import os



# read two images from arg.parse

parser = argparse.ArgumentParser(description='sift match scorer')
parser.add_argument('--dir1')
parser.add_argument('--dir2')
parser.add_argument('--use_gray')
parser.add_argument('--x_lim', default=0.3, type=float)
parser.add_argument('--y_lim', default=0.3, type=float)
args = parser.parse_args()

# get dir names
dir1 = args.dir1
dir2 = args.dir2

# files list
files_dir1 = []
files_dir2 = []

# loop through dir 1 and find all file names, put file paths in both respective lists
for file in os.listdir('{}'.format(dir1)):

    if file.endswith(".png"):

        file1 = os.path.join(dir1, file)
        file2 = os.path.join(dir2, file)

        files_dir1.append(file1)
        files_dir2.append(file2)



# loop thru file list and grab the two images, and compare them

def compare(files_dir1, files_dir2):

    all_perc_matches = []
    total_num_unique_feats = 0

    for i in range(len(files_dir1)):

        # grab the image path in both dirs
        img_path_1 = files_dir1[i]
        img_path_2 = files_dir2[i]

        # read the imgs
        img1 = cv2.imread(img_path_1)
        img2 = cv2.imread(img_path_2)

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

        # convert to grayscale

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


        # # binary threshold
        # _, thresh1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
        # _, thresh2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)


        # cv2.imwrite('thresh1.jpg', gray1)
        # cv2.imwrite('thresh2.jpg', gray2)


        # detect sift features
        sift = cv2.xfeatures2d.SIFT_create()
        use_gray = args.use_gray

        if use_gray == 'True':
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
        # print("Keypoints 1ST Image: " + str(len(kp1)))
        # print("Keypoints 2ND Image: " + str(len(kp2)))


        # match sift features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # threshold for what is considered local in each direction of a feature match 
        x_local = args.x_lim
        y_local = args.y_lim

        # filter good matches by distance (L2 from discriptors), using ratio test

        char_num_unique_feats = 0
        char_spatial_matches = []  

        for m, n in matches:

            # n is the second closest match, so this measures the uniqueness of the m match
            if m.distance < 0.8 * n.distance:

                char_num_unique_feats += 1

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

                    # add the match points
                    char_spatial_matches.append([m])

        if char_num_unique_feats == 0:
            print('no good matches')
            exit()
        else:
            char_perc_good_matches = len(char_spatial_matches) / char_num_unique_feats * 100
            all_perc_matches.append(char_perc_good_matches)
            
        # perc_matches.append(perc_good_matches)

        print('Letter: {},   # unique feats: {},   # spatial matches: {},  {:.1f}% spatial matches'.format(os.path.basename(img_path_1), char_num_unique_feats, len(char_spatial_matches), char_perc_good_matches))

        if use_gray == 'True':
            img_combined = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, char_spatial_matches, None, flags=2)
        else:
            img_combined = cv2.drawMatchesKnn(img1, kp1, img2, kp2, char_spatial_matches, None, flags=2)


        # name of the file
        base = os.path.basename(img_path_1)
        file_name_only = os.path.splitext(base)[0]  # grab first part of file only

        comb_name = '{}_combined.jpg'.format(file_name_only)

        new_dir_name = '{}_{}'.format(dir1, dir2)

        if not os.path.exists(new_dir_name):
            os.makedirs(new_dir_name)

        # path of where to store it
        comb_img_path = new_dir_name + '/' + comb_name

        # write file
        cv2.imwrite(comb_img_path, img_combined)

        # plt.imshow(img_combined),plt.show()

        # update total stats
        total_num_unique_feats += char_num_unique_feats

    # plot stats for entire word
    print('Word stats:')
    print('Num unique features {}'.format(total_num_unique_feats))
    print('Avg {:.1f}% of matches'.format(sum(all_perc_matches)/len(all_perc_matches)))


# call compare with dir names
compare(files_dir1, files_dir2)



