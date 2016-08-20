#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob

# Load previously saved data

mtx = np.load('calib-matrix/mtx.npy')
dist = np.load('calib-matrix/dist.npy')

rect = (0, 0, 0, 0)
startPoint = False
endPoint = False

selectedPoint = False


def on_mouse(
    event,
    x,
    y,
    flags,
    params,
    ):

    global rect, startPoint, endPoint, selectedPoint

    # get mouse click

    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:

        # Resets and delete box once you are done

            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)

        if startPoint == False:

        # First click, waits for final click to create box

            rect = (x, y, 0, 0)
            startPoint = True
        elif endPoint == False:

        # creates the box (I think}

            rect = (rect[0], rect[1], x, y)
            print '________________'
            print 'Rectangle location: ', rect[0], ' ', rect[1], ' ', \
                x, ' ', y
            endPoint = True

    return


def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green

    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color

    for (i, j) in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), 255, 3)

    # draw top layer in red color

    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0),
                   5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0),
                   5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255),
                   5)
    return img


def detectAndDescribe(image):

    # convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect and extract features from the image

    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays

    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features

    return (kps, features)


def matchKeypoints(
    kpsA,
    kpsB,
    featuresA,
    featuresB,
    ratio,
    reprojThresh,
    ):

    # compute the raw matches and initialize the list of actual
    # matches

    matcher = cv2.DescriptorMatcher_create('BruteForce')
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    match_ar = []

    i = 0

    # loop over the raw matches
  
    for m in rawMatches:

        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)

        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            if i == 0:
                match_ar = np.array([[[m[0].trainIdx],
                                    [m[0].queryIdx]]], dtype=np.float)
                match_ar = match_ar.transpose()
                match_ar = list(match_ar)
                print type(match_ar)
                i = i + 1
            else:
                m_add = np.array([[m[0].trainIdx],
                                 [m[0].queryIdx]])[None, :]
                m_add = m_add.transpose()
                match_ar = np.concatenate([match_ar, m_add])
   
    
    # computing a homography requires at least 4 matches

    if len(matches) > 4:

        # construct the two sets of points

        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

        # return the matches along with the homograpy matrix
        # and status of each matched point
        # return (matches, H, status)

        return (matches, match_ar, H, status)

    # otherwise, no homograpy could be computed

    return None


def drawMatches(
    imageA,
    imageB,
    kpsA,
    kpsB,
    matches,
    status,
    ):

    # initialize the output visualization image

    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
    print imageA.shape
    print imageB.shape
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches

    for ((trainIdx, queryIdx), s) in zip(matches, status):

        # only process the match if the keypoint was successfully
        # matched

        if s == 1:

            # draw the match

            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization

    return vis


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
            0.001)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)



axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
cubeAxis = np.float32([
    [0, 0, 0],
    [0, 3, 0],
    [3, 3, 0],
    [3, 0, 0],
    [0, 0, -3],
    [0, 3, -3],
    [3, 3, -3],
    [3, 0, -3],
    ])

# Here we are going to try to define our corners

#fname = 'images/left01.jpg'
fname = 'images/checkerboard4.jpg'

img = cv2.imread(fname)

cv2.namedWindow('Label')
cv2.setMouseCallback('Label', on_mouse)

'''
while 1:
    if selectedPoint == True:
        break

    if startPoint == True and endPoint == True:
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]),
                      (255, 0, 255), 2)
    cv2.imshow('Label', img)
    if cv2.waitKey(20) & 255 == 27:
        break
cv2.destroyAllWindows()
'''


'''
reference_color = img[rect[1]:rect[3], rect[0]:rect[2]]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
reference_img = gray[rect[1]:rect[3], rect[0]:rect[2]]
'''
reference_color = img[101:400, 92:574]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
reference_img = gray[101:400, 92:574]


# ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
# corners2 = cv2.cornerHarris(gray,2,3,0.04)

####### Attempting to preform homography #######

sift = cv2.xfeatures2d.SIFT_create()

(kp1, des1) = detectAndDescribe(gray)
(kp2, des2) = detectAndDescribe(reference_img)


(M, M_ar, H, status) = matchKeypoints(kp1,kp2,des1,des2,0.75,4.0,)

if M is None:
    print 'No matches found'
    exit()

print 'Matches found'

# vis = drawMatches(gray, reference_img, kp1, kp2, M, status)

vis = drawMatches(img,reference_color,kp1,kp2,M,status)
cv2.imshow('vis', vis)
cv2.waitKey(0)

corners2 = M_ar

# corners = np.asarray(M)
'''
print corners2
print corners2.shape
'''
#corners = np.random.random((10,2,1))
#corners = np.float32([[rect[0], rect[1], 0], [rect[2], rect[1], 0], [rect[2], rect[3], 0], [rect[0], rect[3], 0]])



if 1 == 1:

# if ret == True:
    # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # Find the rotation and translation vectors.
    print 'here'
    (ret, rvecs, tvecs) = cv2.solvePnP(objp, corners2, mtx, dist)

    # project 3D points to image plane

    (imgpts, jac) = cv2.projectPoints(cubeAxis, rvecs, tvecs, mtx, dist)

    img = drawCube(img, corners2, imgpts)
    cv2.imshow('img', img)
    k = cv2.waitKey(0) & 255
else:
    print 'No corners were detected'


			
