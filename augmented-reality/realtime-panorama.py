# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 

# import the necessary packages
from panorama.panorama import Stitcher
import imutils
import numpy as np
import os
import cv2

def cropFocus(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
	contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)
	crop = img[y:y+h,x:x+w]
	return crop


'''
firstImage = 'images/yosemite2.jpg'
secondImage = 'images/yosemite3.jpg'
'''

root_path = 'panorama-input/'
slash = '/'
root = os.listdir(root_path)

i = 0

imgPathList = []

# Creating list of paths
for imgPath in root:
	imgPathList.append(imgPath)

imgPathList.sort()

print imgPathList
for fn in imgPathList:
	print fn
	if i == 0:
		# This is our first image
		mainImage = cv2.imread(root_path + slash + fn)
		mainImage = imutils.resize(mainImage, width=400)
		#cv2.imwrite("mainImage.jpg", mainImage)
		i = i + 1
		continue
	else:
		# We shall combine current image with main image
		#mainImage = cv2.imread("mainImage.jpg")
		imageB = cv2.imread(root_path + slash + fn)
		imageB = imutils.resize(imageB, width=400)

		# stitch the images together to create a panorama
		stitcher = Stitcher()
		result =  stitcher.stitch([mainImage, imageB], showMatches=False)
		mainImage = cropFocus(result)

		# show the images
		
		cv2.imshow("Image A", mainImage)
		cv2.imshow("Image B", imageB)
		
		cv2.imwrite("result.jpg", result)

		cv2.imshow("Result", result)
		cv2.waitKey(0)
		i = i + 1

