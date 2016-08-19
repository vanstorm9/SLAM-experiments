from panorama.panorama import Stitcher
import imutils
import numpy as np
import os
import cv2

widthResize = 600

def cropFocus(img):
	subtractThresh = 20


	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
	contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)
	crop = img[y:y+h,x:x+w-subtractThresh]
	return crop

def listDirCreate(root):
	for imgPath in root:
		imgPathList.append(imgPath)
	imgPathList.sort()
	return imgPathList

def drawHarris(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()]=[0,0,255]
	return img


root_path = 'panorama-input/'
slash = '/'
root = os.listdir(root_path)

i = 0
result = None
imgPathList = []

# Creating list of paths
listDirCreate = listDirCreate(root)

print imgPathList
for fn in imgPathList:
	print fn
	if i == 0:
		# This is our first image
		mainImage = cv2.imread(root_path + slash + fn)
		mainImage = imutils.resize(mainImage, width=widthResize)
		#cv2.imwrite("mainImage.jpg", mainImage)
		i = i + 1
		continue
	else:
		# We shall combine current image with main image
		#mainImage = cv2.imread("mainImage.jpg")
		imageB = cv2.imread(root_path + slash + fn)
		imageB = imutils.resize(imageB, width=widthResize)

		# stitch the images together to create a panorama
		stitcher = Stitcher()
		result, vis =  stitcher.stitch([mainImage, imageB], showMatches=False)
		mainImage = cropFocus(result)


		# show the images
		'''	
		cv2.imshow("Image A", mainImage)
		cv2.imshow("Image B", imageB)
		'''
		'''
		cv2.imwrite("result.jpg", result)

		cv2.imshow("Result", result)
		cv2.waitKey(0)
		'''
		i = i + 1


cv2.imwrite("result.jpg", result)

pointresult = result.copy()
pointresult = drawHarris(pointresult)

cv2.imshow("Result", result)
cv2.imshow("Points", pointresult)


cv2.waitKey(0)
