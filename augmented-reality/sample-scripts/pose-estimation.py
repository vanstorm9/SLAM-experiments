import numpy as np
import cv2
import glob
 
# Load previously saved data
mtx = np.load('calib-matrix/mtx.npy')
dist = np.load('calib-matrix/dist.npy')

def drawCube(img, corners, imgpts):
	imgpts = np.int32(imgpts).reshape(-1,2)
 
	# draw ground floor in green
	img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
 
	# draw pillars in blue color
	for i,j in zip(range(4),range(4,8)):
		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
 
	# draw top layer in red color
	img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
 
	return img

def draw(img, corners, imgpts):
	corner = tuple(corners[0].ravel())
	img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
	img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
	img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
	return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)


axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
cubeAxis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])



#for fname in glob.glob('images/left*.jpg'):
#for fname in glob.glob('images/table*.jpg'):
for fname in glob.glob('images/*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
	
    print type(corners)
    print corners.shape
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
	# Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        # project 3D points to image plane
	imgpts, jac = cv2.projectPoints(cubeAxis, rvecs, tvecs, mtx, dist)

        img = drawCube(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        if k == ord('s'):
            cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()


