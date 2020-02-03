# Face landmarks Detection
# usage:
# python facelandmarkdetect.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/face1.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import os
import imutils
from math import atan
import dlib
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import ndimage
from math import *
import operator
from PIL import Image

IMGSIZE = 500
inter = cv2.INTER_AREA

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return sqrt(dx*dx+dy*dy)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

if os.path.isfile(args["shape_predictor"]):
    pass
else:
    # print("Oops...! File is not available. Shall I downlaod ?")
    cmd = "wget -c --progress=bar http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    os.system(cmd)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale


widthImage = 500  # Size standarized
heightImage = 500  # Size standarized

imageFrontal = plt.imread("./images/vic0.jpg")

landmarksFrontal = plt.imread("./images/landmarks0.jpg")
landmarksImage = plt.imread("./images/landmarksImage.jpg")

imageInput = plt.imread(args["image"])
imageInputOrig = imageInput

#imageInput = cv2.resize(imageInput, (IMGSIZE,IMGSIZE) , interpolation = inter)
#imageFrontal = cv2.resize(imageFrontal, (IMGSIZE,IMGSIZE) , interpolation = inter)


gray = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY)

whiteImg = np.zeros(
    [imageInput.shape[0], imageInput.shape[1], 3], dtype=np.uint8)
whiteImg.fill(255)
# detect faces in the grayscale image
rects = detector(gray, 1)

np_eyes = []
# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(imageInput, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(imageInput, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    print("shape[0] :", shape[0])

    i = 0
    for (x, y) in shape:

        cv2.circle(imageInput, (x, y), 1, (0, 0, 255), -1)
        cv2.circle(whiteImg, (x, y), 3, (0, 0, 255), -1)
        if i == 36 or i==39 or i==42 or  i==45: # 36:left corner (Left eye) , 45:right corner(right eye)
            cv2.circle(whiteImg, (x, y), 3, (255, 0, 0), -1)
            np_eyes.append((x,y))
        i=i+1

numberstuple = (5,1,7,9,6,3)
divisor= 2.0
divisornodecimals = 2

tmp1 = tuple(map(operator.add, np_eyes[0], np_eyes[1]))
tmp2 = tuple(map(operator.add, np_eyes[2], np_eyes[3]))

eyeLeft = tuple(int(ti/2) for ti in tmp1)
eyeRight = tuple(int(ti/2) for ti in tmp2)

print("eyeLeft : ", eyeLeft)
print("eyeRight: ", eyeRight)
#angle = atan( eyeLeft[1] - eyeRight[1] ) / (eyeLeft[0] - eyeRight[0] )
eye_direction = (eyeRight[0] - eyeLeft[0], eyeRight[1] - eyeLeft[1])
# calc rotation angle in radians
rotation = -atan2(float(eye_direction[1]),float(eye_direction[0]))


degree=(rotation*180)/pi
print("rad: ", rotation)
print("degrade : ", degree)
#cv2.imwrite("landmarks0.jpg", cv2.cvtColor(whiteImg, cv2.COLOR_RGB2BGR))
#cv2.imwrite("landmarksImage.jpg", cv2.cvtColor(imageInput,cv2.COLOR_RGB2BGR))
#imgplot = plt.imshow(imageInput)

# show the output image with the face detections + facial landmarks

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = cos(angle)
  sine = sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = floor(float(offset_pct[0])*dest_sz[0])
  offset_v = floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image

imageVic = Image.open(args["image"])
newImage = ScaleRotateTranslate(imageVic, center=eyeLeft, angle=rotation)
#newImage = CropFace(imageVic, eye_left=(252,364), eye_right=(420,366), offset_pct=(0.1,0.1), dest_sz=(500,500))


fig = plt.figure(figsize=(20, 20))
a = fig.add_subplot(2, 3, 1)
imgplot = plt.imshow(imageFrontal)
a.set_title('Before')
plt.xticks([])
plt.yticks([])


a = fig.add_subplot(2, 3, 2)
imgplot = plt.imshow(landmarksFrontal)
a.set_title('Landmarks')
plt.xticks([])
plt.yticks([])


a = fig.add_subplot(2, 3, 3)
imgplot = plt.imshow(landmarksImage)
a.set_title('After')
plt.xticks([])
plt.yticks([])


# ------------------------------------
a = fig.add_subplot(2, 3, 4)
imgplot = plt.imshow(imageInputOrig)
a.set_title('Before')
plt.xticks([])
plt.yticks([])


a = fig.add_subplot(2, 3, 5)
imgplot = plt.imshow(whiteImg)
a.set_title('Landmarks')
plt.xticks([])
plt.yticks([])

imageInput = imutils.rotate(imageInput,45)
a = fig.add_subplot(2, 3, 6)
imgplot = plt.imshow(newImage)
a.set_title('After')
plt.xticks([])
plt.yticks([])

fname = "results/"+"result_" + args["image"][1]

plt.show()
'''


 

plt.imshow(imageInput)
plt.show()
'''
