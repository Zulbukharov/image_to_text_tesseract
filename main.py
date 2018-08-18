from PIL import Image
import pytesseract
import argparse
import cv2
import os

# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from pyimageserach import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    print("w")
    if len(approx) == 4:
        print ("ww")
        screenCnt = approx
        break

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
img = "image.png"
cv2.imwrite(img, imutils.resize(warped, height = 650))
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR`d")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
# args = vars(ap.parse_args())

#loading image and convert it to gratscale
image = cv2.imread(img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #check to see if we should apply thresholding to preprocess the image
# if args["preprocess"] == "thresh":
#     gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
# elif args["preprocess"] == "blur":
#     gray = cv2.medianBlur(gray, 3)

cv2.imwrite("image.png", gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
# cv2.imshow("im", "image.png")
text = pytesseract.image_to_string(Image.open("image.png"), lang="eng")
# os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", Image.open("image.png"))
cv2.imshow("Output", gray)
cv2.waitKey(0)