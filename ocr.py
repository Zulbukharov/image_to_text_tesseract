# -*- coding: utf-8 -*-
from PIL import Image
import pytesseract
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the image")
args = vars(ap.parse_args())
img = Image.open(args["input"])
img.load()
text = pytesseract.image_to_string(img)
print(text)