#!/usr/bin/python3
from scipy.misc import imread, imresize
import numpy as np
import sys

if len(sys.argv) != 2:
    print ('Usage python3 ./Convert_image_to_txt.py image_path')
    exit(0)

img = imread(sys.argv[1], mode='RGB')
img = imresize(img, (224, 224))
sys_bak = sys.stdout

sys.stdout = open ('./'+sys.argv[1]+'.txt','w')

for x in np.nditer(img):
	print(x)

sys.stdout = sys_bak








