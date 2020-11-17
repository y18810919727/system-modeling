#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch



import os
import cv2

filelist = [os.path.join('./pngs', file)for file in os.listdir('./pngs')]
filelist = sorted(filelist, key = lambda x:(len(x), x) )

def save_video(img_list, path):
    import cv2
    import os

    image_folder = 'images'
    video_name = path

    frame = cv2.imread(img_list[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in img_list:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

list(filter(lambda item:item.endswith('.png') and '01' in item,  filelist))
save_video(list(filter(lambda item:item.endswith('.png') and '01' in item,  filelist)), '01.avi')
save_video(list(filter(lambda item:item.endswith('.png') and '23' in item,  filelist)), '23.avi')