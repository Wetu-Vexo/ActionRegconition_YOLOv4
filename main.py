import numpy as np
import opencv as cv
import matplotlib.pyplot as plt
import os
import sys
import time
import math

# create folder for collection dataset
def create_folder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print("Error: Creating directory. " + folder_name)

create_folder('dataset')

#create 4 subfolder for each class
for i in range(4):
    create_folder('dataset/' + str(i))

# create dataset with opencv
# 30 images for each class
# total 120 images
# Path: main.ipynb

for i in range(4):
    cap = cv.VideoCapture(0)
    print('collecting images for class: ' + str(i))
    time.sleep(5)
    for j in range(30):
        ret, frame = cap.read()
        if ret == False:
            continue
        cv.imwrite('dataset/'+str(i)+'/'+str(j)+'.jpg', frame)
        cv.imshow('frame', frame)
        time.sleep(1)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
