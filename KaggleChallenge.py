import os
import cv2
import random
import numpy as np

DIR ='C:/Users/Ataul/OneDrive/Documents/kagglecatsanddogs_3367a'
os.chdir(DIR)
trainingSet=[]

def loadData(DIR):
    for category in os.listdir(DIR):
        for img in os.listdir(os.path.join(DIR,category)):
            imgArrayForm = cv2.imread(os.path.join(os.path.join(DIR,category),img),0)
            imgArrayForm= cv2.resize(imgArrayForm, (50,50))
            trainingSet.append([imgArrayForm,category])

LoadData(DIR)

random.shuffle(trainingSet)
