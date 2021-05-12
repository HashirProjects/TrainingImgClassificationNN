import os
import cv2
import random
import numpy as np

DIR = 'C:/Users/Ataul/OneDrive/Desktop/Programming/ML/KaggleAnimalRecognition/kagglecatsanddogs_3367a'
os.chdir(DIR)

def loadData():
    resolution=50
    trainingSet=[]
    for category in os.listdir(DIR):
        categoryDir=os.path.join(DIR,category)
        for img in os.listdir(categoryDir):
            imgArrayForm = cv2.imread(os.path.join(categoryDir,img),0)
            try:
                imgArrayForm= cv2.resize(imgArrayForm, (resolution,resolution))#lowers resolution to increase training speed
                trainingSet.append([imgArrayForm,category])
            except:
                print(img)#just so i can see which images are broken
                
    return trainingSet

def createTrainingSet:
    
    random.shuffle(trainingSet)
    trainingSet= loadData(trainingSet)
    trainingValues=[]
    results=[]
    for i in trainingSet:
        trainingValues[i]= trainingSet[i][0]
        if trainingSet[i][1] == 'cat':
            results[i]= 0
        else:
            results[i]= 1
    return results, trainingValues
    

#machine learning basics:
#J() is a cost function used to mesure the success of a model
#X(L) is a matrix of all the input variables
#theta(L) is a matrix of all the weights of all the nodes in each layer
#a() is an activation function
#Z(L) = a(X(L)transposed*theta(L)) gives the output for each layer, if it is the last layer this will give the final output
#J(Z(final layer)) is used to calculate how well this model fitted the data
#the method you use to update the weights to recieve a better model changes what you do from here.
#tensorflow allows you to basically ignore all this, but it helps knowing how this works when optimising your algorithm
