import os
import cv2
import random
import numpy as np
import tensorflow as tf

DIR = 'C:/Users/Ataul/OneDrive/Desktop/Programming/ML/KaggleAnimalRecognition/kagglecatsanddogs_3367a'
resolution=50
os.chdir(DIR)

def loadData():
    
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
                break
                
    return trainingSet

def createTrainingSet():

    trainingSet=[]
    trainingSet= loadData()
    
    trainingValues=[]
    results=[]
    
    for i in trainingSet:
        trainingValues.append(i[0])
        if i[1] == 'cat':
            results.append(0)
        else:
            results.append(1)

    random.shuffle(results)
    random.shuffle(trainingValues)
    
    results = np.array(results)
    trainingValues = np.array(trainingValues)
    
    return results, trainingValues

results, trainingValues= createTrainingSet()
    

#machine learning basics:
#J() is a cost function used to mesure the success of a model
#X(L) is a matrix of all the input variables
#theta(L) is a matrix of all the weights of all the nodes in each layer
#a() is an activation function
#Z(L) = a(X(L)transposed*theta(L)) gives the output for each layer, if it is the last layer this will give the final output
#J(Z(final layer)) is used to calculate how well this model fitted the data
#the method you use to update the weights to recieve a better model changes what you do from here.
#in normal gradient descent, you backprop, using the derivatives to work out optimise each weight.
#tensorflow allows you to basically ignore all this, but it helps knowing how this works when optimising your algorithm

def main():

    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.flatten())
    model.add(tf.keras.layers.Dense(units=100, input_shape=[resolution*resolution], activation='relu'))
    model.add(tf.keras.layers.Dense(units=12, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    

    model.compile(optimizer='sgd', loss='mean_squared_logarithmic_error')
