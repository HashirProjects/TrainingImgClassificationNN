import os
import cv2
import random
import numpy as np
from tensorflow import keras

DIR = 'C:/Users/Ataul/OneDrive/Desktop/Programming/ML/KaggleAnimalRecognition/kagglecatsanddogs_3367a'
resolution=50
os.chdir(DIR)

def loadData(noOfImgs):
    
    dataSet=[]
    
    for category in os.listdir(DIR):
        
        categoryDir=os.path.join(DIR,category)

        counter=0
        
        for img in os.listdir(categoryDir):
            if counter < noOfImgs:
                
                imgArrayForm = cv2.imread(os.path.join(categoryDir,img),0)
                counter= counter+1
            
                try:
                    imgArrayForm= cv2.resize(imgArrayForm, (resolution,resolution))#lowers resolution to increase training speed and reduce memory problems
                    dataSet.append([imgArrayForm,category])
                except:
                    print(img)#just so i can see which images are broken
            else:
                break
                
    return dataSet

def createDataSet(noOfImages):

    DataSet=[]
    DataSet= loadData(noOfImages)
    random.shuffle(DataSet)
    
    Values=[]
    results=[]
    
    for x in DataSet:
        
        Values.append(x[0])
        if x[1] == 'Cat':
            results.append([0,1])
        else:
            results.append([1,0])
    
    results = np.array(results)
    Values = np.array(Values)
    Values= Values/255#makes all pixel values between 0 and 1 (helps to increase training speed if training values are closer together)
    
    return results, Values



#machine learning basics:
#J() is a cost function used to mesure the success of a model
#X(L) is a matrix of all the input variables
#theta(L) is a matrix of all the weights of all the nodes in each layer
#a() is an activation function
#Z(L) = a(X(L)transposed*theta(L)) gives the output for each layer, if it is the last layer this will give the final output
#J(Z(final layer)) is used to calculate how well this model fitted the data
#the method you use to update the weights to recieve a better model changes depending on the problem.
#in normal gradient descent, you backprop, using the derivatives to work out optimise each weight.
#tensorflow and keras allows you to basically ignore all this, but it helps knowing how this works when optimising your algorithm

def main():

    results, trainingValues= createDataSet(200)

    model = keras.Sequential()
    
    model.add(keras.layers.Flatten(input_shape=(resolution,resolution)))
    model.add(keras.layers.Dense(units=512, activation='relu' ))
    model.add(keras.layers.Dense(units=256, activation='relu' ))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(units=16, activation='relu'))
    model.add(keras.layers.Dense(units=2, activation='softmax'))
    #you can use binary classification too since there are only two classes

    model.compile(optimizer='sgd', loss='categorical_crossentropy')#i wrote a pretty large model  since the task is relatively complex

    model.fit(trainingValues, results, epochs=200)# you have to experiement and see if more epochs are reducing the loss more or more training examples

    testResults, testValues= createDataSet(2000)

    print(testResults)

    cv2.imshow("image", testValues[0])

    print(model(testValues))

    #then store weights and you got your model

    
main()
