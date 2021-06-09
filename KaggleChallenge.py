import os
import cv2
import random
import numpy as np
from tensorflow import keras
import pickle

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
                    print(img)#just so I can see which images are broken
            else:
                break
            
    with open("C:/Users/Ataul/OneDrive/Desktop/Programming/ML/KaggleAnimalRecognition/ImgData.txt", "wb") as file:
        pickle.dump(dataSet, file)
                
    return dataSet

def createDataSet(ratio):

    DataSet=[]
    #DataSet= loadData(noOfImages)
    
    with open("C:/Users/Ataul/OneDrive/Desktop/Programming/ML/KaggleAnimalRecognition/ImgData.txt", "rb") as file:
        DataSet= pickle.load(file)
        
    random.shuffle(DataSet)
    
    Values=[]
    results=[]
    
    for x in DataSet:
        
        Values.append(x[0])
        if x[1] == 'Cat':
            results.append([0,1])
        else:
            results.append([1,0])
    
    testResults = np.array(results[int(len(results)*ratio):])
    trainingResults = np.array(results[:int(len(results)*ratio)])
    
    testValues = np.array(Values[int(len(Values)*ratio):])
    trainingValues = np.array(Values[:int(len(Values)*ratio)])

    
    
    trainingValues= trainingValues/255#makes all pixel values between 0 and 1 (helps to increase training speed if training values are closer together)
    testValues= testValues/255
    
    return trainingResults, trainingValues, testResults, testValues



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

    #noOfImgs=30000
    #loadData(noOfImgs)

    results, trainingValues, testResults, testValues= createDataSet(0.9)
    print (len(testValues))
    print (results)
    
    model = keras.Sequential()
    
    model.add(keras.layers.Flatten(input_shape=(resolution,resolution)))
    #model.add(keras.layers.Dense(units=1024, activation='relu' ))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(keras.layers.Dense(units=512, activation='relu' ))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=3750, activation='relu' ))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1000, activation='relu' ))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=100, activation='relu' ))
    model.add(keras.layers.Dense(units=2, activation='softmax'))

    
    #you can use binary classification too since there are only two classes
    initial_learning_rate = 0.01
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #initial_learning_rate,
        #decay_steps=100000,
        #decay_rate=0.96,
        #staircase=True)
    
    #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=initial_learning_rate), loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])#i wrote a pretty large model  since the task is relatively complex

    model.fit(trainingValues, results, epochs=1)# you have to experiement and see if more epochs are reducing the loss more
    model.save('C:/Users/Ataul/OneDrive/Desktop/Programming/ML/KaggleAnimalRecognition/KerasModel')

    cv2.imshow("img", testValues[0])

    print(model.evaluate(testValues, testResults))


    
main()
