import numpy as np
import os
import cv2
from sklearn import preprocessing
import pandas as pd

class Data:
    def __init__(self, imagesPath=None, trainLabelsPath=None, testLabelsPath=None) -> None:
        self.imagesPath = imagesPath
        self.trainLabelsPath = trainLabelsPath
        self.testLabelsPath = testLabelsPath
        self.trainImages = None
        self.trainLabels = None
        self.validationImages = None
        self.validationLabels = None
        self.testImages = None
        self.outFile = None
        self.predicted_image_index = 17001
            
    def LoadData(self):
        filenames = os.listdir(self.imagesPath)
        trainImages = []
        for file in filenames:
            data = cv2.imread(self.imagesPath + file, cv2.IMREAD_GRAYSCALE)
            trainImages.append(data.flatten())
        
        columnTypes = {'id': str, 'class':int}
        
        dataframe = pd.read_csv(self.trainLabelsPath, dtype=columnTypes)
        trainLabels = dataframe['class'].values
        
        
        dataframe = pd.read_csv(self.testLabelsPath, dtype=columnTypes)
        validationLabels = dataframe['class'].values

        validationImages = trainImages[len(trainLabels):len(trainLabels) + len(validationLabels)]
        testImages = trainImages[len(trainLabels) + len(validationLabels):]
        trainImages = trainImages[:len(trainLabels)]

        return (trainImages, trainLabels, validationImages, validationLabels, testImages)
    
    
    def NormalizeData(self, trainImages, testImages, submitData, type=None):
        scaler = None
        if type == 'standard':
            scaler = preprocessing.StandardScaler()
        elif type == 'min_max':
            scaler = preprocessing.MinMaxScaler()
        elif type == 'l1':
            scaler = preprocessing.Normalizer(norm='l1')
        elif type == 'l2':
            scaler = preprocessing.Normalizer(norm='l2')
        if scaler is not None:
            scaler.fit(trainImages)
            trainImages = scaler.transform(trainImages)
            testImages = scaler.transform(testImages)
            submitData = scaler.transform(submitData) 
        return (trainImages, testImages, submitData)
    
    def OpenFile(self):
        self.outFile = open('submissions.csv', 'w')
        self.outFile.write('id,class\n')
        
    def CloseFile(self):
        self.outFile.close()
    
    def PrintData(self, predicted_values):
        for x in predicted_values:
            self.outFile.write(f'{self.predicted_image_index:06d},{int(x)}\n')
            self.predicted_image_index += 1