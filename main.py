from Model import Model
from DataAnalysis import DataAnalysis
from DataHandler import DataHandler
import pandas as pd

dataHandler = DataHandler('loanapproval.csv')
trainX, testX, trainY, testY = dataHandler.dataSplit()

model = Model(dataHandler.numericalCols, dataHandler.categoricalCols)
model.train(trainX, trainY)

analyzer = DataAnalysis(model, dataHandler)

analyzer.plotAccCurve(testX, testY)
analyzer.calculateIncomeThreshold()
analyzer.demoBias(pd.concat([trainX, testX]))
analyzer.featureImportance()