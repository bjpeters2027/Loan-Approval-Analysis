from LogisticRegression import LRModel
from NearestNeighbor import KNNModel
from DataAnalysis import DataAnalysis
from DataHandler import DataHandler
import pandas as pd

dataHandler = DataHandler('loanapproval.csv')
trainX, testX, trainY, testY = dataHandler.dataSplit()

print("\n" + "="*50)
print("EVALUATING LOGISTIC REGRESSION")
print("="*50)
LRmodel = LRModel(dataHandler.numericalCols, dataHandler.categoricalCols)
LRmodel.train(trainX, trainY)

LRAnalyzer = DataAnalysis(LRmodel, dataHandler, "LogisticRegression")
LRAnalyzer.plotAccCurve(testX, testY)
LRAnalyzer.calculateIncomeThreshold(trainX)
LRAnalyzer.demoBias(pd.concat([trainX, testX]))
LRAnalyzer.featureImportance()

print("\n" + "="*50)
print("EVALUATING KKNN")
print("="*50)
KNNmodel = KNNModel(dataHandler.numericalCols, dataHandler.categoricalCols)
KNNmodel.train(trainX, trainY)

KNNAnalyzer = DataAnalysis(KNNmodel, dataHandler, "KNN")
KNNAnalyzer.plotAccCurve(testX, testY)
KNNAnalyzer.calculateIncomeThreshold(trainX)
KNNAnalyzer.demoBias(pd.concat([trainX, testX]))
KNNAnalyzer.featureImportance()