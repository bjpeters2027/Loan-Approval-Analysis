
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
class DataAnalysis:
    def __init__(self, trainedModel, dataHandler):
        self.model = trainedModel
        self.dataHandler = dataHandler

    # Plots the curve that compares the false and true positive rates
    def plotAccCurve(self, testX, testY):
        yProb = self.model.predictProb(testX)
        fpr, tpr, _ = roc_curve(testY, yProb)
        # This is the area under the curve. This represents a percent chance to properly classify or accuracy
        AccArea = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#006699', lw=2, label=f'Accuracy of the Curve = {AccArea:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title('True Positive v.s. False Positive Rates')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig("TPR-FPR-Curve")
        plt.show()
    
    # This predicts the minimum income for loan approval when all other factors are average
    def calculateIncomeThreshold(self):
        classifier = self.model.pipeline.named_steps['classifier']
        scaler = self.model.pipeline.named_steps['preprocessor'].named_transformers_['num']
        
        incomeIdx = self.dataHandler.numericalCols.index('annual_income')
        incomeCoef = classifier.coef_[0][incomeIdx]
        intercept = classifier.intercept_[0]

        incomeThresh = -intercept / incomeCoef

        scaledVars = np.zeros((1, len(self.dataHandler.numericalCols)))
        scaledVars[0, incomeIdx] = incomeThresh
        actualIncThresh = scaler.inverse_transform(scaledVars)[0, incomeIdx] * -1

        print(f"Minimum Annual Income for >50% approval: ${actualIncThresh :,.2f}")

    # This makes a bar graph showing likelihood of approval based on gender and marital status
    def demoBias(self, testX):
        testXAnalysis = testX.copy()
        testXAnalysis['predicted_approval'] = self.model.predict(testX)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.barplot(data=testXAnalysis, x='gender', y='predicted_approval', errorbar=None)
        plt.title('Approval Rate by Gender')
        plt.ylabel('Approval Rate')

        plt.subplot(1, 2, 2)
        sns.barplot(data=testXAnalysis, x='marital_status', y='predicted_approval', errorbar=None)
        plt.title('Approval Rate by Marital Status')
        plt.ylabel('')

        plt.tight_layout()
        plt.savefig("Approval-Rate-Gend-Marital")
        plt.show()

    # This plots the impact each feature has on approval
    def featureImportance(self):
        featureNames = self.model.getFeatures()
        coefficients = self.model.pipeline.named_steps['classifier'].coef_[0]

        importanceDf = pd.DataFrame({'Feature': featureNames, 'Impact': coefficients})
        importanceDf = importanceDf.sort_values(by='Impact', key=abs, ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importanceDf, x='Impact', y='Feature', hue='Impact', palette='vlag', legend=False)
        plt.title('Feature Impact')
        plt.xlabel('Effect on Approval Probability')
        plt.ylabel('Applicant Feature')
        plt.savefig("Feature-Impact")
        plt.show()