import scipy.sparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

class DataAnalysis:
    def __init__(self, trainedModel, dataHandler, modelName="Model"):
        self.model = trainedModel
        self.dataHandler = dataHandler
        self.modelName = modelName

    # Plots the curve that compares the false and true positive rates
    def plotAccCurve(self, testX, testY):
        yProb = self.model.predictProb(testX)
        fpr, tpr, _ = roc_curve(testY, yProb)
        AccArea = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#006699', lw=2, label=f'Area Under Curve = {AccArea:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title(f'True Positive v.s. False Positive Rates ({self.modelName})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(f"TPR-FPR-Curve-{self.modelName}")
        plt.show()
    
    # Predicts the minimum income for loan approval when all other factors are average
    def calculateIncomeThreshold(self, X):
        classifier = self.model.pipeline.named_steps['classifier']
        
        # Check if the model has coefficients
        if not hasattr(classifier, 'coef_'):
            print(f"\n--- Income Threshold ({self.modelName}) ---")
            print("Cannot calculate exact threshold: Distance based model")
            return None

        preprocessor = self.model.pipeline.named_steps['preprocessor']
        scaler = preprocessor.named_transformers_['num']
        
        baseline = pd.DataFrame(index=[0])
        for col in X.columns:
            if col in self.dataHandler.numericalCols:
                baseline[col] = X[col].mean()
            else:
                baseline[col] = X[col].mode()[0]
                
        newBase = preprocessor.transform(baseline)
        
        if scipy.sparse.issparse(newBase):
            newBase = newBase.toarray()
        newBase = newBase[0] 

        incomeIdx = self.dataHandler.numericalCols.index('annual_income')
        incomeCoef = classifier.coef_[0][incomeIdx]

        newBase[incomeIdx] = 0.0 
        
        base_log_odds = classifier.intercept_[0] + np.dot(classifier.coef_[0], newBase)
        scaled_income_thresh = -base_log_odds / incomeCoef
        scaledVars = np.zeros((1, len(self.dataHandler.numericalCols)))
        scaledVars[0, incomeIdx] = scaled_income_thresh
        actualIncThresh = scaler.inverse_transform(scaledVars)[0, incomeIdx]
        
        print(f"\n--- Income Threshold ({self.modelName}) ---")
        print(f"Minimum Annual Income for >50% approval: ${actualIncThresh:,.2f}")
        
        return actualIncThresh

    # Makes a bar graph showing likelihood of approval based on gender and marital status
    def demoBias(self, testX):
        testXAnalysis = testX.copy()
        testXAnalysis['predicted_approval'] = self.model.predict(testX)

        print(f"\n--- Approval Rates by Gender ({self.modelName}) ---")
        gender_rates = testXAnalysis.groupby('gender')['predicted_approval'].mean() * 100
        print(gender_rates.apply(lambda x: f"{x:.2f}%").to_string())

        print(f"\n--- Approval Rates by Marital Status ({self.modelName}) ---")
        marital_rates = testXAnalysis.groupby('marital_status')['predicted_approval'].mean() * 100
        print(marital_rates.apply(lambda x: f"{x:.2f}%").to_string())

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(data=testXAnalysis, x='gender', y='predicted_approval', errorbar=None)
        plt.title(f'Approval Rate by Gender ({self.modelName})')
        plt.ylabel('Approval Rate')

        plt.subplot(1, 2, 2)
        sns.barplot(data=testXAnalysis, x='marital_status', y='predicted_approval', errorbar=None)
        plt.title(f'Approval Rate by Marital Status ({self.modelName})')
        plt.ylabel('')

        plt.tight_layout()
        plt.savefig(f"Approval-Rate-Gend-Marital-{self.modelName}")
        plt.show()

    # Plots the impact each feature has on approval
    def featureImportance(self):
        classifier = self.model.pipeline.named_steps['classifier']
        
        if not hasattr(classifier, 'coef_'):
            print(f"\n--- Feature Impacts ({self.modelName}) ---")
            print("Cannot plot feature impact: Distance based model.")
            return None

        featureNames = self.model.getFeatures()
        coefficients = classifier.coef_[0]

        importanceDf = pd.DataFrame({'Feature': featureNames, 'Impact': coefficients})
        importanceDf = importanceDf.sort_values(by='Impact', key=abs, ascending=False)

        print(f"\n--- Feature Impacts ({self.modelName}) ---")
        print(importanceDf.to_string(index=False)) 

        plt.figure(figsize=(7, 4))
        sns.barplot(data=importanceDf, x='Impact', y='Feature', hue='Impact', palette='vlag', legend=False)
        plt.title(f'Feature Impact ({self.modelName})')
        plt.xlabel('Effect on Odds')
        plt.ylabel('Applicant Feature')
        plt.tight_layout()
        plt.savefig(f"Feature-Impact-{self.modelName}")
        plt.show()