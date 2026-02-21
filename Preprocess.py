import pandas as pd

class Preprocesser:
    def __init__(self, fileName: str):
        self.data = pd.read_csv(fileName)
    
    def dataSplit(self):
        df = self.data
        trainCut = round(0.8 * len(df))
        testCut = trainCut + 1
        X = df[["age","gender","marital_status","annual_income","loan_amount","credit_score","num_dependents","existing_loans_count","employment_status"]]
        Y = df["loan_approved"]
        trainX = X.iloc[0:trainCut]
        testX = X.iloc[testCut:len(df)]
        trainY = Y.iloc[0:trainCut]
        testY = Y.iloc[testCut:len(df)]
        return (testX, testY, trainX, trainY)

