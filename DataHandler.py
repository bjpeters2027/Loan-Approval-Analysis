import pandas as pd
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.categoricalCols = ['gender', 'marital_status', 'employment_status']
        self.numericalCols = ['age', 'annual_income', 'loan_amount', 'credit_score', 'num_dependents', 'existing_loans_count']
        
    def dataSplit(self, testSize=0.2, randomState=42):
        df = pd.read_csv(self.filepath)
        X = df.drop(columns=['applicant_id', 'loan_approved'])
        y = df['loan_approved']
        return train_test_split(X, y, test_size=testSize, random_state=randomState)
