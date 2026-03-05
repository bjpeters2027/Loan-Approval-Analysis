import pandas as pd
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        # The features are split into categorical and numerical categories.
        # This is important for the ML model beacuse we need to one-hot encode the categorical data
        self.categoricalCols = ['gender', 'marital_status', 'employment_status']
        self.numericalCols = ['age', 'annual_income', 'loan_amount', 'credit_score', 'num_dependents', 'existing_loans_count']
        
    # This function splits the data into traqining and testing sets.
    # Defaults to 80% training and 20% testing data
    def dataSplit(self, testSize=0.2):
        df = pd.read_csv(self.filepath)
        X = df.drop(columns=['applicant_id', 'loan_approved'])
        y = df['loan_approved']
        return train_test_split(X, y, test_size=testSize)
