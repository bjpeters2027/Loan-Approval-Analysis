from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class LRModel:
    def __init__(self, numericalCols, categoricalCols):
        self.numericalCols = numericalCols
        self.categoricalCols = categoricalCols
        self.pipeline = self.buildPipeline()

    # This function creates the sci-kit learn pipeline which represents a logistic regression classifier
    def buildPipeline(self): 
        # This uses the separated data in DataHandler.py and one-hot encodes the categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numericalCols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categoricalCols) 
            ])
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=10000))
        ])

    def train(self, trainX, trainY): # Trains ML model
        self.pipeline.fit(trainX, trainY)

    # This gives a discrete approved or rejected prediiction
    def predict(self, X): 
        return self.pipeline.predict(X)

    # This gives a contiuous approved or rejected probability
    def predictProb(self, X):
        return self.pipeline.predict_proba(X)[:, 1]
    
    # Returns the feature names for analyzing their impact
    def getFeatures(self):
        catEncoder = self.pipeline.named_steps['preprocessor'].named_transformers_['cat']
        rawFeatures = list(catEncoder.get_feature_names_out(self.categoricalCols))
        
        # Cleans up the names
        cleanFeatures = []
        for feature in rawFeatures:
            for col in self.categoricalCols:
                if feature.startswith(col + '_'):
                    cleanFeatures.append(feature.replace(col + '_', '', 1))
                    break
                    
        return self.numericalCols + cleanFeatures