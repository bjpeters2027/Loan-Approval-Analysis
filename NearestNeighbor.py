from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

class KNNModel:
    def __init__(self, numericalCols, categoricalCols, n_neighbors=5):
        self.numericalCols = numericalCols
        self.categoricalCols = categoricalCols
        self.n_neighbors = n_neighbors
        self.pipeline = self.buildPipeline()
    def buildPipeline(self): 
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numericalCols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categoricalCols) 
            ])
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=self.n_neighbors))
        ])

    def train(self, trainX, trainY): # Trains ML model
        self.pipeline.fit(trainX, trainY)

    def predict(self, X): 
        return self.pipeline.predict(X)
    def predictProb(self, X):
        return self.pipeline.predict_proba(X)[:, 1]
    
    def getFeatures(self):
        catEncoder = self.pipeline.named_steps['preprocessor'].named_transformers_['cat']
        rawFeatures = list(catEncoder.get_feature_names_out(self.categoricalCols))
        
        cleanFeatures = []
        for feature in rawFeatures:
            for col in self.categoricalCols:
                if feature.startswith(col + '_'):
                    cleanFeatures.append(feature.replace(col + '_', '', 1))
                    break
                    
        return self.numericalCols + cleanFeatures