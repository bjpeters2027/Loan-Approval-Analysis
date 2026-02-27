from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class Model:
    def __init__(self, numericalCols, categoricalCols):
        self.numericalCols = numericalCols
        self.categoricalCols = categoricalCols
        self.pipeline = self.buildPipeline()

    def buildPipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numericalCols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categoricalCols)
            ])
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=10000))
        ])

    def train(self, trainX, trainY):
        self.pipeline.fit(trainX, trainY)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predictProba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]
    
    def getFeatures(self):
        catEncoder = self.pipeline.named_steps['preprocessor'].named_transformers_['cat']
        rawFeatures = list(catEncoder.get_feature_names_out(self.categoricalCols))
        
        # Clean up the names by stripping out the 'column_name_' prefix
        cleanFeatures = []
        for feature in rawFeatures:
            for col in self.categoricalCols:
                if feature.startswith(col + '_'):
                    # Remove the prefix and keep just the category name (e.g., 'Employed')
                    cleanFeatures.append(feature.replace(col + '_', '', 1))
                    break
                    
        return self.numericalCols + cleanFeatures