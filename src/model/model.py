from sklearn.ensemble import RandomForestClassifier

class Model:
    # This class provides an interface for the model (while this is not
    # strictly needed for a Random Forest classifier, it shows an example
    # of how the class could be constructed if the model is bespoke)
    def __init__(self) -> None:
        self.model = []
        self.initialize()
    
    def initialize(self):
        self.model = RandomForestClassifier()
    
    def train(self,X,y):
        self.model.fit(X,y)
        
    def predict_proba(self,X):
        prediction = self.model.predict_proba(X)
        classes = self.model.classes_
        return [prediction, classes]

    def predict(self,X):
        prediction = self.model.predict(X)
        classes = self.model.classes_
        return [prediction, classes]