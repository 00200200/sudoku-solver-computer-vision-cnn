from sklearn.metrics import accuracy_score, confusion_matrix
from src.common import tools

class Results:
    # A results class which calculates the predictions and metrics from a model
    # evaluation process
    def __init__(self,y_true,y_pred,classes) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes
        self.metrics = {}
    
    def get_metrics(self):
        self.metrics['confusion_matrix'] = confusion_matrix(self.y_true,self.y_pred)
        self.metrics["accuracy"] = accuracy_score(self.y_true,self.y_pred)

    def print_metrics(self):
        for key in self.metrics:
            print(f"{key} =\n {self.metrics[key]}")
        
if __name__ == "__main__":
    config = tools.load_config()
    
    # Load results
    resultspath = config["resultsrawpath"]
    Results = tools.pickle_load(resultspath)
    
    # Calculate metrics
    Results.get_metrics()
    Results.print_metrics()
    
    # Save metrics
    validationpath = config["resultsevaluatedpath"]
    tools.pickle_dump(validationpath, Results)
    