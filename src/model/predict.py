# from src.model import model
from src.evaluate import evaluate
import src.common.tools as tools
import src.data.dataio as dataio

def predict(config):
    # Load the data to make predictions on
    filepath = config["dataprocesseddirectory"] + "test.csv"
    [X, y] = dataio.load(filepath)
    
    # Load the model
    modelpath = config["modelpath"]
    Model = tools.pickle_load(modelpath)
    
    # Make predictions from the trained model
    [y_hat, classes] = Model.predict(X)
    
    # Save results to a convenient data structure
    Result = evaluate.Results(y,y_hat,classes)
    resultspath = config["resultsrawpath"]
    tools.pickle_dump(resultspath,Result)

if __name__ == "__main__":
    config = tools.load_config()
    predict(config)