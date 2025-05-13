import src.common.tools as tools
import src.data.dataio as dataio
from src.model import model


def train(config):
    # Load the data
    filepath = config["dataprocesseddirectory"] + "train.csv"
    [X, y] = dataio.load(filepath)

    # Train the model
    Model = model.Model()
    Model.train(X, y)

    # Save the trained model
    tools.pickle_dump(config["modelpath"], Model)


if __name__ == "__main__":
    config = tools.load_config()
    train(config)
