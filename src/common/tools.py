import yaml
import pickle

def load_config():
    # Read in the configuration file
    with open('config.yaml') as p:
        config = yaml.safe_load(p)
    return config

def pickle_dump(path,variable):
    # Serialize data from memory to file
    with open(path, 'wb') as handle:
        pickle.dump(variable, handle)

def pickle_load(path):
    # Read and load serialized data from file
    with open(path, 'rb') as handle:
        loaded = pickle.load(handle)
    return loaded